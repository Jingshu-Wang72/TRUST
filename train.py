import argparse
import csv
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import UCRResampleDataset, label_map_from_train
from .features import DEFAULT_VIEWS, parse_views
from .model import TRUST, evidential_ce_loss
from .utils import ensure_dir, set_seed


def collate_views(batch):
    view_lists, targets = zip(*batch)
    n_views = len(view_lists[0])
    views = [torch.stack([sample_views[i] for sample_views in view_lists], dim=0) for i in range(n_views)]
    return views, torch.stack(targets, dim=0)


@torch.no_grad()
def evaluate(model: TRUST, loader: DataLoader, device: torch.device, epoch: int) -> dict[str, float]:
    model.eval()
    y_true, y_pred = [], []
    losses = []
    for views, target in loader:
        views = [view.to(device) for view in views]
        target = target.to(device)
        _, fused_alpha = model(views, epoch=epoch)
        losses.append(float(evidential_ce_loss(target, fused_alpha, model.n_classes, epoch, 1).item()))
        pred = torch.argmax(fused_alpha, dim=1)
        y_true.extend(target.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())
    return {"loss": float(np.mean(losses)), "acc": float(accuracy_score(y_true, y_pred))}


def train_one(args: argparse.Namespace) -> dict[str, float | str | int]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    view_names = parse_views(args.views)

    labels = label_map_from_train(args.data_root, args.dataset, args.resample_id)
    dataset_kwargs = {
        "minirocket_num_kernels": args.minirocket_num_kernels,
        "minirocket_random_state": args.minirocket_random_state,
        "multirocket_num_kernels": args.multirocket_num_kernels,
        "multirocket_random_state": args.multirocket_random_state,
        "hydra_num_kernels": args.hydra_num_kernels,
        "hydra_n_groups": args.hydra_n_groups,
        "hydra_max_num_channels": args.hydra_max_num_channels,
        "hydra_random_state": args.hydra_random_state,
    }
    train_set = UCRResampleDataset(
        args.data_root, args.dataset, "TRAIN", args.resample_id, view_names,
        label_map=labels, **dataset_kwargs
    )
    test_set = UCRResampleDataset(
        args.data_root, args.dataset, "TEST", args.resample_id, view_names,
        label_map=labels, view_stats=train_set.view_stats, **dataset_kwargs
    )
    train_loader = DataLoader(
        train_set,
        batch_size=min(args.batch_size, len(train_set)),
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_views,
        drop_last=len(train_set) > 1,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=min(args.eval_batch_size, len(test_set)),
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_views,
    )

    model = TRUST(
        view_dims=train_set.view_dims,
        n_classes=train_set.n_classes,
        hidden=args.hidden,
        top_k=args.top_k,
        use_pseudo=not args.no_pseudo,
        use_pseudo_in_fusion=not args.no_pseudo_in_fusion,
        min_warmup_epochs=args.min_warmup_epochs,
        stable_topk_steps=args.stable_topk_steps,
        max_warmup_epochs=args.max_warmup_epochs,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ensure_dir(args.output_dir)
    history_path = os.path.join(args.output_dir, "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "selected_views"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses = []
            progress = tqdm(train_loader, desc=f"{args.dataset} r{args.resample_id} epoch {epoch}", leave=False)
            for views, target in progress:
                views = [view.to(device) for view in views]
                target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                view_alphas, fused_alpha = model(views, epoch=epoch)
                losses = [evidential_ce_loss(target, alpha, model.n_classes, epoch, args.annealing_epoch) for alpha in view_alphas]
                losses.append(evidential_ce_loss(target, fused_alpha, model.n_classes, epoch, args.annealing_epoch))
                loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))
                progress.set_postfix(loss=f"{np.mean(train_losses):.4f}")

            selected = "|".join(view_names[i] for i in (model.selected_views or []))
            writer.writerow({
                "epoch": epoch,
                "train_loss": f"{np.mean(train_losses):.6f}",
                "selected_views": selected,
            })
            handle.flush()

    if args.save_model:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "model_final.pt"))

    final_metrics = evaluate(model, test_loader, device, args.epochs)
    with open(os.path.join(args.output_dir, "test_metrics.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "test_loss", "test_acc"])
        writer.writeheader()
        writer.writerow({
            "epoch": args.epochs,
            "test_loss": f"{final_metrics['loss']:.6f}",
            "test_acc": f"{final_metrics['acc']:.6f}",
        })

    return {
        "dataset": args.dataset,
        "resample_id": int(args.resample_id),
        "epoch": int(args.epochs),
        "test_acc": float(final_metrics["acc"]),
        "selected_views": "|".join(view_names[i] for i in (model.selected_views or [])),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train TRUST on one UCR resample.")
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--resample_id", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--views", default=DEFAULT_VIEWS)
    parser.add_argument("--minirocket_num_kernels", type=int, default=9996)
    parser.add_argument("--minirocket_random_state", type=int, default=42)
    parser.add_argument("--multirocket_num_kernels", type=int, default=10000)
    parser.add_argument("--multirocket_random_state", type=int, default=42)
    parser.add_argument("--hydra_num_kernels", type=int, default=8)
    parser.add_argument("--hydra_n_groups", type=int, default=64)
    parser.add_argument("--hydra_max_num_channels", type=int, default=8)
    parser.add_argument("--hydra_random_state", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--hidden", nargs="*", type=int, default=[128, 64])
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--annealing_epoch", type=int, default=20)
    parser.add_argument("--min_warmup_epochs", type=int, default=5)
    parser.add_argument("--stable_topk_steps", type=int, default=3)
    parser.add_argument("--max_warmup_epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_pseudo", action="store_true")
    parser.add_argument("--no_pseudo_in_fusion", action="store_true")
    parser.add_argument("--save_model", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_one(args)
    print(result)


if __name__ == "__main__":
    main()
