import argparse
import csv
import os
import statistics

from .data import find_resample_datasets
from .train import build_parser as build_train_parser
from .train import train_one
from .utils import ensure_dir, parse_int_ranges


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], datasets: list[str]) -> list[dict]:
    output = []
    for dataset in datasets:
        values = [float(row["test_acc"]) for row in rows if row["dataset"] == dataset and row["status"] == "ok"]
        output.append({
            "dataset": dataset,
            "num_ok": len(values),
            "test_acc_mean": f"{statistics.mean(values):.6f}" if values else "NA",
            "test_acc_std": f"{statistics.pstdev(values):.6f}" if len(values) > 1 else ("0.000000" if values else "NA"),
        })
    return output


def build_parser() -> argparse.ArgumentParser:
    train_parser = build_train_parser()
    parser = argparse.ArgumentParser(description="Run TRUST on all UCR datasets and resamples.")
    for action in train_parser._actions:
        if action.dest in {"help", "dataset", "resample_id", "output_dir"}:
            continue
        parser._add_action(action)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--datasets", default="", help="Comma-separated dataset subset. Default: scan data_root.")
    parser.add_argument("--exclude_datasets", default="", help="Comma-separated datasets to skip.")
    parser.add_argument("--resample_ids", default="0-29")
    parser.add_argument("--expected_datasets", type=int, default=112)
    parser.add_argument("--skip_existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_dir(args.output_dir)
    resample_ids = parse_int_ranges(args.resample_ids)
    datasets = find_resample_datasets(args.data_root, resample_id=resample_ids[0])
    if args.datasets:
        requested = [name.strip() for name in args.datasets.split(",") if name.strip()]
        datasets = [name for name in datasets if name in requested]
    if args.exclude_datasets:
        excluded = {name.strip() for name in args.exclude_datasets.split(",") if name.strip()}
        datasets = [name for name in datasets if name not in excluded]
    if len(datasets) != args.expected_datasets:
        print(f"Warning: found {len(datasets)} datasets, expected {args.expected_datasets}.")
    if not datasets:
        raise SystemExit(f"No datasets found in {args.data_root}")

    all_runs_path = os.path.join(args.output_dir, "all_runs.csv")
    rows: list[dict] = []
    if args.skip_existing and os.path.exists(all_runs_path):
        with open(all_runs_path, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    done = {(row["dataset"], int(row["resample_id"])) for row in rows if row.get("status") == "ok"}

    for dataset in datasets:
        for resample_id in resample_ids:
            if args.skip_existing and (dataset, resample_id) in done:
                continue
            run_args = argparse.Namespace(**vars(args))
            run_args.dataset = dataset
            run_args.resample_id = resample_id
            run_args.output_dir = os.path.join(args.output_dir, dataset, f"resample_{resample_id}")
            try:
                result = train_one(run_args)
                row = {
                    "dataset": dataset,
                    "resample_id": resample_id,
                    "status": "ok",
                    "test_acc": f"{float(result['test_acc']):.6f}",
                    "epoch": result["epoch"],
                    "selected_views": result["selected_views"],
                    "error": "",
                }
            except Exception as exc:
                row = {
                    "dataset": dataset,
                    "resample_id": resample_id,
                    "status": "error",
                    "test_acc": "NA",
                    "epoch": "",
                    "selected_views": "",
                    "error": repr(exc),
                }
            rows = [r for r in rows if not (r["dataset"] == dataset and int(r["resample_id"]) == resample_id)]
            rows.append(row)
            write_csv(
                all_runs_path,
                rows,
                ["dataset", "resample_id", "status", "test_acc", "epoch", "selected_views", "error"],
            )
            write_csv(
                os.path.join(args.output_dir, "summary.csv"),
                summarize(rows, datasets),
                ["dataset", "num_ok", "test_acc_mean", "test_acc_std"],
            )


if __name__ == "__main__":
    main()
