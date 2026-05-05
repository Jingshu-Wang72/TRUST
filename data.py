from __future__ import annotations

import os

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None
    Dataset = object

from .features import build_views, parse_views


def find_resample_datasets(data_root: str, resample_id: int = 0) -> list[str]:
    if not os.path.isdir(data_root):
        return []
    datasets = []
    for name in os.listdir(data_root):
        folder = os.path.join(data_root, name)
        if not os.path.isdir(folder):
            continue
        train_file = os.path.join(folder, f"{name}{resample_id}_TRAIN.ts")
        test_file = os.path.join(folder, f"{name}{resample_id}_TEST.ts")
        if os.path.exists(train_file) and os.path.exists(test_file):
            datasets.append(name)
    return sorted(datasets)


def resolve_ts_file(data_root: str, dataset: str, split: str, resample_id: int) -> str:
    split = split.upper()
    path = os.path.join(data_root, dataset, f"{dataset}{int(resample_id)}_{split}.ts")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing UCR resample file: {path}")
    return path


def parse_ts_file(path: str) -> tuple[np.ndarray, np.ndarray]:
    x_rows, y_rows = [], []
    in_data = False
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if not in_data:
                if line.lower() == "@data":
                    in_data = True
                continue
            series_s, label_s = line.rsplit(":", 1)
            x_rows.append([float(v) for v in series_s.split(",") if v.strip()])
            y_rows.append(label_s.strip())
    if not x_rows:
        raise ValueError(f"No samples found in {path}")
    lengths = {len(row) for row in x_rows}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent sequence lengths in {path}: {sorted(lengths)}")
    return np.asarray(x_rows, dtype=np.float32), np.asarray(y_rows)


def label_map_from_train(data_root: str, dataset: str, resample_id: int) -> dict[str, int]:
    _, y_train = parse_ts_file(resolve_ts_file(data_root, dataset, "TRAIN", resample_id))
    labels = sorted(set(y_train.tolist()))
    return {label: idx for idx, label in enumerate(labels)}


class UCRResampleDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset: str,
        split: str,
        resample_id: int,
        view_names: list[str],
        trend_window: int = 5,
        label_map: dict[str, int] | None = None,
        minirocket_num_kernels: int = 9996,
        minirocket_random_state: int = 42,
        multirocket_num_kernels: int = 10000,
        multirocket_random_state: int = 42,
        hydra_num_kernels: int = 8,
        hydra_n_groups: int = 64,
        hydra_max_num_channels: int = 8,
        hydra_random_state: int = 42,
        view_stats: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        self.data_root = data_root
        self.dataset = dataset
        self.split = split.upper()
        self.resample_id = int(resample_id)
        self.view_names = parse_views(",".join(view_names))

        x, y_raw = parse_ts_file(resolve_ts_file(data_root, dataset, self.split, self.resample_id))
        x_fit = x
        rocket_views = {"minirocket", "multirocket", "hydra"}
        if self.split != "TRAIN" and rocket_views.intersection({name.lower() for name in self.view_names}):
            x_fit, _ = parse_ts_file(resolve_ts_file(data_root, dataset, "TRAIN", self.resample_id))
        if label_map is None:
            label_map = {label: idx for idx, label in enumerate(sorted(set(y_raw.tolist())))}
        self.label_map = label_map
        self.y = np.asarray([label_map[label] for label in y_raw], dtype=np.int64)
        view_dict, self.view_stats = build_views(
            x,
            self.view_names,
            trend_window=trend_window,
            x_fit=x_fit,
            minirocket_num_kernels=minirocket_num_kernels,
            minirocket_random_state=minirocket_random_state,
            multirocket_num_kernels=multirocket_num_kernels,
            multirocket_random_state=multirocket_random_state,
            hydra_num_kernels=hydra_num_kernels,
            hydra_n_groups=hydra_n_groups,
            hydra_max_num_channels=hydra_max_num_channels,
            hydra_random_state=hydra_random_state,
            view_stats=view_stats,
        )
        self.views = [view_dict[name].astype(np.float32) for name in self.view_names]

    @property
    def n_classes(self) -> int:
        return len(self.label_map)

    @property
    def view_dims(self) -> list[int]:
        return [view.shape[1] for view in self.views]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        if torch is None:
            raise ModuleNotFoundError("PyTorch is required to materialize dataset tensors.")
        views = [torch.from_numpy(view[index]) for view in self.views]
        return views, torch.tensor(self.y[index], dtype=torch.long)
