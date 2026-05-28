from bisect import bisect_right
from pathlib import Path
from typing import Any

import hydra
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset


def _ensure_paths(cfg: DictConfig) -> None:
    if not cfg.get("paths"):
        return
    if not cfg.paths.get("root_dir"):
        root_dir = Path(__file__).resolve().parents[3]
        cfg.paths.root_dir = str(root_dir)
    if not cfg.paths.get("data_dir"):
        cfg.paths.data_dir = str(Path(cfg.paths.root_dir) / "data")
    if not cfg.paths.get("log_dir"):
        cfg.paths.log_dir = str(Path(cfg.paths.root_dir) / "logs")


def _shape(value: Any) -> str:
    if torch.is_tensor(value):
        return str(tuple(value.shape))
    if isinstance(value, list):
        return f"list[{len(value)}]"
    return type(value).__name__


def _print_tensor_stats(name: str, value: Any) -> None:
    if not torch.is_tensor(value):
        print(f"{name} dtype/min/max: <not a tensor>")
        return

    min_value = value.min().item()
    max_value = value.max().item()
    print(
        f"{name} dtype: {value.dtype}, min: {min_value:.6g}, max: {max_value:.6g}"
    )


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = dict(self.dataset[idx])
        sample["_dataset_index"] = idx
        return sample


def _resolve_dataset_index(dataset: Dataset, idx: int) -> tuple[Dataset, int]:
    if isinstance(dataset, ConcatDataset):
        dataset_idx = bisect_right(dataset.cumulative_sizes, idx)
        prev_size = 0 if dataset_idx == 0 else dataset.cumulative_sizes[dataset_idx - 1]
        child_idx = idx - prev_size
        return _resolve_dataset_index(dataset.datasets[dataset_idx], child_idx)

    if isinstance(dataset, Subset):
        return _resolve_dataset_index(dataset.dataset, dataset.indices[idx])

    return dataset, idx


def _get_temporal_indices(
    dataset: Dataset, idx: int, temporal_len: int, temporal_stride: int
) -> tuple[int, list[int]]:
    leaf_dataset, local_idx = _resolve_dataset_index(dataset, idx)
    if hasattr(leaf_dataset, "get_temporal_indices"):
        indices = leaf_dataset.get_temporal_indices(local_idx)
    else:
        indices = [
            max(local_idx - step * temporal_stride, 0)
            for step in range(temporal_len - 1, -1, -1)
        ]
    return local_idx, indices


def _debug_dataloader(
    datamodule: LightningDataModule, dataset: Dataset, split: str
) -> DataLoader:
    shuffle = getattr(datamodule, f"shuffle_{split}", False)
    return DataLoader(
        IndexedDataset(dataset),
        batch_size=getattr(datamodule, "batch_size", 1),
        shuffle=shuffle,
        num_workers=getattr(datamodule, "num_workers", 0),
        pin_memory=getattr(datamodule, "pin_memory", False),
    )


def _next_batch(datamodule: LightningDataModule) -> tuple[dict[str, Any], Dataset]:
    datamodule.prepare_data()
    datamodule.setup("fit")

    split_names = ["train", "val", "test"]
    for split in split_names:
        dataset = getattr(datamodule, f"data_{split}", None)
        if dataset is None:
            continue
        loader = _debug_dataloader(datamodule, dataset, split)
        try:
            return next(iter(loader)), dataset
        except StopIteration:
            continue

    raise RuntimeError("No non-empty dataloader is available for debug_image_seq")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    _ensure_paths(cfg)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    batch, source_dataset = _next_batch(datamodule)

    print("Dataset target:", cfg.dataset._target_)
    print("return_image_seq:", cfg.dataset.get("return_image_seq", False))
    print("temporal_len:", cfg.dataset.get("temporal_len", None))
    print("temporal_stride:", cfg.dataset.get("temporal_stride", None))
    camera_views = list(cfg.dataset.get("camera_views", []))
    print("camera_views:", camera_views)
    for view_idx, view_name in enumerate(camera_views):
        print(f"view {view_idx} = {view_name}")
    print("")

    keys = ["image_front", "image_left", "image_right", "image_seq", "goal", "path"]
    for key in keys:
        if key in batch:
            print(f"{key}: {_shape(batch[key])}")
        else:
            print(f"{key}: <missing>")

    print("")
    if "_dataset_index" in batch:
        temporal_len = int(cfg.dataset.get("temporal_len", 4))
        temporal_stride = int(cfg.dataset.get("temporal_stride", 1))
        dataset_indices = batch["_dataset_index"].tolist()
        print("temporal index debug:")
        for sample_pos, dataset_idx in enumerate(dataset_indices[:5]):
            local_idx, temporal_indices = _get_temporal_indices(
                source_dataset,
                int(dataset_idx),
                temporal_len=temporal_len,
                temporal_stride=temporal_stride,
            )
            print(
                f"sample {sample_pos}: dataset index={int(dataset_idx)}, "
                f"local index={local_idx}, temporal indices={temporal_indices}"
            )
    else:
        print("temporal index debug: <dataset indices missing>")

    print("")
    if "image_front" in batch:
        _print_tensor_stats("image_front", batch["image_front"])
    else:
        print("image_front dtype/min/max: <missing>")

    if "image_seq" in batch:
        _print_tensor_stats("image_seq", batch["image_seq"])
    else:
        print("image_seq dtype/min/max: <missing>")

    if "image_seq" in batch:
        image_seq = batch["image_seq"]
        assert (
            image_seq.ndim == 6
        ), "image_seq should be [B, T, V, C, H, W]"
        B, T, V, C, H, W = image_seq.shape
        assert C == 3, "image_seq should be [B, T, V, 3, H, W]"
        print(
            "image_seq shape check: "
            f"OK [B={B}, T={T}, V={V}, C={C}, H={H}, W={W}]"
        )

        current_front = image_seq[:, -1, 0]
        if "image_front" in batch:
            image_front = batch["image_front"]
            print(
                "image_seq[:, -1, 0] vs image_front shape: "
                f"{tuple(current_front.shape)} vs {tuple(image_front.shape)}"
            )
            if current_front.shape == image_front.shape:
                print("current front shape check: OK")
            else:
                print("current front shape check: FAILED")

            if (
                torch.is_tensor(image_front)
                and current_front.dtype == image_front.dtype
                and current_front.shape == image_front.shape
            ):
                diff = (current_front - image_front).abs().mean()
                print(
                    "mean absolute difference "
                    f"(image_seq[:, -1, 0] vs image_front): {diff.item():.8g}"
                )
            else:
                print(
                    "mean absolute difference "
                    "(image_seq[:, -1, 0] vs image_front): <skipped>"
                )
        else:
            print("image_seq[:, -1, 0] vs image_front shape: <image_front missing>")


if __name__ == "__main__":
    main()
