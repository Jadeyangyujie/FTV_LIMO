from pathlib import Path
from typing import Any

import hydra
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig


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


def _next_batch(datamodule: LightningDataModule) -> dict[str, Any]:
    datamodule.prepare_data()
    datamodule.setup("fit")

    loader_names = ["train_dataloader", "val_dataloader", "test_dataloader"]
    for loader_name in loader_names:
        try:
            loader = getattr(datamodule, loader_name)()
        except RuntimeError:
            continue
        try:
            return next(iter(loader))
        except StopIteration:
            continue

    raise RuntimeError("No non-empty dataloader is available for debug_image_seq")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    _ensure_paths(cfg)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    batch = _next_batch(datamodule)

    print("Dataset target:", cfg.dataset._target_)
    print("return_image_seq:", cfg.dataset.get("return_image_seq", False))
    print("temporal_len:", cfg.dataset.get("temporal_len", None))
    print("temporal_stride:", cfg.dataset.get("temporal_stride", None))
    print("camera_views:", list(cfg.dataset.get("camera_views", [])))
    print("")

    keys = ["image_front", "image_left", "image_right", "image_seq", "goal", "path"]
    for key in keys:
        if key in batch:
            print(f"{key}: {_shape(batch[key])}")
        else:
            print(f"{key}: <missing>")

    if "image_seq" in batch:
        assert (
            batch["image_seq"].ndim == 6
        ), "image_seq should be [B, T, V, C, H, W]"


if __name__ == "__main__":
    main()
