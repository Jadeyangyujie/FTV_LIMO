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
        root_dir = Path(__file__).resolve().parents[4]
        cfg.paths.root_dir = str(root_dir)
    if not cfg.paths.get("data_dir"):
        cfg.paths.data_dir = str(Path(cfg.paths.root_dir) / "data")
    if not cfg.paths.get("log_dir"):
        cfg.paths.log_dir = str(Path(cfg.paths.root_dir) / "logs")


def _shape(value: Any) -> str:
    if torch.is_tensor(value):
        return str(tuple(value.shape))
    return type(value).__name__


def _first_train_batch(datamodule: LightningDataModule) -> dict[str, torch.Tensor]:
    datamodule.prepare_data()
    datamodule.setup("fit")
    loader = datamodule.train_dataloader()
    return next(iter(loader))


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    _ensure_paths(cfg)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    batch = _first_train_batch(datamodule)

    print("image_seq:", _shape(batch["image_seq"]))
    print("goal:", _shape(batch["goal"]))
    print("path:", _shape(batch["path"]))

    net: torch.nn.Module = hydra.utils.instantiate(cfg.model.net)
    net.eval()

    with torch.no_grad():
        out = net({"image_seq": batch["image_seq"], "goal": batch["goal"]})

    print("output:", tuple(out.shape))

    expected_shape = batch["path"].shape
    assert tuple(out.shape) == tuple(expected_shape), (
        f"output shape {tuple(out.shape)} should match path shape "
        f"{tuple(expected_shape)}"
    )
    assert tuple(out.shape) == (batch["image_seq"].shape[0], 50, 3), (
        "output should be [B, 50, 3]"
    )
    print("shape check: OK")


if __name__ == "__main__":
    main()
