from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
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


def _print_grad(name: str, parameter: torch.nn.Parameter) -> None:
    grad = parameter.grad
    if grad is None:
        print(f"{name}: grad is None=True, grad mean=<missing>")
    else:
        print(f"{name}: grad is None=False, grad mean={grad.abs().mean().item():.8g}")


@hydra.main(version_base="1.3", config_path="../../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    _ensure_paths(cfg)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    batch = _first_train_batch(datamodule)

    net: torch.nn.Module = hydra.utils.instantiate(cfg.model.net)
    net.train()
    net.zero_grad(set_to_none=True)

    out = net(batch)
    loss = F.mse_loss(out, batch["path"])
    loss.backward()

    print("image_seq:", _shape(batch["image_seq"]))
    print("goal:", _shape(batch["goal"]))
    print("path:", _shape(batch["path"]))
    print("output:", tuple(out.shape))
    print("loss:", float(loss.detach().cpu()))

    expected_shape = batch["path"].shape
    assert tuple(out.shape) == tuple(expected_shape), (
        f"output shape {tuple(out.shape)} should match path shape "
        f"{tuple(expected_shape)}"
    )

    params = dict(net.named_parameters())
    for name in [
        "compression_queries",
        "view_queries",
        "goal_proj.weight",
        "out_proj.weight",
    ]:
        _print_grad(name, params[name])


if __name__ == "__main__":
    main()
