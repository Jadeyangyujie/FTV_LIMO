from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import wandb
import numpy as np
import yaml
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.regression import MeanAbsoluteError

from limo.src.utils.visualization import create_combined_visualization


class LimoModel(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        compile: bool,
        camera_info: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.loss = loss

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.train_mae = MeanAbsoluteError()
        self.val_mae = MeanAbsoluteError()
        self.test_mae = MeanAbsoluteError()

        self.val_loss_best = MinMetric()
        self.camera_info = self._load_camera_info(camera_info)

    def _load_camera_info(
        self, camera_info_path: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        if camera_info_path is None:
            return None

        path = Path(camera_info_path)
        if not path.exists():
            self.print(
                f"camera_info file not found: {camera_info_path}. Falling back to image-only visualization."
            )
            return None

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.net(batch)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_mae.reset()
        self.val_loss_best.reset()

    def model_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = batch["path"]
        preds = self(batch)
        loss = self.loss(preds, path)
        return loss, preds, path

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:

        if batch_idx < 3 and "image_seq" in batch:
            p = next(self.parameters())
            print(
                f"[DDP DEBUG] "
                f"global_rank={self.global_rank}, "
                f"local_rank={self.local_rank}, "
                f"batch_idx={batch_idx}, "
                f"param_device={p.device}, "
                f"image_seq_device={batch['image_seq'].device}, "
                f"goal_device={batch['goal'].device}, "
                f"cuda_mem={torch.cuda.memory_allocated() / 1024**2:.1f} MiB",
                flush=True,
            )

            
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_mae(preds, targets)

        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True,
        )
        self.log(
            "train/mae", self.train_mae, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True,
        )
        self.log(
            "learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=False, prog_bar=True,sync_dist=False,
        )

        return {"loss": loss, "preds": preds}

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_mae(preds, targets)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        self.log("val/mae", self.val_mae, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)

        return {"loss": loss, "preds": preds}

    def predict_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        preds = self(batch)
        uuids = batch["uuid"]
        paths = preds.detach().cpu().numpy()
        return {"uuid": uuids, "path_pred": paths}

    def on_validation_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if self.trainer.is_global_zero and batch_idx == 0 and wandb.run is not None:
            self.log_images_wandb(outputs, batch, split="val")

    def on_train_batch_end(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        if self.trainer.is_global_zero and batch_idx == 0 and wandb.run is not None:
            self.log_images_wandb(outputs, batch, split="train")

    def on_validation_epoch_end(self) -> None:
        cur_val_loss = self.val_loss.compute()
        self.val_loss_best(cur_val_loss)
        self.log(
            "val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True
        )

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_mae(preds, targets)

        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True
        )
        self.log("test/mae", self.test_mae, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)

    def setup(self, stage: str) -> None:
        self.net.setup()
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            total_steps = self.trainer.estimated_stepping_batches
            scheduler = self.hparams.scheduler(
                optimizer=optimizer, total_steps=total_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @staticmethod
    def _to_image_np(tensor: torch.Tensor) -> np.ndarray:
        """Convert [B, C, H, W] float images to uint8 [B, H, W, C]."""
        tensor = tensor.detach().float().cpu().clamp(0.0, 1.0)
        return (tensor.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)

    @staticmethod
    def _make_grid(images: np.ndarray, pad: int = 8) -> np.ndarray:
        """Create a simple image grid from [rows, cols, H, W, C]."""
        rows, cols, height, width, channels = images.shape
        grid_h = rows * height + max(rows - 1, 0) * pad
        grid_w = cols * width + max(cols - 1, 0) * pad
        grid = np.ones((grid_h, grid_w, channels), dtype=np.uint8) * 255

        for row in range(rows):
            y0 = row * (height + pad)
            for col in range(cols):
                x0 = col * (width + pad)
                grid[y0 : y0 + height, x0 : x0 + width] = images[row, col]
        return grid

    def _make_sequence_images(
        self,
        image_seq: torch.Tensor,
        num_imgs: int,
    ) -> list[wandb.Image]:
        """Log FTV image_seq as rows=time and columns=view."""
        image_seq = image_seq[:num_imgs].detach().float().cpu().clamp(0.0, 1.0)
        if image_seq.ndim != 6:
            return []

        sequence_images = []
        for sample_idx in range(image_seq.shape[0]):
            sample = image_seq[sample_idx]  # [T, V, C, H, W]
            sample_np = (
                sample.permute(0, 1, 3, 4, 2).numpy() * 255
            ).astype(np.uint8)
            grid = self._make_grid(sample_np)
            caption = (
                f"sample={sample_idx}, rows=time old->current, "
                "cols=view order from camera_views"
            )
            sequence_images.append(wandb.Image(grid, caption=caption))
        return sequence_images

    @staticmethod
    def _make_path_error_table(
        predicted_paths: np.ndarray,
        ground_truth_paths: np.ndarray,
        goals: np.ndarray,
    ) -> wandb.Table:
        columns = [
            "sample",
            "mean_xy_error",
            "final_xy_error",
            "mean_abs_theta_error",
            "goal_x",
            "goal_y",
            "goal_theta",
            "pred_final_x",
            "pred_final_y",
            "pred_final_theta",
            "target_final_x",
            "target_final_y",
            "target_final_theta",
        ]
        table = wandb.Table(columns=columns)

        xy_error = np.linalg.norm(
            predicted_paths[..., :2] - ground_truth_paths[..., :2], axis=-1
        )
        theta_error = np.abs(predicted_paths[..., 2] - ground_truth_paths[..., 2])

        for sample_idx in range(predicted_paths.shape[0]):
            pred_final = predicted_paths[sample_idx, -1]
            target_final = ground_truth_paths[sample_idx, -1]
            goal = goals[sample_idx]
            table.add_data(
                sample_idx,
                float(xy_error[sample_idx].mean()),
                float(xy_error[sample_idx, -1]),
                float(theta_error[sample_idx].mean()),
                float(goal[0]),
                float(goal[1]),
                float(goal[2]),
                float(pred_final[0]),
                float(pred_final[1]),
                float(pred_final[2]),
                float(target_final[0]),
                float(target_final[1]),
                float(target_final[2]),
            )
        return table

    def log_images_wandb(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        split: str,
        num_imgs: int = 6,
    ):
        """Log predicted paths as combined visualizations to wandb."""
        assert wandb.run is not None, "This can only be used with wandb active"

        # Get predictions, ground truth, and goals
        predicted_paths = outputs["preds"][:num_imgs].detach().cpu().numpy()
        ground_truth_paths = batch["path"][:num_imgs].detach().cpu().numpy()
        goals_tensor = batch["goal"][:num_imgs]
        goals = goals_tensor.detach().cpu().numpy()

        # Get images. FTV batches can omit duplicated current-frame images to
        # reduce dataloader and device-transfer overhead.
        if "image_front" in batch:
            images = batch["image_front"][:num_imgs]
        elif "image_seq" in batch:
            images = batch["image_seq"][:num_imgs, -1, 0]
        else:
            raise KeyError("batch must contain image_front or image_seq for logging")

        images = self._to_image_np(images)
        images_left = (
            self._to_image_np(batch["image_left"][:num_imgs])
            if "image_left" in batch
            else None
        )
        images_right = (
            self._to_image_np(batch["image_right"][:num_imgs])
            if "image_right" in batch
            else None
        )

        if "image_seq" in batch and batch["image_seq"].ndim == 6:
            current_views = batch["image_seq"][:num_imgs, -1]
            if images_left is None and current_views.shape[1] > 1:
                images_left = self._to_image_np(current_views[:, 1])
            if images_right is None and current_views.shape[1] > 2:
                images_right = self._to_image_np(current_views[:, 2])

        visualizations = []
        for i in range(len(predicted_paths)):
            combined_img = create_combined_visualization(
                images[i],
                [ground_truth_paths[i], predicted_paths[i]],
                goals[i : i + 1],
                camera_info=self.camera_info,
                image_left=images_left[i] if images_left is not None else None,
                image_right=images_right[i] if images_right is not None else None,
            )
            visualizations.append(wandb.Image(combined_img))

        xy_error = np.linalg.norm(
            predicted_paths[..., :2] - ground_truth_paths[..., :2], axis=-1
        )
        theta_error = np.abs(predicted_paths[..., 2] - ground_truth_paths[..., 2])

        log_payload: Dict[str, Any] = {
            f"{split}/predictions": visualizations,
            f"{split}/path_error_table": self._make_path_error_table(
                predicted_paths, ground_truth_paths, goals
            ),
            f"{split}/viz_mean_xy_error": float(xy_error.mean()),
            f"{split}/viz_final_xy_error": float(xy_error[:, -1].mean()),
            f"{split}/viz_mean_abs_theta_error": float(theta_error.mean()),
        }

        if "image_seq" in batch:
            sequence_images = self._make_sequence_images(
                batch["image_seq"], num_imgs=min(3, num_imgs)
            )
            if sequence_images:
                log_payload[f"{split}/image_sequence_grid"] = sequence_images

        wandb.log(log_payload)
