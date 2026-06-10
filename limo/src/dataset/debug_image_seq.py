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
        # debug_image_seq.py is expected at:
        # limo/src/dataset/debug_image_seq.py
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
    """
    Resolve global index inside nested ConcatDataset / Subset to leaf dataset
    and its local index.
    """
    if isinstance(dataset, ConcatDataset):
        dataset_idx = bisect_right(dataset.cumulative_sizes, idx)
        prev_size = 0 if dataset_idx == 0 else dataset.cumulative_sizes[dataset_idx - 1]
        child_idx = idx - prev_size
        return _resolve_dataset_index(dataset.datasets[dataset_idx], child_idx)

    if isinstance(dataset, Subset):
        return _resolve_dataset_index(dataset.dataset, dataset.indices[idx])

    return dataset, idx


def _leaf_dataset_desc(leaf_dataset: Dataset) -> str:
    dataset_type = getattr(leaf_dataset, "dataset_type", None)
    mission_name = getattr(leaf_dataset, "mission_name", None)
    image_id_key = getattr(leaf_dataset, "image_id_key", None)

    parts = [type(leaf_dataset).__name__]
    if dataset_type is not None:
        parts.append(f"type={dataset_type}")
    if mission_name is not None:
        parts.append(f"mission={mission_name}")
    if image_id_key is not None:
        parts.append(f"image_id_key={image_id_key}")

    return ", ".join(parts)


def _get_current_image_id(leaf_dataset: Dataset, local_idx: int) -> int | None:
    if hasattr(leaf_dataset, "get_image_id"):
        return int(leaf_dataset.get_image_id(local_idx))

    if hasattr(leaf_dataset, "image_ids"):
        return int(leaf_dataset.image_ids[local_idx])

    return None


def _get_actual_temporal_indices_and_image_ids(
    leaf_dataset: Dataset,
    local_idx: int,
    temporal_len: int,
    temporal_stride: int,
) -> tuple[list[int], list[int] | None]:
    """
    actual = the sample indices/image_ids actually used by the current Dataset
    to load image_seq.
    """
    if hasattr(leaf_dataset, "get_temporal_indices"):
        temporal_indices = list(leaf_dataset.get_temporal_indices(local_idx))
    else:
        temporal_indices = [
            max(local_idx - step * temporal_stride, 0)
            for step in range(temporal_len - 1, -1, -1)
        ]

    if hasattr(leaf_dataset, "get_image_id"):
        temporal_image_ids = [
            int(leaf_dataset.get_image_id(i)) for i in temporal_indices
        ]
    elif hasattr(leaf_dataset, "image_ids"):
        temporal_image_ids = [
            int(leaf_dataset.image_ids[i]) for i in temporal_indices
        ]
    else:
        temporal_image_ids = None

    return temporal_indices, temporal_image_ids


def _get_expected_temporal_image_ids_by_image_id(
    leaf_dataset: Dataset,
    local_idx: int,
    temporal_len: int,
    temporal_stride: int,
) -> list[int] | None:
    """
    expected = what temporal image ids should be if we roll back by image_id.

    Example:
        current image_id = 100
        temporal_len = 4
        temporal_stride = 1
        expected = [97, 98, 99, 100]

    At segment starts, repeated padding is allowed.
    """
    current_image_id = _get_current_image_id(leaf_dataset, local_idx)
    if current_image_id is None:
        return None

    if hasattr(leaf_dataset, "segment_start_indices"):
        segment_start_idx = int(leaf_dataset.segment_start_indices[local_idx])
        segment_start_image_id = _get_current_image_id(leaf_dataset, segment_start_idx)
    else:
        segment_start_image_id = _get_current_image_id(leaf_dataset, 0)

    if segment_start_image_id is None:
        return None

    return [
        max(current_image_id - step * temporal_stride, segment_start_image_id)
        for step in range(temporal_len - 1, -1, -1)
    ]


def _temporal_status(
    actual_image_ids: list[int] | None,
    expected_image_ids: list[int] | None,
) -> str:
    if actual_image_ids is None or expected_image_ids is None:
        return "UNKNOWN_NO_IMAGE_IDS"

    if actual_image_ids == expected_image_ids:
        return "OK"

    actual_unique = len(set(actual_image_ids))
    expected_unique = len(set(expected_image_ids))

    if actual_unique < expected_unique:
        return "BAD_REPEATED_BY_SAMPLE_IDX"

    return "BAD_MISMATCH"


def _print_temporal_debug_record(
    source_dataset: Dataset,
    dataset_idx: int,
    sample_pos: int | None,
    temporal_len: int,
    temporal_stride: int,
) -> str:
    leaf_ds, local_idx = _resolve_dataset_index(source_dataset, dataset_idx)

    actual_indices, actual_image_ids = _get_actual_temporal_indices_and_image_ids(
        leaf_ds,
        local_idx,
        temporal_len=temporal_len,
        temporal_stride=temporal_stride,
    )

    expected_image_ids = _get_expected_temporal_image_ids_by_image_id(
        leaf_ds,
        local_idx,
        temporal_len=temporal_len,
        temporal_stride=temporal_stride,
    )

    current_image_id = _get_current_image_id(leaf_ds, local_idx)
    status = _temporal_status(actual_image_ids, expected_image_ids)

    prefix = f"sample {sample_pos}: " if sample_pos is not None else ""

    print(
        f"{prefix}"
        f"status={status}, "
        f"dataset index={dataset_idx}, "
        f"local index={local_idx}, "
        f"current image_id={current_image_id}, "
        f"actual sample indices={actual_indices}, "
        f"actual image_ids={actual_image_ids}, "
        f"expected image_ids_by_image_id={expected_image_ids}, "
        f"source=({_leaf_dataset_desc(leaf_ds)})"
    )

    return status


def _scan_temporal_consistency(
    source_dataset: Dataset,
    temporal_len: int,
    temporal_stride: int,
    max_scan: int = 200,
    max_bad_print: int = 30,
) -> None:
    print("")
    print(
        "temporal consistency scan: "
        f"scan first {min(max_scan, len(source_dataset))} samples"
    )

    status_counts: dict[str, int] = {}
    bad_printed = 0

    for dataset_idx in range(min(max_scan, len(source_dataset))):
        leaf_ds, local_idx = _resolve_dataset_index(source_dataset, dataset_idx)

        actual_indices, actual_image_ids = _get_actual_temporal_indices_and_image_ids(
            leaf_ds,
            local_idx,
            temporal_len=temporal_len,
            temporal_stride=temporal_stride,
        )

        expected_image_ids = _get_expected_temporal_image_ids_by_image_id(
            leaf_ds,
            local_idx,
            temporal_len=temporal_len,
            temporal_stride=temporal_stride,
        )

        status = _temporal_status(actual_image_ids, expected_image_ids)
        status_counts[status] = status_counts.get(status, 0) + 1

        if status != "OK" and bad_printed < max_bad_print:
            current_image_id = _get_current_image_id(leaf_ds, local_idx)
            print(
                f"BAD sample: "
                f"status={status}, "
                f"dataset index={dataset_idx}, "
                f"local index={local_idx}, "
                f"current image_id={current_image_id}, "
                f"actual sample indices={actual_indices}, "
                f"actual image_ids={actual_image_ids}, "
                f"expected image_ids_by_image_id={expected_image_ids}, "
                f"source=({_leaf_dataset_desc(leaf_ds)})"
            )
            bad_printed += 1

    print("")
    print("temporal consistency scan summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")


def _find_non_padding_temporal_sample(
    source_dataset: Dataset,
    temporal_len: int,
    temporal_stride: int,
    max_scan: int = 2000,
) -> int | None:
    """
    Find a sample whose temporal image_ids contain at least two different frames.

    This avoids checking only the first few segment-start samples, where repeated
    padding like [51, 51, 51, 51] is expected and correct.
    """
    for dataset_idx in range(min(max_scan, len(source_dataset))):
        leaf_ds, local_idx = _resolve_dataset_index(source_dataset, dataset_idx)

        actual_indices, actual_image_ids = _get_actual_temporal_indices_and_image_ids(
            leaf_ds,
            local_idx,
            temporal_len=temporal_len,
            temporal_stride=temporal_stride,
        )

        if actual_image_ids is None:
            continue

        if len(set(actual_image_ids)) > 1:
            print(
                "Found non-padding temporal sample: "
                f"dataset_idx={dataset_idx}, "
                f"local_idx={local_idx}, "
                f"actual_indices={actual_indices}, "
                f"actual_image_ids={actual_image_ids}, "
                f"source=({_leaf_dataset_desc(leaf_ds)})"
            )
            return dataset_idx

    print(
        "Could not find a non-padding temporal sample in scan range. "
        "This can happen if the selected split begins with very sparse or segmented samples."
    )
    return None


def _inspect_one_sample_visual_temporal_diff(
    source_dataset: Dataset,
    dataset_idx: int,
    view_idx: int = 0,
) -> None:
    """
    Load one sample directly and compare adjacent temporal frames.
    """
    print("")
    print("Inspect selected non-padding sample visual difference:")

    leaf_ds, local_idx = _resolve_dataset_index(source_dataset, dataset_idx)
    actual_indices, actual_image_ids = _get_actual_temporal_indices_and_image_ids(
        leaf_ds,
        local_idx,
        temporal_len=getattr(leaf_ds, "temporal_len", 4),
        temporal_stride=getattr(leaf_ds, "temporal_stride", 1),
    )

    sample = source_dataset[dataset_idx]
    if "image_seq" not in sample:
        print("selected sample has no image_seq")
        return

    seq = sample["image_seq"]  # [T, V, C, H, W]
    print(f"selected dataset_idx={dataset_idx}")
    print(f"selected source=({_leaf_dataset_desc(leaf_ds)})")
    print(f"selected local_idx={local_idx}")
    print(f"selected actual_indices={actual_indices}")
    print(f"selected actual_image_ids={actual_image_ids}")
    print(f"selected image_seq shape={tuple(seq.shape)}")

    if not torch.is_tensor(seq):
        print("selected image_seq is not a tensor")
        return

    if seq.ndim != 5:
        print("selected image_seq should be [T, V, C, H, W]")
        return

    T, V, C, H, W = seq.shape
    if view_idx >= V:
        print(f"view_idx={view_idx} out of range for V={V}")
        return

    diffs = []
    for t in range(1, T):
        diff_t = (seq[t, view_idx] - seq[t - 1, view_idx]).abs().mean()
        diffs.append(float(diff_t.item()))

    visual_status = "OK"
    if T > 1 and max(diffs) < 1e-8:
        visual_status = "SUSPICIOUS_ALL_TEMPORAL_FRAMES_IDENTICAL"
    else:
        visual_status = "OK_NON_PADDING_FRAMES_DIFFER"

    print(f"selected view_idx={view_idx}")
    print(f"selected adjacent frame mean abs diffs={diffs}")
    print(f"selected visual status={visual_status}")


def _debug_dataloader(
    datamodule: LightningDataModule,
    dataset: Dataset,
    split: str,
) -> DataLoader:
    # Debug must be deterministic. Do not shuffle.
    return DataLoader(
        IndexedDataset(dataset),
        batch_size=getattr(datamodule, "batch_size", 1),
        shuffle=False,
        num_workers=getattr(datamodule, "num_workers", 0),
        pin_memory=getattr(datamodule, "pin_memory", False),
    )


def _next_batch(datamodule: LightningDataModule) -> tuple[dict[str, Any], Dataset, str]:
    datamodule.prepare_data()
    datamodule.setup("fit")

    split_names = ["train", "val", "test"]
    for split in split_names:
        dataset = getattr(datamodule, f"data_{split}", None)
        if dataset is None:
            continue

        loader = _debug_dataloader(datamodule, dataset, split)

        try:
            return next(iter(loader)), dataset, split
        except StopIteration:
            continue

    raise RuntimeError("No non-empty dataloader is available for debug_image_seq")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    _ensure_paths(cfg)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.dataset)
    batch, source_dataset, split = _next_batch(datamodule)

    print("=" * 100)
    print("Dataset / config info")
    print("=" * 100)
    print("Dataset target:", cfg.dataset._target_)
    print("Selected split:", split)
    print("dataset_type:", cfg.dataset.get("dataset_type", None))
    print("return_image_seq:", cfg.dataset.get("return_image_seq", False))
    print("temporal_len:", cfg.dataset.get("temporal_len", None))
    print("temporal_stride:", cfg.dataset.get("temporal_stride", None))

    camera_views = list(cfg.dataset.get("camera_views", []))
    print("camera_views:", camera_views)
    for view_idx, view_name in enumerate(camera_views):
        print(f"view {view_idx} = {view_name}")

    print("batch_size:", getattr(datamodule, "batch_size", None))
    print("num_workers:", getattr(datamodule, "num_workers", None))
    print("source_dataset length:", len(source_dataset))

    print("")
    print("=" * 100)
    print("Batch keys / shapes")
    print("=" * 100)

    keys = [
        "image_front",
        "image_left",
        "image_right",
        "image_seq",
        "goal",
        "path",
        "_dataset_index",
    ]

    for key in keys:
        if key in batch:
            print(f"{key}: {_shape(batch[key])}")
        else:
            print(f"{key}: <missing>")

    temporal_len = int(cfg.dataset.get("temporal_len", 4))
    temporal_stride = int(cfg.dataset.get("temporal_stride", 1))

    print("")
    print("=" * 100)
    print("Temporal index debug for current batch")
    print("=" * 100)

    if "_dataset_index" in batch:
        dataset_indices = batch["_dataset_index"].tolist()
        max_samples_to_print = 20

        status_counts: dict[str, int] = {}

        for sample_pos, dataset_idx in enumerate(dataset_indices[:max_samples_to_print]):
            status = _print_temporal_debug_record(
                source_dataset=source_dataset,
                dataset_idx=int(dataset_idx),
                sample_pos=sample_pos,
                temporal_len=temporal_len,
                temporal_stride=temporal_stride,
            )
            status_counts[status] = status_counts.get(status, 0) + 1

        print("")
        print("temporal index debug summary in current batch:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count}")
    else:
        print("temporal index debug: <dataset indices missing>")

    max_scan = int(cfg.get("debug_max_scan", 200))
    max_bad_print = int(cfg.get("debug_max_bad_print", 30))

    _scan_temporal_consistency(
        source_dataset=source_dataset,
        temporal_len=temporal_len,
        temporal_stride=temporal_stride,
        max_scan=max_scan,
        max_bad_print=max_bad_print,
    )

    print("")
    print("=" * 100)
    print("Tensor stats")
    print("=" * 100)

    if "image_front" in batch:
        _print_tensor_stats("image_front", batch["image_front"])
    else:
        print("image_front dtype/min/max: <missing>")

    if "image_seq" in batch:
        _print_tensor_stats("image_seq", batch["image_seq"])
    else:
        print("image_seq dtype/min/max: <missing>")

    print("")
    print("=" * 100)
    print("image_seq shape / current-front check")
    print("=" * 100)

    if "image_seq" in batch:
        image_seq = batch["image_seq"]

        assert image_seq.ndim == 6, "image_seq should be [B, T, V, C, H, W]"

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

        print("")
        print("=" * 100)
        print("Temporal visual difference debug for current batch")
        print("=" * 100)

        max_samples_to_print = min(8, B)
        view_idx = 0

        for sample_pos in range(max_samples_to_print):
            frame_diffs = []
            for t in range(1, T):
                diff_t = (
                    image_seq[sample_pos, t, view_idx]
                    - image_seq[sample_pos, t - 1, view_idx]
                ).abs().mean()
                frame_diffs.append(float(diff_t.item()))

            max_diff = max(frame_diffs) if frame_diffs else 0.0

            visual_status = "OK"
            if T > 1 and max_diff < 1e-8:
                visual_status = "SUSPICIOUS_ALL_TEMPORAL_FRAMES_IDENTICAL"

            print(
                f"sample {sample_pos}: "
                f"view_idx={view_idx}, "
                f"adjacent frame mean abs diffs={frame_diffs}, "
                f"status={visual_status}"
            )
    else:
        print("image_seq shape / current-front check: <image_seq missing>")

    print("")
    print("=" * 100)
    print("Non-padding temporal visual check")
    print("=" * 100)

    non_padding_scan = int(cfg.get("debug_non_padding_scan", 2000))
    chosen_idx = _find_non_padding_temporal_sample(
        source_dataset=source_dataset,
        temporal_len=temporal_len,
        temporal_stride=temporal_stride,
        max_scan=non_padding_scan,
    )

    if chosen_idx is not None:
        _inspect_one_sample_visual_temporal_diff(
            source_dataset=source_dataset,
            dataset_idx=chosen_idx,
            view_idx=0,
        )


if __name__ == "__main__":
    main()