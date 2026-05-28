import csv
import re
import shutil
import tarfile
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Literal, Sequence, Tuple

import torch
import zarr
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

from limo.src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

CAMERA_VIEW_TO_TOPIC = {
    "front": "hdr_front",
    "left": "hdr_left",
    "right": "hdr_right",
}
DEFAULT_CAMERA_VIEWS = ["front", "left", "right"]
IMAGE_ID_KEYS = ("image_id", "image_ids", "frame_id", "frame_ids")


def normalize_camera_views(camera_views: Sequence[str] | None) -> list[str]:
    views = list(camera_views) if camera_views is not None else DEFAULT_CAMERA_VIEWS
    if not views:
        raise ValueError("camera_views must contain at least one camera view")

    normalized = []
    for view in views:
        view_name = str(view).lower()
        if view_name not in CAMERA_VIEW_TO_TOPIC:
            valid = ", ".join(CAMERA_VIEW_TO_TOPIC)
            raise ValueError(
                f"Unsupported camera view '{view}'. Supported views: {valid}"
            )
        normalized.append(view_name)
    return normalized


def parse_missions_csv(missions_csv: Path) -> dict[str, str]:
    """Parse missions CSV and return a dict mapping Timestamp to Split."""
    timestamp_to_split = {}
    with missions_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            timestamp = row.get("Timestamp", "").strip()
            split = row.get("Split", "").strip()
            if timestamp:
                timestamp_to_split[timestamp] = split
    return timestamp_to_split


def pull_missions_from_hf(
    missions: list[str],
    topics: list[str],
    dataset_folder: Path,
    force_local: bool = False,
) -> Path:
    if force_local:
        log.info("`force_local` is True, skipping Hugging Face download.")
        return Path(dataset_folder)
    log.info("Downloading missions from Hugging Face...")
    for mission, topic in product(missions, topics):
        repo_file = get_repo_file(mission, topic)
        extracted_path = get_extracted_topic_path(dataset_folder, mission, topic)
        if has_extracted_contents(extracted_path):
            log.info(f"Skipping existing topic: {extracted_path}")
            continue

        log.info(f"Downloading {repo_file}...")
        cached_file = hf_hub_download(
            repo_id="leggedrobotics/grand_tour_dataset",
            filename=repo_file,
            revision="refs/pr/6",  # REMOVE LATER
            repo_type="dataset",
        )

        dest_path = dataset_folder / repo_file
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(cached_file, "r") as tar:
                tar.extractall(path=dest_path.parent)
        except tarfile.ReadError as e:
            log.error(f"Error opening or extracting tar file '{repo_file}': {e}")
            raise
    return Path(dataset_folder)


def get_repo_file(mission: str, topic: str) -> str:
    if topic.startswith("hdr_"):
        return f"{mission}/images/{topic}.tar"
    if topic in {"teleop_paths", "geometric_paths"}:
        return f"{mission}/data/{topic}.tar"
    return f"{mission}/{topic}.tar"


def get_extracted_topic_path(dataset_folder: Path, mission: str, topic: str) -> Path:
    if topic.startswith("hdr_"):
        return dataset_folder / mission / "images" / topic
    if topic in {"teleop_paths", "geometric_paths"}:
        return dataset_folder / mission / "data" / topic
    return dataset_folder / mission / topic


def has_extracted_contents(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def move_dataset(cache, dataset_folder, allow_patterns=["*"]):
    def convert_glob_patterns_to_regex(glob_patterns):
        regex_parts = []
        for pat in glob_patterns:
            # Escape regex special characters except for * and ?
            pat = re.escape(pat)
            # Convert escaped glob wildcards to regex equivalents
            pat = pat.replace(r"\*", ".*").replace(r"\?", ".")
            # Make sure it matches full paths
            regex_parts.append(f".*{pat}$")

        # Join with |
        combined = "|".join(regex_parts)
        return re.compile(combined)

    pattern = convert_glob_patterns_to_regex(allow_patterns)
    files = [f for f in Path(cache).rglob("*") if pattern.match(str(f))]
    tar_files = [f for f in files if f.suffix == ".tar"]

    for source_path in tar_files:
        dest_path = dataset_folder / source_path.relative_to(cache)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(source_path, "r") as tar:
                tar.extractall(path=dest_path.parent)
        except tarfile.ReadError as e:
            log.error(f"Error opening or extracting tar file '{source_path}': {e}")
        except Exception as e:
            log.error(
                f"An unexpected error occurred while processing {source_path}: {e}"
            )

    other_files = [f for f in files if not f.suffix == ".tar" and f.is_file()]
    for source_path in other_files:
        dest_path = dataset_folder / source_path.relative_to(cache)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)


class MissionDataset(Dataset):
    def __init__(
        self,
        dataset_type: Literal["tel", "geo", "aug"],
        dataset_folder: Path,
        mission_name: str,
        transform: transforms.Compose,
        with_side_cams: bool = False,
        return_image_seq: bool = False,
        temporal_len: int = 4,
        temporal_stride: int = 1,
        camera_views: Sequence[str] | None = None,
    ):
        self.dataset_type = dataset_type
        self.dataset_folder = dataset_folder
        self.mission_name = mission_name
        self.transform = transform
        self.with_side_cams = with_side_cams
        self.return_image_seq = return_image_seq
        self.temporal_len = temporal_len
        self.temporal_stride = temporal_stride
        self.camera_views = normalize_camera_views(camera_views)

        if self.temporal_len <= 0:
            raise ValueError("temporal_len must be positive")
        if self.temporal_stride <= 0:
            raise ValueError("temporal_stride must be positive")

        mission_dir = dataset_folder / mission_name
        if not mission_dir.exists():
            err = f"Mission dataset '{mission_name}' not found in {dataset_folder}"
            log.error(err)
            raise FileNotFoundError(err)

        if dataset_type == "tel":
            self.z = zarr.open_group(
                str(mission_dir / "data" / "teleop_paths"), mode="r"
            )
        elif dataset_type == "geo":
            self.z = zarr.open_group(
                str(mission_dir / "data" / "geometric_paths"), mode="r"
            )
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

        self.image_id_key, self.image_ids = self.load_image_ids()
        self.segment_start_indices = self.compute_segment_start_indices()

    def __len__(self):
        return len(self.z["path"])

    def load_image_ids(self) -> tuple[str, list[int]]:
        available_keys = set(self.z.array_keys())
        for key in IMAGE_ID_KEYS:
            if key not in available_keys:
                continue
            image_ids = [int(image_id) for image_id in self.z[key][:]]
            if len(image_ids) != len(self):
                raise ValueError(
                    f"Image id field '{key}' has length {len(image_ids)}, "
                    f"expected {len(self)}"
                )
            return key, image_ids

        keys = ", ".join(sorted(available_keys))
        raise KeyError(
            "Could not find an image id field in path zarr. "
            f"Tried {IMAGE_ID_KEYS}. Available array keys: {keys}"
        )

    def compute_segment_start_indices(self) -> list[int]:
        segment_starts = [0] * len(self.image_ids)
        for idx in range(1, len(self.image_ids)):
            image_id_delta = self.image_ids[idx] - self.image_ids[idx - 1]
            if 0 <= image_id_delta <= 1:
                segment_starts[idx] = segment_starts[idx - 1]
            else:
                segment_starts[idx] = idx
        return segment_starts

    def get_image_id(self, idx: int) -> int:
        return self.image_ids[idx]

    def load_image(self, topic: str, idx: int) -> Image.Image:
        image_path = (
            self.dataset_folder
            / self.mission_name
            / "images"
            / topic
            / f"{self.get_image_id(idx):06d}.jpeg"
        )
        if not image_path.exists():
            log.error(f"Image not found at {image_path}")
            raise FileNotFoundError(f"Image not found at {image_path}")
        return Image.open(image_path).convert("RGB")

    def load_transformed_image(self, topic: str, idx: int) -> torch.Tensor:
        return self.transform(self.load_image(topic, idx))

    def get_temporal_indices(self, idx: int) -> list[int]:
        segment_start = self.segment_start_indices[idx]
        return [
            max(idx - step * self.temporal_stride, segment_start)
            for step in range(self.temporal_len - 1, -1, -1)
        ]

    def load_image_seq(self, idx: int) -> torch.Tensor:
        frames = []
        for frame_idx in self.get_temporal_indices(idx):
            views = []
            for view in self.camera_views:
                topic = CAMERA_VIEW_TO_TOPIC[view]
                views.append(self.load_transformed_image(topic, frame_idx))
            frames.append(torch.stack(views, dim=0))
        return torch.stack(frames, dim=0)

    def __getitem__(self, idx):
        image_front = self.load_transformed_image("hdr_front", idx)

        goal = torch.tensor(self.z["goal"][idx], dtype=torch.float32)
        path = torch.tensor(self.z["path"][idx], dtype=torch.float32)

        batch = {
            "image_front": image_front,
            "goal": goal,
            "path": path,
        }

        if self.with_side_cams:
            image_left = self.load_transformed_image("hdr_left", idx)
            batch["image_left"] = image_left

            image_right = self.load_transformed_image("hdr_right", idx)
            batch["image_right"] = image_right

        if self.return_image_seq:
            batch["image_seq"] = self.load_image_seq(idx)

        return batch


def get_mission_dataset(
    dataset_type: Literal["tel", "geo", "aug"],
    dataset_folder: Path,
    mission_name: str,
    transform: transforms.Compose,
    with_side_cams: bool = False,
    return_image_seq: bool = False,
    temporal_len: int = 4,
    temporal_stride: int = 1,
    camera_views: Sequence[str] | None = None,
) -> Dataset:
    if dataset_type == "aug":
        geo_ds = MissionDataset(
            "geo",
            dataset_folder,
            mission_name,
            transform,
            with_side_cams,
            return_image_seq,
            temporal_len,
            temporal_stride,
            camera_views,
        )
        tel_ds = MissionDataset(
            "tel",
            dataset_folder,
            mission_name,
            transform,
            with_side_cams,
            return_image_seq,
            temporal_len,
            temporal_stride,
            camera_views,
        )
        return ConcatDataset([geo_ds, tel_ds])
    if dataset_type in ["tel", "geo"]:
        return MissionDataset(
            dataset_type,
            dataset_folder,
            mission_name,
            transform,
            with_side_cams,
            return_image_seq,
            temporal_len,
            temporal_stride,
            camera_views,
        )
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")


def get_dataset(
    dataset_type: Literal["tel", "geo", "aug"],
    dataset_folder: Path,
    missions_csv: Path,
    with_side_cams: bool = False,
    image_size: Tuple[int, int] = (308, 476),
    return_image_seq: bool = False,
    temporal_len: int = 4,
    temporal_stride: int = 1,
    camera_views: Sequence[str] | None = None,
):
    missions = parse_missions_csv(missions_csv)
    camera_views = normalize_camera_views(camera_views)

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )

    topics = ["hdr_front"]
    if with_side_cams:
        topics += ["hdr_left", "hdr_right"]
    if return_image_seq:
        topics += [CAMERA_VIEW_TO_TOPIC[view] for view in camera_views]
    topics = list(dict.fromkeys(topics))
    if dataset_type in ["tel", "aug"]:
        topics.append("teleop_paths")
    if dataset_type in ["geo", "aug"]:
        topics.append("geometric_paths")

    grandtour_folder = dataset_folder / "grandtour"
    grandtour_folder.mkdir(parents=True, exist_ok=True)
    datset_dir = pull_missions_from_hf(list(missions.keys()), topics, grandtour_folder)

    datasets = defaultdict(list)
    for mission, split in missions.items():
        datasets[split].append(
            get_mission_dataset(
                dataset_type=dataset_type,
                dataset_folder=datset_dir,
                mission_name=mission,
                transform=transform,
                with_side_cams=with_side_cams,
                return_image_seq=return_image_seq,
                temporal_len=temporal_len,
                temporal_stride=temporal_stride,
                camera_views=camera_views,
            )
        )

    splits: dict[str, Dataset] = dict()
    for split, ds_list in datasets.items():
        splits[split] = ConcatDataset(ds_list)
        log.info(f"Split '{split}' has {len(splits[split])} samples")
    return splits
