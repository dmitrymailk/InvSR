#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from pathlib import Path

from PIL import Image


TOTAL_TRAIN_SPLITS = 85
SPLITS_PER_SHARD = 5
TOTAL_TRAIN_SHARDS = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lsdir-root", default="data/LSDIR")
    parser.add_argument("--prepared-root", default="data/LSDIR-prepared")
    parser.add_argument("--archive-root", default="data/LSDIR-archives")
    parser.add_argument(
        "--train-split",
        action="append",
        dest="train_splits",
        help="LSDIR train split to download, e.g. 0001000. Repeat to download several splits.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not try to download from Hugging Face. Expect data to already exist under --lsdir-root.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded shard archives after extraction instead of deleting them.",
    )
    return parser.parse_args()


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target_path.resolve())


def count_pngs(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.rglob("*.png"))


def all_train_splits() -> list[str]:
    return [f"{index * 1000:07d}" for index in range(1, TOTAL_TRAIN_SPLITS + 1)]


def shard_name_for_split(train_split: str) -> str:
    split_value = int(train_split)
    if split_value % 1000 != 0 or split_value < 1000:
        raise SystemExit(f"Unexpected LSDIR split name: {train_split}")

    split_index = split_value // 1000
    shard_index = (split_index - 1) // 5
    return f"shard-{shard_index:02d}.tar.gz"


def archive_slug(archive_path: Path) -> str:
    return archive_path.name.removesuffix(".tar.gz")


def splits_for_shard_name(shard_name: str) -> set[str]:
    shard_index = int(shard_name.removeprefix("shard-").removesuffix(".tar.gz"))
    first_split = shard_index * SPLITS_PER_SHARD + 1
    last_split = min(first_split + SPLITS_PER_SHARD - 1, TOTAL_TRAIN_SPLITS)
    return {f"{index * 1000:07d}" for index in range(first_split, last_split + 1)}


def requested_train_splits_present(
    destination_root: Path, train_splits: list[str]
) -> bool:
    return all(
        count_pngs(destination_root / train_split) > 0 for train_split in train_splits
    )


def extract_train_archive(
    archive_path: Path,
    requested_splits: set[str],
    destination_root: Path,
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    shard_splits = splits_for_shard_name(archive_path.name)
    requested_in_archive = shard_splits.intersection(requested_splits)
    if not requested_in_archive:
        return

    marker_path = destination_root / f".{archive_slug(archive_path)}.complete"
    if requested_in_archive == shard_splits and marker_path.exists():
        return

    with tarfile.open(archive_path, "r:gz") as tar:
        extracted = 0
        for member in tar.getmembers():
            split_name = member.name.split("/", 1)[0]
            if split_name not in requested_in_archive:
                continue
            relative_name = member.name.split("/", 1)[1] if "/" in member.name else ""
            if not relative_name:
                continue

            target_path = destination_root / split_name / relative_name
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            if target_path.exists():
                extracted += 1
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with target_path.open("wb") as handle:
                handle.write(source.read())
            extracted += 1

    if extracted == 0:
        raise SystemExit(
            f"Requested splits {sorted(requested_in_archive)} were not found inside archive {archive_path.name}."
        )

    missing_splits = [
        split_name
        for split_name in requested_in_archive
        if count_pngs(destination_root / split_name) == 0
    ]
    if missing_splits:
        raise SystemExit(
            f"Some requested LSDIR splits were not extracted from {archive_path.name}: {missing_splits}"
        )

    if requested_in_archive == shard_splits:
        marker_path.write_text("complete\n")


def extract_val_split(archive_path: Path, destination_root: Path) -> None:
    hr_val_root = destination_root / "HR" / "val"
    x4_val_root = destination_root / "X4" / "val"
    marker_path = destination_root / f".{archive_slug(archive_path)}.complete"
    if (
        marker_path.exists()
        and count_pngs(hr_val_root) > 0
        and count_pngs(x4_val_root) > 0
    ):
        return

    with tarfile.open(archive_path, "r:gz") as tar:
        extracted = 0
        for member in tar.getmembers():
            if member.name.startswith("val1/HR/val/"):
                relative_name = member.name[len("val1/HR/val/") :]
                target_path = hr_val_root / relative_name
            elif member.name.startswith("val1/X4/val/"):
                relative_name = member.name[len("val1/X4/val/") :]
                target_path = x4_val_root / relative_name
            else:
                continue

            if not relative_name:
                continue
            if member.isdir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with target_path.open("wb") as handle:
                handle.write(source.read())
            extracted += 1

    if extracted == 0:
        raise SystemExit(
            f"Validation archive {archive_path.name} did not contain HR/X4 val files."
        )

    marker_path.write_text("complete\n")


def ensure_downloaded(
    lsdir_root: Path,
    archive_root: Path,
    train_splits: list[str],
    skip_download: bool,
    keep_archives: bool,
) -> None:
    hr_train_dir = lsdir_root / "HR" / "train"
    if (
        requested_train_splits_present(hr_train_dir, train_splits)
        and count_pngs(lsdir_root / "X4" / "val") > 0
    ):
        return

    if skip_download:
        raise SystemExit(
            "LSDIR data is missing under data/LSDIR and --skip-download was specified."
        )

    from huggingface_hub import hf_hub_download

    archive_root.mkdir(parents=True, exist_ok=True)
    shard_names = sorted({shard_name_for_split(split) for split in train_splits})

    print("Downloading LSDIR training shards from Hugging Face...")
    for shard_name in shard_names + ["val1.tar.gz"]:
        archive_path = Path(
            hf_hub_download(
                repo_id="ofsoundof/LSDIR",
                filename=shard_name,
                local_dir=str(archive_root),
            )
        )
        if shard_name == "val1.tar.gz":
            extract_val_split(archive_path, lsdir_root)
        else:
            extract_train_archive(archive_path, set(train_splits), hr_train_dir)
        if not keep_archives and archive_path.exists():
            archive_path.unlink()


def prepare_train_split(lsdir_root: Path, prepared_root: Path) -> int:
    train_src_root = lsdir_root / "HR" / "train"
    train_dst_root = prepared_root / "train_hr_flat"
    train_dst_root.mkdir(parents=True, exist_ok=True)

    prepared_count = 0
    for image_path in sorted(train_src_root.glob("*/*.png")):
        link_name = f"{image_path.parent.name}_{image_path.name}"
        ensure_symlink(train_dst_root / link_name, image_path)
        prepared_count += 1
    return prepared_count


def prepare_val_split(lsdir_root: Path, prepared_root: Path) -> int:
    lq_src_root = lsdir_root / "X4" / "val"
    gt_src_root = lsdir_root / "HR" / "val"
    lq_dst_root = prepared_root / "val_lq_x4_mod8"
    gt_dst_root = prepared_root / "val_gt_x4_mod8"

    if lq_dst_root.exists():
        shutil.rmtree(lq_dst_root)
    if gt_dst_root.exists():
        shutil.rmtree(gt_dst_root)

    lq_dst_root.mkdir(parents=True, exist_ok=True)
    gt_dst_root.mkdir(parents=True, exist_ok=True)

    if not lq_src_root.exists() or not gt_src_root.exists():
        return 0

    prepared_count = 0
    for lq_path in sorted(lq_src_root.glob("*.png")):
        stem = lq_path.stem
        if not stem.endswith("x4"):
            continue
        gt_name = f"{stem[:-2]}.png"
        gt_path = gt_src_root / gt_name
        if not gt_path.exists():
            continue

        with Image.open(lq_path) as lq_image, Image.open(gt_path) as gt_image:
            lq_width, lq_height = lq_image.size
            gt_width, gt_height = gt_image.size

            target_gt_width = (gt_width // 8) * 8
            target_gt_height = (gt_height // 8) * 8
            if target_gt_width == 0 or target_gt_height == 0:
                continue

            target_lq_width = target_gt_width // 4
            target_lq_height = target_gt_height // 4
            if target_lq_width == 0 or target_lq_height == 0:
                continue

            if target_lq_width > lq_width or target_lq_height > lq_height:
                continue

            cropped_lq = lq_image.crop((0, 0, target_lq_width, target_lq_height))
            cropped_gt = gt_image.crop((0, 0, target_gt_width, target_gt_height))

            cropped_lq.save(lq_dst_root / lq_path.name)
            cropped_gt.save(gt_dst_root / lq_path.name)
        prepared_count += 1

    return prepared_count


def main() -> int:
    args = parse_args()
    train_splits = args.train_splits or all_train_splits()

    lsdir_root = Path(args.lsdir_root)
    prepared_root = Path(args.prepared_root)
    archive_root = Path(args.archive_root)

    ensure_downloaded(
        lsdir_root,
        archive_root,
        train_splits,
        args.skip_download,
        args.keep_archives,
    )

    train_count = prepare_train_split(lsdir_root, prepared_root)
    val_count = prepare_val_split(lsdir_root, prepared_root)

    if train_count == 0:
        raise SystemExit("No LSDIR HR train images were found after preparation.")

    print(f"Prepared LSDIR train images: {train_count}")
    print(f"Prepared LSDIR x4 val pairs: {val_count}")
    print("Train root:", prepared_root / "train_hr_flat")
    if val_count > 0:
        print("Val LQ root:", prepared_root / "val_lq_x4_mod8")
        print("Val GT root:", prepared_root / "val_gt_x4_mod8")
    else:
        print(
            "Validation folders were not created because HR/val or X4/val is missing."
        )
    print("Next step: ./run_train_lsdir_x4_single_gpu.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
