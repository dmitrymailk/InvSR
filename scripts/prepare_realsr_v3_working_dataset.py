#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-lq-out", required=True)
    parser.add_argument("--val-gt-out", required=True)
    return parser.parse_args()


def reset_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()


def require_dir(path: Path) -> None:
    if not path.is_dir():
        raise SystemExit(f"missing required directory: {path}")


def build_train_dataset(raw_root: Path, train_out: Path) -> int:
    train_count = 0
    for camera in ("Canon", "Nikon"):
        for scale in ("2", "3", "4"):
            src_dir = raw_root / camera / "Train" / scale
            require_dir(src_dir)
            for src in sorted(src_dir.glob("*_HR.png")):
                stem = src.name.replace("_HR.png", "")
                dst = train_out / f"{camera}_train_x{scale}_{stem}.png"
                os.symlink(src.resolve(), dst)
                train_count += 1
    return train_count


def build_validation_dataset(raw_root: Path, val_lq_out: Path, val_gt_out: Path) -> int:
    val_count = 0
    for camera in ("Canon", "Nikon"):
        src_dir = raw_root / camera / "Test" / "4"
        require_dir(src_dir)
        for hr_path in sorted(src_dir.glob("*_HR.png")):
            stem = hr_path.name.replace("_HR.png", "")
            lr_path = src_dir / f"{stem}_LR4.png"
            if not lr_path.is_file():
                raise SystemExit(f"missing LR4 pair for {hr_path}")

            file_name = f"{camera}_test_x4_{stem}.png"

            gt = Image.open(hr_path).convert("RGB")
            lq = Image.open(lr_path).convert("RGB")

            gt_w = gt.width - (gt.width % 8)
            gt_h = gt.height - (gt.height % 8)
            if gt_w == 0 or gt_h == 0:
                raise SystemExit(
                    f"invalid GT size after mod8 crop: {hr_path} -> {(gt.width, gt.height)}"
                )

            gt = gt.crop((0, 0, gt_w, gt_h))

            lq_w = gt_w // 4
            lq_h = gt_h // 4
            if lq_w == 0 or lq_h == 0:
                raise SystemExit(
                    f"invalid LQ size after x4 crop: {lr_path} -> {(lq.width, lq.height)}"
                )

            lq = lq.crop((0, 0, lq_w, lq_h))

            gt.save(val_gt_out / file_name)
            lq.save(val_lq_out / file_name)
            val_count += 1
    return val_count


def main() -> None:
    args = parse_args()

    raw_root = Path(args.raw_root)
    train_out = Path(args.train_out)
    val_lq_out = Path(args.val_lq_out)
    val_gt_out = Path(args.val_gt_out)

    reset_output_dir(train_out)
    reset_output_dir(val_lq_out)
    reset_output_dir(val_gt_out)

    train_count = build_train_dataset(raw_root, train_out)
    val_count = build_validation_dataset(raw_root, val_lq_out, val_gt_out)

    expected_train = 505
    expected_val = 30
    if train_count != expected_train:
        raise SystemExit(
            f"unexpected train count: {train_count}, expected {expected_train}"
        )
    if val_count != expected_val:
        raise SystemExit(f"unexpected val count: {val_count}, expected {expected_val}")

    print(f"Prepared train_hr: {train_count}")
    print(f"Prepared val pairs: {val_count}")
    print(f"train_hr: {train_out}")
    print(f"val_lq:  {val_lq_out}")
    print(f"val_gt:  {val_gt_out}")


if __name__ == "__main__":
    main()
