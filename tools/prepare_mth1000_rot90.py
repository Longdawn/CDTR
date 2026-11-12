#!/usr/bin/env python3
"""
Create a rotated horizontal comparison dataset from mth1000_dtlr.

This script rotates every line image 90 degrees counterclockwise so that the
original top-to-bottom reading order becomes left-to-right. It writes a
new dataset directory with the same split structure and text labels, and
adds an explicit horizontal direction tag to each sample.
"""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare rotated MTH1000-DTLR dataset")
    parser.add_argument(
        "--src-root",
        default="data/mth1000_dtlr",
        help="Source dataset root containing labels.pkl and lines/",
    )
    parser.add_argument(
        "--dst-root",
        default="data/mth1000_dtlr_rot90",
        help="Destination dataset root to create",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the destination root if it already exists",
    )
    return parser.parse_args()


def prepare_destination(dst_root: Path, overwrite: bool) -> None:
    if dst_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Destination {dst_root} already exists. Use --overwrite to replace it."
            )
        shutil.rmtree(dst_root)
    (dst_root / "lines").mkdir(parents=True, exist_ok=True)


def rotate_images(src_lines: Path, dst_lines: Path) -> None:
    image_paths = sorted(src_lines.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No .jpg images found under {src_lines}")

    total = len(image_paths)
    for idx, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            rotated = image.transpose(Image.Transpose.ROTATE_90)
            rotated.save(dst_lines / image_path.name, quality=95)
        if idx % 2000 == 0 or idx == total:
            print(f"[prepare_mth1000_rot90] rotated {idx}/{total} images")


def rewrite_labels(src_labels: Path, dst_labels: Path) -> None:
    data = pickle.load(open(src_labels, "rb"))
    if "ground_truth" not in data:
        raise KeyError(f"labels file {src_labels} does not contain 'ground_truth'")

    new_data = {"ground_truth": {}}
    for split, samples in data["ground_truth"].items():
        new_samples = []
        for sample in samples:
            new_sample = dict(sample)
            new_sample["direction"] = "horizontal"
            new_samples.append(new_sample)
        new_data["ground_truth"][split] = new_samples

    with open(dst_labels, "wb") as f:
        pickle.dump(new_data, f)


def main() -> None:
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    src_labels = src_root / "labels.pkl"
    src_lines = src_root / "lines"

    if not src_labels.exists():
        raise FileNotFoundError(f"Missing source labels file: {src_labels}")
    if not src_lines.exists():
        raise FileNotFoundError(f"Missing source lines directory: {src_lines}")

    prepare_destination(dst_root, overwrite=args.overwrite)
    rewrite_labels(src_labels, dst_root / "labels.pkl")
    rotate_images(src_lines, dst_root / "lines")

    print(f"[prepare_mth1000_rot90] done: {src_root} -> {dst_root}")


if __name__ == "__main__":
    main()
