import argparse
import json
import pickle
import random
import re
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser("Prepare CHDAC polygon columns as DTLR single-column data")
    parser.add_argument("--zip-path", default="data/dataset.zip", type=Path)
    parser.add_argument("--raw-root", default="data/CHDAC", type=Path)
    parser.add_argument("--output-root", default="data/chdac_dtlr", type=Path)
    parser.add_argument("--margin", default=8, type=int)
    parser.add_argument("--split-seed", default=42, type=int)
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--valid-ratio", default=0.1, type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_id(value):
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")


def maybe_extract(zip_path, raw_root):
    if raw_root.exists():
        return False
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    raw_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(raw_root.parent))
    extracted = raw_root.parent / "dataset"
    if extracted.exists() and extracted != raw_root:
        extracted.rename(raw_root)
    return True


def page_splits(page_ids, train_ratio, valid_ratio, seed):
    pages = sorted(page_ids)
    random.Random(seed).shuffle(pages)
    total = len(pages)
    n_train = int(total * train_ratio)
    n_valid = int(total * valid_ratio)
    if total >= 3:
        n_train = max(1, min(n_train, total - 2))
        n_valid = max(1, min(n_valid, total - n_train - 1))
    return {
        "train": set(pages[:n_train]),
        "valid": set(pages[n_train : n_train + n_valid]),
        "test": set(pages[n_train + n_valid :]),
    }


def assign_split(page_id, splits):
    for split, pages in splits.items():
        if page_id in pages:
            return split
    raise KeyError(page_id)


def bbox_from_points(points, width, height, margin):
    if len(points) < 6 or len(points) % 2 != 0:
        raise ValueError(f"expected an even polygon point list, got {len(points)}")
    xs = [float(points[i]) for i in range(0, len(points), 2)]
    ys = [float(points[i]) for i in range(1, len(points), 2)]
    xmin = max(int(min(xs)) - margin, 0)
    xmax = min(int(max(xs)) + margin, width)
    ymin = max(int(min(ys)) - margin, 0)
    ymax = min(int(max(ys)) + margin, height)
    if xmax <= xmin:
        xmax = min(xmin + 1, width)
    if ymax <= ymin:
        ymax = min(ymin + 1, height)
    return xmin, ymin, xmax, ymax


def length_bins(samples):
    lengths = [len(sample["text"]) for sample in samples]
    return {
        "1": sum(length == 1 for length in lengths),
        "2": sum(length == 2 for length in lengths),
        "3-5": sum(3 <= length <= 5 for length in lengths),
        "6-10": sum(6 <= length <= 10 for length in lengths),
        "11+": sum(length >= 11 for length in lengths),
    }


def main():
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    output_lines = output_root / "lines"

    extracted = maybe_extract(args.zip_path.resolve(), raw_root)

    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"{output_root} exists. Use --overwrite to replace generated files.")
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_lines.mkdir(parents=True, exist_ok=True)

    label_path = raw_root / "train" / "label.json"
    image_root = raw_root / "train" / "image"
    labels_json = json.loads(label_path.read_text(encoding="utf-8"))
    splits = page_splits(labels_json.keys(), args.train_ratio, args.valid_ratio, args.split_seed)

    labels = {"ground_truth": {"train": [], "valid": [], "test": []}}
    charset = Counter()
    skipped = Counter()
    crop_sizes = []

    for page_name in sorted(labels_json):
        image_path = image_root / page_name
        if not image_path.exists():
            skipped["missing_images"] += 1
            continue
        split = assign_split(page_name, splits)
        page_id = safe_id(Path(page_name).stem)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            for col_idx, record in enumerate(labels_json[page_name]):
                text = (record.get("transcription") or "").strip()
                points = record.get("points") or []
                if not text:
                    skipped["empty_transcription"] += 1
                    continue
                try:
                    bbox = bbox_from_points(points, width, height, args.margin)
                except ValueError:
                    skipped["bad_points"] += 1
                    continue
                line_id = f"{page_id}_{col_idx:03d}"
                crop = image.crop(bbox)
                crop.save(output_lines / f"{line_id}.jpg", quality=95)
                crop_sizes.append(crop.size)
                labels["ground_truth"][split].append(
                    {"id": line_id, "text": text, "direction": "vertical"}
                )
                charset.update(text)

    with (output_root / "labels.pkl").open("wb") as f:
        pickle.dump(labels, f)
    with (output_root / "charset.pkl").open("wb") as f:
        pickle.dump(sorted(charset), f)

    split_counts = {split: len(samples) for split, samples in labels["ground_truth"].items()}
    meta = {
        "dataset": "CHDAC",
        "zip_path": str(args.zip_path),
        "raw_root": str(raw_root),
        "output_root": str(output_root),
        "extracted_raw_root": extracted,
        "margin": args.margin,
        "split_seed": args.split_seed,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "page_splits": {split: len(pages) for split, pages in splits.items()},
        "splits": split_counts,
        "total_columns": sum(split_counts.values()),
        "total_chars": sum(
            len(sample["text"])
            for samples in labels["ground_truth"].values()
            for sample in samples
        ),
        "unique_chars": len(charset),
        "length_bins": {
            split: length_bins(samples) for split, samples in labels["ground_truth"].items()
        },
        "crop_size": {
            "min_width": min((size[0] for size in crop_sizes), default=0),
            "max_width": max((size[0] for size in crop_sizes), default=0),
            "min_height": min((size[1] for size in crop_sizes), default=0),
            "max_height": max((size[1] for size in crop_sizes), default=0),
        },
        "skipped": dict(skipped),
        "loader_options": {
            "mth1000_root": output_root.name,
            "mth1000_raw_root": raw_root.name,
            "mth1000_image_ext": "jpg",
        },
    }
    (output_root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
