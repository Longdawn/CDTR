import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        "Prepare single-column line datasets from TKHMTH2200 archives"
    )
    parser.add_argument(
        "--src-root",
        default="data/TKHMTH2200",
        type=str,
        help="Root containing MTH1000/, MTH1200/, TKH/",
    )
    parser.add_argument(
        "--dst-root",
        default="data",
        type=str,
        help="Destination parent directory for generated *_dtlr datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["MTH1000", "MTH1200", "TKH"],
        help="Subset of datasets to process",
    )
    parser.add_argument(
        "--margin",
        default=8,
        type=int,
        help="Extra margin around each textline bounding rectangle",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing images and labels.pkl",
    )
    return parser.parse_args()


def parse_textline_file(path):
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for idx, raw_line in enumerate(f):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            text = parts[0]
            coords = list(map(float, parts[1:9]))
            xs = coords[0::2]
            ys = coords[1::2]
            records.append(
                {
                    "line_idx": idx,
                    "text": text,
                    "coords": coords,
                    "xmin": min(xs),
                    "xmax": max(xs),
                    "ymin": min(ys),
                    "ymax": max(ys),
                }
            )
    return records


def page_level_splits(page_ids):
    page_ids = sorted(page_ids)
    total = len(page_ids)
    n_train = int(total * 0.8)
    n_valid = int(total * 0.1)
    n_test = total - n_train - n_valid
    return {
        "train": set(page_ids[:n_train]),
        "valid": set(page_ids[n_train : n_train + n_valid]),
        "test": set(page_ids[n_train + n_valid : n_train + n_valid + n_test]),
    }


def collect_image_paths(img_root):
    image_paths = {}
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for path in img_root.glob(pattern):
            if path.stem not in image_paths:
                image_paths[path.stem] = path
    return dict(sorted(image_paths.items(), key=lambda kv: kv[0]))


def load_existing_mth1000_page_splits(existing_labels_path):
    with existing_labels_path.open("rb") as f:
        data = pickle.load(f)
    split_pages = defaultdict(set)
    for split, samples in data["ground_truth"].items():
        for sample in samples:
            page_id, _ = sample["id"].rsplit("_", 1)
            split_pages[split].add(page_id)
    return {k: set(v) for k, v in split_pages.items()}


def resolve_splits(dataset_name, src_root):
    if dataset_name == "MTH1000":
        existing = src_root.parent / "mth1000_dtlr" / "labels.pkl"
        if existing.exists():
            return load_existing_mth1000_page_splits(existing)
    page_ids = list(collect_image_paths(src_root / dataset_name / "img").keys())
    return page_level_splits(page_ids)


def build_output_root_name(dataset_name):
    return f"tkhmth2200_{dataset_name.lower()}_dtlr"


def crop_line_image(image, record, margin):
    width, height = image.size
    xmin = max(int(record["xmin"]) - margin, 0)
    xmax = min(int(record["xmax"]) + margin, width)
    ymin = max(int(record["ymin"]) - margin, 0)
    ymax = min(int(record["ymax"]) + margin, height)
    if xmax <= xmin:
        xmax = min(xmin + 1, width)
    if ymax <= ymin:
        ymax = min(ymin + 1, height)
    return image.crop((xmin, ymin, xmax, ymax))


def assign_split(page_id, split_pages):
    for split, page_set in split_pages.items():
        if page_id in page_set:
            return split
    return "train"


def process_dataset(dataset_name, src_root, dst_root, margin, overwrite):
    ds_root = src_root / dataset_name
    img_root = ds_root / "img"
    textline_root = ds_root / "label_textline"
    output_root = dst_root / build_output_root_name(dataset_name)
    output_lines = output_root / "lines"
    output_root.mkdir(parents=True, exist_ok=True)
    output_lines.mkdir(parents=True, exist_ok=True)

    split_pages = resolve_splits(dataset_name, src_root)
    labels = {"ground_truth": {"train": [], "valid": [], "test": []}}

    image_paths = list(collect_image_paths(img_root).values())
    total_lines = 0
    for image_path in image_paths:
        page_id = image_path.stem
        textline_path = textline_root / f"{page_id}.txt"
        if not textline_path.exists():
            continue
        records = parse_textline_file(textline_path)
        if not records:
            continue

        split = assign_split(page_id, split_pages)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            for record in records:
                line_id = f"{page_id}_{record['line_idx']:03d}"
                line_path = output_lines / f"{line_id}.jpg"
                if overwrite or not line_path.exists():
                    crop = crop_line_image(image, record, margin)
                    crop.save(line_path, quality=95)
                labels["ground_truth"][split].append(
                    {
                        "id": line_id,
                        "text": record["text"],
                        "direction": "vertical",
                    }
                )
                total_lines += 1

    labels_path = output_root / "labels.pkl"
    if labels_path.exists() and not overwrite:
        raise FileExistsError(f"{labels_path} already exists. Use --overwrite to replace it.")
    with labels_path.open("wb") as f:
        pickle.dump(labels, f)

    metadata = {
        "dataset": dataset_name,
        "source_root": str(ds_root),
        "output_root": str(output_root),
        "margin": margin,
        "splits": {k: len(v) for k, v in labels["ground_truth"].items()},
        "page_splits": {k: len(v) for k, v in split_pages.items()},
        "total_lines": total_lines,
        "raw_root_for_loader": f"TKHMTH2200/{dataset_name}",
    }
    (output_root / "meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    for dataset_name in args.datasets:
        process_dataset(
            dataset_name=dataset_name,
            src_root=src_root,
            dst_root=dst_root,
            margin=args.margin,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
