import argparse
import json
import pickle
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


PAGE_NS = {"p": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2017-07-15"}


def parse_args():
    parser = argparse.ArgumentParser("Prepare HDRC PAGE-XML as DTLR single-column data")
    parser.add_argument("--image-root", default="data/HDRC", type=Path)
    parser.add_argument("--xml-root", default="data/HDRC_GT", type=Path)
    parser.add_argument("--output-root", default="data/hdrc_dtlr", type=Path)
    parser.add_argument("--margin", default=8, type=int)
    parser.add_argument("--split-seed", default=42, type=int)
    parser.add_argument("--train-ratio", default=0.8, type=float)
    parser.add_argument("--valid-ratio", default=0.1, type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_id(value):
    return re.sub(r"[^0-9A-Za-z_]+", "_", value).strip("_")


def collect_image_paths(image_root):
    paths = {}
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        for path in image_root.glob(pattern):
            paths.setdefault(path.stem, path)
    return dict(sorted(paths.items()))


def parse_points(points):
    coords = []
    for pair in points.split():
        x, y = pair.split(",", 1)
        coords.append((float(x), float(y)))
    if not coords:
        raise ValueError("empty PAGE-XML Coords points")
    return coords


def parse_page_xml(path):
    root = ET.parse(path).getroot()
    records = []
    for order, line in enumerate(root.findall(".//p:TextLine", PAGE_NS)):
        coords_node = line.find("p:Coords", PAGE_NS)
        text_node = line.find("p:TextEquiv/p:Unicode", PAGE_NS)
        if coords_node is None or text_node is None:
            continue
        text = (text_node.text or "").strip()
        points = coords_node.attrib.get("points", "").strip()
        if not text or not points:
            continue
        coords = parse_points(points)
        xs = [x for x, _ in coords]
        ys = [y for _, y in coords]
        records.append(
            {
                "order": order,
                "line_id": safe_id(line.attrib.get("id", f"l{order}")),
                "text": text,
                "xmin": min(xs),
                "xmax": max(xs),
                "ymin": min(ys),
                "ymax": max(ys),
                "reading_direction": line.attrib.get("readingDirection", ""),
            }
        )
    return records


def grouped_page_splits(page_ids, train_ratio, valid_ratio, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for page_id in page_ids:
        grouped[page_id.split("_", 1)[0]].append(page_id)

    splits = {"train": set(), "valid": set(), "test": set()}
    for _, pages in sorted(grouped.items()):
        pages = sorted(pages)
        rng.shuffle(pages)
        total = len(pages)
        n_train = int(total * train_ratio)
        n_valid = int(total * valid_ratio)
        if total >= 3:
            n_train = max(1, min(n_train, total - 2))
            n_valid = max(1, min(n_valid, total - n_train - 1))
        splits["train"].update(pages[:n_train])
        splits["valid"].update(pages[n_train : n_train + n_valid])
        splits["test"].update(pages[n_train + n_valid :])
    return splits


def assign_split(page_id, split_pages):
    for split, pages in split_pages.items():
        if page_id in pages:
            return split
    return "train"


def crop_record(image, record, margin):
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


def summarize(labels, charset, split_pages, output_root, args, skipped):
    split_counts = {k: len(v) for k, v in labels["ground_truth"].items()}
    length_bins = {}
    for split, samples in labels["ground_truth"].items():
        lengths = [len(sample["text"]) for sample in samples]
        length_bins[split] = {
            "1": sum(length == 1 for length in lengths),
            "2": sum(length == 2 for length in lengths),
            "3-5": sum(3 <= length <= 5 for length in lengths),
            "6-10": sum(6 <= length <= 10 for length in lengths),
            "11+": sum(length >= 11 for length in lengths),
        }
    return {
        "dataset": "HDRC",
        "image_root": str(args.image_root),
        "xml_root": str(args.xml_root),
        "output_root": str(output_root),
        "margin": args.margin,
        "split_seed": args.split_seed,
        "splits": split_counts,
        "page_splits": {k: len(v) for k, v in split_pages.items()},
        "total_lines": sum(split_counts.values()),
        "total_chars": sum(
            len(sample["text"])
            for samples in labels["ground_truth"].values()
            for sample in samples
        ),
        "unique_chars": len(charset),
        "length_bins": length_bins,
        "skipped": skipped,
        "loader_options": {
            "mth1000_root": output_root.name,
            "mth1000_raw_root": "HDRC",
            "mth1000_image_ext": "jpg",
        },
    }


def main():
    args = parse_args()
    image_root = args.image_root.resolve()
    xml_root = args.xml_root.resolve()
    output_root = args.output_root.resolve()
    output_lines = output_root / "lines"

    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"{output_root} exists. Use --overwrite to replace generated files.")
    output_root.mkdir(parents=True, exist_ok=True)
    output_lines.mkdir(parents=True, exist_ok=True)

    image_paths = collect_image_paths(image_root)
    xml_paths = {path.stem: path for path in xml_root.glob("*.xml")}
    page_ids = sorted(set(image_paths) & set(xml_paths))
    split_pages = grouped_page_splits(
        page_ids,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.split_seed,
    )

    labels = {"ground_truth": {"train": [], "valid": [], "test": []}}
    charset = Counter()
    skipped = Counter()

    for page_id in page_ids:
        records = parse_page_xml(xml_paths[page_id])
        if not records:
            skipped["pages_without_textlines"] += 1
            continue
        split = assign_split(page_id, split_pages)
        with Image.open(image_paths[page_id]) as image:
            image = image.convert("RGB")
            for record in records:
                line_id = f"{page_id}_{record['line_id']}"
                line_path = output_lines / f"{line_id}.jpg"
                crop = crop_record(image, record, args.margin)
                crop.save(line_path, quality=95)
                labels["ground_truth"][split].append(
                    {"id": line_id, "text": record["text"], "direction": "vertical"}
                )
                charset.update(record["text"])

    with (output_root / "labels.pkl").open("wb") as f:
        pickle.dump(labels, f)
    with (output_root / "charset.pkl").open("wb") as f:
        pickle.dump(sorted(charset), f)

    meta = summarize(labels, charset, split_pages, output_root, args, dict(skipped))
    (output_root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
