import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def parse_args():
    parser = argparse.ArgumentParser("Build a real ancient glyph crop bank from MTH/TKH char boxes.")
    parser.add_argument(
        "--source-roots",
        nargs="+",
        default=[
            "data/TKHMTH2200/MTH1000",
            "data/TKHMTH2200/MTH1200",
            "data/TKHMTH2200/TKH",
        ],
    )
    parser.add_argument("--charset", default="data/tkhmth2200_mth1000_mth1200_charset.pkl")
    parser.add_argument("--output-root", default="data/ancient_glyph_bank_mth")
    parser.add_argument("--max-per-char", type=int, default=40)
    parser.add_argument("--min-size", type=int, default=10)
    parser.add_argument("--max-size", type=int, default=180)
    parser.add_argument("--padding", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_charset(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return [char for char, _ in sorted(obj.items(), key=lambda kv: kv[1])]
    return list(obj)


def find_image(img_dir, stem):
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        path = img_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def parse_char_box_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    char = parts[0]
    try:
        coords = [float(x) for x in parts[1:5]]
    except ValueError:
        return None
    x1, y1, x2, y2 = coords
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return char, x1, y1, x2, y2


def estimate_border_background(gray):
    arr = np.asarray(gray, dtype=np.float32)
    if arr.size == 0:
        return 240.0
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return float(np.median(border))


def make_alpha_glyph(crop):
    crop = crop.convert("RGB")
    gray = crop.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    bg = estimate_border_background(gray)
    # Ancient ink is usually darker than paper. Difference catches red/brown ink too
    # after RGB->L conversion, while the dark condition suppresses paper texture.
    mask = ((bg - arr) > 18.0) | (arr < 185.0)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.35))
    rgba = crop.convert("RGBA")
    rgba.putalpha(mask_img)
    return rgba


def crop_glyph(image, box, padding, min_size, max_size):
    char, x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    if w < min_size or h < min_size or w > max_size or h > max_size:
        return None
    ix1 = max(0, int(np.floor(x1 - padding)))
    iy1 = max(0, int(np.floor(y1 - padding)))
    ix2 = min(image.width, int(np.ceil(x2 + padding)))
    iy2 = min(image.height, int(np.ceil(y2 + padding)))
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    crop = image.crop((ix1, iy1, ix2, iy2))
    return make_alpha_glyph(crop), (ix1, iy1, ix2, iy2)


def main():
    args = parse_args()
    random.seed(args.seed)

    charset = load_charset(args.charset)
    charset_set = set(charset)
    output_root = Path(args.output_root)
    glyph_dir = output_root / "glyphs"
    if output_root.exists() and args.overwrite:
        for path in sorted(output_root.glob("**/*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    glyph_dir.mkdir(parents=True, exist_ok=True)

    label_files = []
    for root_str in args.source_roots:
        root = Path(root_str)
        label_dir = root / "label_char"
        img_dir = root / "img"
        for label_path in label_dir.glob("*.txt"):
            image_path = find_image(img_dir, label_path.stem)
            if image_path is not None:
                label_files.append((root.name, label_path, image_path))
    random.shuffle(label_files)

    counts = Counter()
    records = []
    skipped = Counter()

    for source_name, label_path, image_path in label_files:
        if len(counts) >= len(charset) and min(counts.get(c, 0) for c in charset) >= args.max_per_char:
            break
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            skipped["bad_image"] += 1
            continue
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = label_path.read_text(encoding="gb18030", errors="ignore").splitlines()

        for line_no, line in enumerate(lines):
            parsed = parse_char_box_line(line)
            if parsed is None:
                skipped["bad_line"] += 1
                continue
            char = parsed[0]
            if char not in charset_set:
                skipped["not_in_charset"] += 1
                continue
            if counts[char] >= args.max_per_char:
                skipped["char_full"] += 1
                continue
            result = crop_glyph(image, parsed, args.padding, args.min_size, args.max_size)
            if result is None:
                skipped["bad_size"] += 1
                continue
            glyph, crop_box = result
            code = "_".join(f"{ord(ch):X}" for ch in char)
            index = counts[char]
            rel_path = Path("glyphs") / f"U{code}_{index:04d}.png"
            glyph.save(output_root / rel_path)
            counts[char] += 1
            records.append(
                {
                    "char": char,
                    "path": str(rel_path),
                    "source": source_name,
                    "page": image_path.name,
                    "label_file": label_path.name,
                    "line_no": line_no,
                    "crop_box": [int(v) for v in crop_box],
                    "size": [glyph.width, glyph.height],
                }
            )

    by_char = defaultdict(list)
    for record in records:
        by_char[record["char"]].append(record)

    (output_root / "records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    (output_root / "index.json").write_text(
        json.dumps(by_char, ensure_ascii=False),
        encoding="utf-8",
    )
    stats = {
        "source_roots": args.source_roots,
        "charset": args.charset,
        "charset_size": len(charset),
        "covered_chars": len(counts),
        "num_glyphs": len(records),
        "max_per_char": args.max_per_char,
        "min_count": min(counts.values()) if counts else 0,
        "count_hist": dict(Counter(counts.values())),
        "missing_chars": [c for c in charset if counts[c] == 0],
        "skipped": dict(skipped),
    }
    (output_root / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
