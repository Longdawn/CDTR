import argparse
import json
import math
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate upright vertical Chinese text-line crops with MTH-style char boxes."
    )
    parser.add_argument("--output-root", default="data/synth_vertical_smoke_dtlr")
    parser.add_argument("--raw-root", default="data/SynthVerticalSmoke")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--bg-dir", default="data/synth_bg_patches")
    parser.add_argument("--font-path", default="text_renderer/example_data/font/simsun.ttf")
    parser.add_argument(
        "--font-paths",
        nargs="+",
        default=None,
        help="Optional list of fonts to sample per line. Overrides --font-path when set.",
    )
    parser.add_argument(
        "--source-labels",
        nargs="+",
        default=[
            "data/tkhmth2200_mth1000_dtlr/labels.pkl",
            "data/tkhmth2200_mth1200_dtlr/labels.pkl",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-font-size", type=int, default=34)
    parser.add_argument("--max-font-size", type=int, default=54)
    parser.add_argument("--min-margin", type=int, default=8)
    parser.add_argument("--max-margin", type=int, default=24)
    parser.add_argument("--max-height", type=int, default=1200)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--length-buckets",
        default="1-2:0.40,3-5:0.30,6-15:0.20,16-28:0.10",
        help="Comma-separated min-max:prob buckets. Default oversamples short vertical text.",
    )
    parser.add_argument(
        "--coverage-min-count",
        type=int,
        default=0,
        help=(
            "If >0, force every font-supported source character to appear at least "
            "this many times before falling back to frequency sampling."
        ),
    )
    parser.add_argument(
        "--forced-chars-per-sample",
        type=int,
        default=1,
        help="Maximum number of under-covered characters forced into one synthetic line.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5000,
        help="Write labels/meta/charset snapshots every N generated images.",
    )
    parser.add_argument(
        "--real-text-ratio",
        type=float,
        default=0.0,
        help="Probability of sampling a substring from source transcriptions instead of random chars.",
    )
    parser.add_argument(
        "--erase-prob",
        type=float,
        default=0.0,
        help="Probability of applying light stripe/patch erasing to simulate damaged print.",
    )
    parser.add_argument(
        "--strong-blur-prob",
        type=float,
        default=0.0,
        help="Probability of applying stronger blur than the default mild blur.",
    )
    parser.add_argument(
        "--box-jitter",
        type=float,
        default=0.0,
        help="Pixel jitter applied to stored character boxes to avoid over-regular boxes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_length_buckets(spec):
    buckets = []
    total = 0.0
    for item in spec.split(","):
        span, prob = item.split(":")
        lo, hi = span.split("-")
        prob = float(prob)
        buckets.append((int(lo), int(hi), prob))
        total += prob
    if not math.isclose(total, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        buckets = [(lo, hi, prob / total) for lo, hi, prob in buckets]
    return buckets


def sample_length(buckets):
    r = random.random()
    acc = 0.0
    for lo, hi, prob in buckets:
        acc += prob
        if r <= acc:
            return random.randint(lo, hi)
    lo, hi, _ = buckets[-1]
    return random.randint(lo, hi)


def load_char_frequency(label_paths):
    counter = Counter()
    for label_path in label_paths:
        path = Path(label_path)
        if not path.exists():
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        for split in ("train", "valid", "test"):
            for sample in data.get("ground_truth", {}).get(split, []):
                counter.update(sample.get("text", ""))
    if not counter:
        raise FileNotFoundError(f"No characters found from source labels: {label_paths}")
    return counter


def load_source_texts(label_paths):
    texts = []
    for label_path in label_paths:
        path = Path(label_path)
        if not path.exists():
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        for split in ("train", "valid", "test"):
            for sample in data.get("ground_truth", {}).get(split, []):
                text = sample.get("text", "")
                if text:
                    texts.append(text)
    return texts


def font_supports_char(font, char):
    bbox = font.getbbox(char)
    if bbox is None:
        return False
    return bbox[2] > bbox[0] and bbox[3] > bbox[1]


def build_sampler(counter, font):
    chars = []
    weights = []
    for char, count in counter.items():
        if char.strip() and font_supports_char(font, char):
            chars.append(char)
            weights.append(max(int(count), 1))
    if not chars:
        raise RuntimeError("The selected font does not support any source-label characters.")
    return chars, weights


def build_font_infos(font_paths, counter, min_font_size):
    infos = []
    for font_path in font_paths:
        path = Path(font_path)
        if not path.exists():
            print(f"skip missing font: {path}", flush=True)
            continue
        probe_font = ImageFont.truetype(str(path), min_font_size)
        chars, weights = build_sampler(counter, probe_font)
        infos.append(
            {
                "path": str(path),
                "chars": chars,
                "weights": weights,
                "char_set": set(chars),
            }
        )
        print(f"font {path}: supports {len(chars)} source chars", flush=True)
    if not infos:
        raise RuntimeError("No usable fonts found.")
    return infos


def collect_backgrounds(bg_dir):
    root = Path(bg_dir)
    if not root.exists():
        return []
    paths = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        paths.extend(root.glob(suffix))
    return sorted(paths)


def make_plain_background(width, height):
    base = random.randint(226, 246)
    arr = np.full((height, width, 3), base, dtype=np.uint8)
    noise = np.random.normal(0.0, random.uniform(2.0, 5.0), arr.shape)
    arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def make_background(width, height, bg_paths):
    if not bg_paths or random.random() < 0.10:
        return make_plain_background(width, height)

    path = random.choice(bg_paths)
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        return make_plain_background(width, height)

    if image.width < width or image.height < height:
        scale = max(width / image.width, height / image.height)
        new_size = (int(image.width * scale + 0.5), int(image.height * scale + 0.5))
        image = image.resize(new_size, Image.BICUBIC)

    x = random.randint(0, image.width - width)
    y = random.randint(0, image.height - height)
    return image.crop((x, y, x + width, y + height))


def sample_text(chars, weights, length):
    return "".join(random.choices(chars, weights=weights, k=length))


def sample_coverage_text(chars, weights, length, coverage_counts, coverage_min_count, forced_chars_per_sample):
    if coverage_min_count <= 0:
        return sample_text(chars, weights, length)

    undercovered = [char for char in chars if coverage_counts[char] < coverage_min_count]
    if not undercovered:
        return sample_text(chars, weights, length)

    forced_n = min(length, max(1, forced_chars_per_sample), len(undercovered))
    # Prioritize the least-covered characters, then randomize ties enough to avoid fixed ordering.
    undercovered.sort(key=lambda char: (coverage_counts[char], random.random()))
    forced = undercovered[:forced_n]
    filler = random.choices(chars, weights=weights, k=max(0, length - forced_n))
    sampled = forced + filler
    random.shuffle(sampled)
    return "".join(sampled)


def sample_real_text(source_texts, supported_chars, length):
    candidates = [text for text in source_texts if len(text) >= length]
    if not candidates:
        return None
    for _ in range(20):
        text = random.choice(candidates)
        start = random.randint(0, len(text) - length)
        piece = text[start : start + length]
        if all(char in supported_chars for char in piece):
            return piece
    return None


def choose_font_and_text(font_infos, counter, source_texts, target_len, coverage_counts, args):
    union_chars = sorted(set().union(*(info["char_set"] for info in font_infos)))
    undercovered = [
        char for char in union_chars if coverage_counts[char] < args.coverage_min_count
    ]

    forced = None
    if args.coverage_min_count > 0 and undercovered:
        undercovered.sort(key=lambda char: (coverage_counts[char], random.random()))
        forced = undercovered[0]
        eligible = [info for info in font_infos if forced in info["char_set"]]
        font_info = random.choice(eligible)
        filler = random.choices(
            font_info["chars"],
            weights=font_info["weights"],
            k=max(0, target_len - 1),
        )
        chars = [forced] + filler
        random.shuffle(chars)
        return font_info, "".join(chars)

    font_info = random.choice(font_infos)
    if source_texts and random.random() < args.real_text_ratio:
        text = sample_real_text(source_texts, font_info["char_set"], target_len)
        if text:
            return font_info, text
    text = sample_text(font_info["chars"], font_info["weights"], target_len)
    return font_info, text


def render_vertical_text(text, font_path, font_size, bg_paths, args):
    font = ImageFont.truetype(font_path, font_size)
    margin_x = random.randint(args.min_margin, args.max_margin)
    margin_y = random.randint(args.min_margin, args.max_margin)
    gap = random.randint(max(-2, -font_size // 12), max(2, font_size // 6))

    glyphs = []
    max_width = 1
    total_height = margin_y * 2
    for char in text:
        bbox = font.getbbox(char)
        width = max(1, bbox[2] - bbox[0])
        height = max(1, bbox[3] - bbox[1])
        glyphs.append((char, bbox, width, height))
        max_width = max(max_width, width)
        total_height += height + gap
    total_height = max(total_height - gap, font_size + 2 * margin_y)

    width = max_width + 2 * margin_x + random.randint(0, max(4, font_size // 3))
    height = min(max(total_height, font_size + 2 * margin_y), args.max_height)
    image = make_background(width, height, bg_paths)
    draw = ImageDraw.Draw(image)
    ink = random.randint(15, 55)
    fill = (ink, ink, ink)

    boxes = []
    y = margin_y
    for char, bbox, glyph_w, glyph_h in glyphs:
        if y + glyph_h > height - args.min_margin:
            break

        x = int((width - glyph_w) / 2)
        # Normalize negative glyph offsets so stored boxes match the visible ink.
        draw_x = x - bbox[0]
        draw_y = y - bbox[1]
        draw.text((draw_x, draw_y), char, font=font, fill=fill)
        jitter = args.box_jitter
        jx1 = random.uniform(-jitter, jitter) if jitter > 0 else 0.0
        jy1 = random.uniform(-jitter, jitter) if jitter > 0 else 0.0
        jx2 = random.uniform(-jitter, jitter) if jitter > 0 else 0.0
        jy2 = random.uniform(-jitter, jitter) if jitter > 0 else 0.0
        x1 = max(0, x + jx1)
        y1 = max(0, y + jy1)
        x2 = min(width, x + glyph_w + jx2)
        y2 = min(height, y + glyph_h + jy2)
        boxes.append((char, x1, y1, x2, y2))
        y += glyph_h + gap

    rendered_text = "".join(item[0] for item in boxes)
    image = augment_image(image, args)
    return image, rendered_text, boxes


def augment_image(image, args):
    if random.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.7)))
    if random.random() < args.strong_blur_prob:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.8, 1.4)))

    arr = np.asarray(image).astype(np.float32)
    if random.random() < 0.70:
        contrast = random.uniform(0.88, 1.12)
        brightness = random.uniform(-8.0, 8.0)
        mean = arr.mean(axis=(0, 1), keepdims=True)
        arr = (arr - mean) * contrast + mean + brightness
    if random.random() < 0.45:
        arr += np.random.normal(0.0, random.uniform(1.0, 4.0), arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, "RGB")

    if random.random() < args.erase_prob:
        draw = ImageDraw.Draw(image)
        for _ in range(random.randint(1, 3)):
            if random.random() < 0.6:
                w = random.randint(2, max(3, image.width // 5))
                x = random.randint(0, max(0, image.width - w))
                y0 = random.randint(0, max(0, image.height - 1))
                h = random.randint(max(4, image.height // 30), max(5, image.height // 10))
                color = random.randint(215, 248)
                draw.rectangle((x, y0, x + w, min(image.height, y0 + h)), fill=(color, color, color))
            else:
                h = random.randint(2, max(3, image.height // 30))
                y = random.randint(0, max(0, image.height - h))
                color = random.randint(215, 248)
                draw.rectangle((0, y, image.width, y + h), fill=(color, color, color))
    return image


def write_textline(raw_root, page_id, text, width, height):
    path = raw_root / "label_textline" / f"{page_id}.txt"
    path.write_text(f"{text},0,0,{width},0,{width},{height},0,{height}\n", encoding="utf-8")


def write_char_boxes(raw_root, page_id, boxes):
    path = raw_root / "label_char" / f"{page_id}.txt"
    with path.open("w", encoding="utf-8") as f:
        for char, x1, y1, x2, y2 in boxes:
            f.write(f"{char} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")


def prepare_dirs(output_root, raw_root, overwrite):
    output_root = Path(output_root)
    raw_root = Path(raw_root)
    paths = [
        output_root / "lines",
        raw_root / "label_textline",
        raw_root / "label_char",
    ]
    if not overwrite:
        existing = [path for path in paths if path.exists() and any(path.iterdir())]
        if existing:
            raise FileExistsError(
                "Output directories are not empty. Pass --overwrite to append/replace files: "
                + ", ".join(str(path) for path in existing)
            )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return output_root, raw_root


def write_dataset_files(output_root, labels, args, font_infos, bg_paths, counter, rendered_char_counter, length_counter):
    with (output_root / "labels.pkl").open("wb") as f:
        pickle.dump(labels, f)

    rendered_charset = sorted(rendered_char_counter)
    charset_path = output_root / "charset.pkl"
    with charset_path.open("wb") as f:
        pickle.dump(rendered_charset, f)

    char_freq_path = output_root / "char_frequency.json"
    char_freq_path.write_text(
        json.dumps(dict(sorted(rendered_char_counter.items())), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta = {
        "num_images": args.num_images,
        "generated_images": sum(len(items) for items in labels["ground_truth"].values()),
        "splits": {split: len(items) for split, items in labels["ground_truth"].items()},
        "font_paths": [info["path"] for info in font_infos],
        "bg_dir": args.bg_dir,
        "num_backgrounds": len(bg_paths),
        "source_labels": args.source_labels,
        "source_charset_size": len(counter),
        "font_supported_source_charset_size": len(set().union(*(info["char_set"] for info in font_infos))),
        "font_supported_sizes": {info["path"]: len(info["chars"]) for info in font_infos},
        "rendered_charset_size": len(rendered_charset),
        "coverage_min_count": args.coverage_min_count,
        "forced_chars_per_sample": args.forced_chars_per_sample,
        "real_text_ratio": args.real_text_ratio,
        "erase_prob": args.erase_prob,
        "strong_blur_prob": args.strong_blur_prob,
        "box_jitter": args.box_jitter,
        "min_rendered_char_count": min(rendered_char_counter.values()) if rendered_char_counter else 0,
        "char_frequency": str(char_freq_path),
        "charset_path": str(charset_path),
        "length_buckets": args.length_buckets,
        "length_hist_capped20": dict(sorted(length_counter.items())),
    }
    (output_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def split_name(index, num_images, valid_ratio, test_ratio):
    valid_start = int(num_images * (1.0 - valid_ratio - test_ratio))
    test_start = int(num_images * (1.0 - test_ratio))
    if index >= test_start:
        return "test"
    if index >= valid_start:
        return "valid"
    return "train"


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_root, raw_root = prepare_dirs(args.output_root, args.raw_root, args.overwrite)
    counter = load_char_frequency(args.source_labels)
    source_texts = load_source_texts(args.source_labels)
    font_paths = args.font_paths if args.font_paths else [args.font_path]
    font_infos = build_font_infos(font_paths, counter, args.min_font_size)
    bg_paths = collect_backgrounds(args.bg_dir)
    buckets = parse_length_buckets(args.length_buckets)

    labels = {"ground_truth": {"train": [], "valid": [], "test": []}}
    length_counter = Counter()
    rendered_char_counter = Counter()
    coverage_counts = Counter()
    attempts = 0
    index = 0
    while index < args.num_images:
        attempts += 1
        if attempts > args.num_images * 20:
            raise RuntimeError(f"Only generated {index}/{args.num_images}; check font/size settings.")

        target_len = sample_length(buckets)
        font_info, text = choose_font_and_text(
            font_infos,
            counter,
            source_texts,
            target_len,
            coverage_counts,
            args,
        )
        font_size = random.randint(args.min_font_size, args.max_font_size)
        image, rendered_text, boxes = render_vertical_text(text, font_info["path"], font_size, bg_paths, args)
        if not rendered_text or len(boxes) != len(rendered_text):
            continue

        page_id = f"synth{index:06d}"
        sample_id = f"{page_id}_000"
        image.save(output_root / "lines" / f"{sample_id}.jpg", quality=args.jpeg_quality)
        write_textline(raw_root, page_id, rendered_text, image.width, image.height)
        write_char_boxes(raw_root, page_id, boxes)

        split = split_name(index, args.num_images, args.valid_ratio, args.test_ratio)
        labels["ground_truth"][split].append(
            {"id": sample_id, "text": rendered_text, "direction": "vertical"}
        )
        length_counter[min(len(rendered_text), 20)] += 1
        rendered_char_counter.update(rendered_text)
        coverage_counts.update(rendered_text)
        index += 1
        if index % 500 == 0:
            print(f"generated {index}/{args.num_images}", flush=True)
        if args.save_every > 0 and index % args.save_every == 0:
            write_dataset_files(
                output_root,
                labels,
                args,
                font_infos,
                bg_paths,
                counter,
                rendered_char_counter,
                length_counter,
            )

    meta = write_dataset_files(
        output_root,
        labels,
        args,
        font_infos,
        bg_paths,
        counter,
        rendered_char_counter,
        length_counter,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
