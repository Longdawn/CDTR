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
    parser = argparse.ArgumentParser("Generate MTH-style vertical ancient semi-synthetic data.")
    parser.add_argument("--glyph-bank", default="data/ancient_glyph_bank_mth")
    parser.add_argument("--output-root", default="data/synth_vertical_ancient_smoke_dtlr")
    parser.add_argument("--raw-root", default="data/SynthVerticalAncientSmoke")
    parser.add_argument("--num-images", type=int, default=5000)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--charset", default="data/tkhmth2200_mth1000_mth1200_charset.pkl")
    parser.add_argument(
        "--source-labels",
        nargs="+",
        default=[
            "data/tkhmth2200_mth1000_dtlr/labels.pkl",
            "data/tkhmth2200_mth1200_dtlr/labels.pkl",
        ],
    )
    parser.add_argument(
        "--font-paths",
        nargs="*",
        default=[
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Medium.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "text_renderer/example_data/font/simsun.ttf",
        ],
    )
    parser.add_argument(
        "--length-buckets",
        default="1-2:0.20,3-5:0.25,6-15:0.35,16-32:0.20",
    )
    parser.add_argument("--real-text-ratio", type=float, default=0.55)
    parser.add_argument("--coverage-min-count", type=int, default=3)
    parser.add_argument("--fallback-font-ratio", type=float, default=0.12)
    parser.add_argument("--min-margin", type=int, default=6)
    parser.add_argument("--max-margin", type=int, default=28)
    parser.add_argument("--max-height", type=int, default=1200)
    parser.add_argument("--target-char-height", default="38-62")
    parser.add_argument("--gap-ratio", default="-0.12:0.24")
    parser.add_argument("--paper-tint", default="218-248")
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_charset(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return [char for char, _ in sorted(obj.items(), key=lambda kv: kv[1])]
    return list(obj)


def parse_range(spec, cast=float):
    if ":" in spec:
        lo, hi = spec.split(":", maxsplit=1)
    else:
        lo, hi = spec.split("-", maxsplit=1)
    return cast(lo), cast(hi)


def parse_length_buckets(spec):
    buckets = []
    total = 0.0
    for item in spec.split(","):
        span, prob = item.split(":")
        lo, hi = span.split("-")
        prob = float(prob)
        buckets.append((int(lo), int(hi), prob))
        total += prob
    return [(lo, hi, prob / total) for lo, hi, prob in buckets]


def sample_length(buckets):
    r = random.random()
    acc = 0.0
    for lo, hi, prob in buckets:
        acc += prob
        if r <= acc:
            return random.randint(lo, hi)
    lo, hi, _ = buckets[-1]
    return random.randint(lo, hi)


def load_glyph_index(glyph_bank):
    root = Path(glyph_bank)
    index_path = root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing glyph index: {index_path}")
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    return {char: records for char, records in raw.items() if records}


def load_source_texts(label_paths):
    texts = []
    char_counter = Counter()
    for label_path in label_paths:
        with open(label_path, "rb") as f:
            data = pickle.load(f)
        for split in ("train", "valid", "test"):
            for sample in data.get("ground_truth", {}).get(split, []):
                text = sample.get("text", "")
                if text:
                    texts.append(text)
                    char_counter.update(text)
    if not texts:
        raise RuntimeError("No source texts loaded.")
    return texts, char_counter


def font_supports(font_path, char, font_size=48):
    try:
        font = ImageFont.truetype(str(font_path), font_size)
        bbox = font.getbbox(char)
    except Exception:
        return False
    return bbox is not None and bbox[2] > bbox[0] and bbox[3] > bbox[1]


def usable_fonts(font_paths):
    return [Path(p) for p in font_paths if Path(p).exists()]


def sample_real_substring(texts, available_chars, length):
    candidates = [t for t in texts if len(t) >= length]
    for _ in range(50):
        text = random.choice(candidates)
        start = random.randint(0, len(text) - length)
        piece = text[start : start + length]
        if all(c in available_chars for c in piece):
            return piece
    return None


def sample_text(length, texts, char_weights, available_chars, coverage_counts, coverage_min_count, real_text_ratio):
    under = [c for c in available_chars if coverage_counts[c] < coverage_min_count]
    if under:
        under.sort(key=lambda c: (coverage_counts[c], random.random()))
        forced = under[0]
        rest = random.choices(list(char_weights), weights=list(char_weights.values()), k=max(0, length - 1))
        chars = [forced] + [c for c in rest if c in available_chars]
        while len(chars) < length:
            chars.append(random.choice(tuple(available_chars)))
        random.shuffle(chars)
        return "".join(chars[:length])

    if random.random() < real_text_ratio:
        piece = sample_real_substring(texts, available_chars, length)
        if piece:
            return piece
    chars = random.choices(list(char_weights), weights=list(char_weights.values()), k=length)
    return "".join(c if c in available_chars else random.choice(tuple(available_chars)) for c in chars)


def make_paper_background(width, height, tint_range):
    lo, hi = tint_range
    base = random.randint(lo, hi)
    arr = np.full((height, width, 3), base, dtype=np.float32)
    # Low-frequency paper texture.
    small = np.random.normal(0.0, random.uniform(3.0, 8.0), (max(2, height // 16), max(2, width // 16), 1))
    tex = Image.fromarray(np.clip(small.squeeze() + 128, 0, 255).astype(np.uint8), "L")
    tex = tex.resize((width, height), Image.BICUBIC)
    tex_arr = np.asarray(tex, dtype=np.float32) - 128.0
    arr += tex_arr[:, :, None]
    arr += np.random.normal(0.0, random.uniform(1.0, 3.0), arr.shape)
    # Mild vertical aging bands and stains.
    if random.random() < 0.45:
        x = random.randint(0, max(0, width - 1))
        band_w = random.randint(2, max(3, width // 5))
        arr[:, x : min(width, x + band_w), :] += random.uniform(-10, 10)
    if random.random() < 0.35:
        yy, xx = np.ogrid[:height, :width]
        cx, cy = random.randint(0, width - 1), random.randint(0, height - 1)
        rx, ry = random.uniform(width * 0.2, width * 0.8), random.uniform(height * 0.05, height * 0.25)
        stain = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 < 1.0
        arr[stain] += random.uniform(-12, 8)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def load_glyph(glyph_bank, record, target_height):
    glyph = Image.open(Path(glyph_bank) / record["path"]).convert("RGBA")
    alpha_bbox = glyph.getbbox()
    if alpha_bbox is None:
        return None
    visible_h = max(1, alpha_bbox[3] - alpha_bbox[1])
    scale = target_height / visible_h * random.uniform(0.88, 1.12)
    new_size = (
        max(2, int(round(glyph.width * scale))),
        max(2, int(round(glyph.height * scale))),
    )
    return glyph.resize(new_size, Image.BICUBIC)


def render_font_char(char, font_paths, target_height):
    font_path = random.choice(font_paths)
    font_size = max(12, int(target_height * random.uniform(0.95, 1.20)))
    font = ImageFont.truetype(str(font_path), font_size)
    bbox = font.getbbox(char)
    if bbox is None:
        return None
    w = max(2, bbox[2] - bbox[0] + 8)
    h = max(2, bbox[3] - bbox[1] + 8)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    ink = random.randint(15, 70)
    draw.text((4 - bbox[0], 4 - bbox[1]), char, font=font, fill=(ink, ink, ink, 255))
    return img


def paste_rgba(base, glyph, x, y):
    alpha_bbox = glyph.getbbox()
    if alpha_bbox is None:
        return None
    base.alpha_composite(glyph, (x, y))
    return (x + alpha_bbox[0], y + alpha_bbox[1], x + alpha_bbox[2], y + alpha_bbox[3])


def augment(image):
    if random.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.15, 0.75)))
    arr = np.asarray(image).astype(np.float32)
    if random.random() < 0.75:
        contrast = random.uniform(0.82, 1.15)
        brightness = random.uniform(-10, 10)
        mean = arr.mean(axis=(0, 1), keepdims=True)
        arr = (arr - mean) * contrast + mean + brightness
    if random.random() < 0.45:
        arr += np.random.normal(0.0, random.uniform(1.0, 4.5), arr.shape)
    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
    if random.random() < 0.22:
        draw = ImageDraw.Draw(image)
        for _ in range(random.randint(1, 3)):
            if random.random() < 0.55:
                y = random.randint(0, max(0, image.height - 2))
                h = random.randint(1, max(2, image.height // 40))
                color = random.randint(215, 250)
                draw.rectangle((0, y, image.width, min(image.height, y + h)), fill=(color, color, color))
            else:
                x = random.randint(0, max(0, image.width - 2))
                w = random.randint(1, max(2, image.width // 8))
                color = random.randint(215, 250)
                draw.rectangle((x, 0, min(image.width, x + w), image.height), fill=(color, color, color))
    return image


def write_outputs(output_root, raw_root, labels, meta):
    with (output_root / "labels.pkl").open("wb") as f:
        pickle.dump(labels, f)
    with (output_root / "charset.pkl").open("wb") as f:
        pickle.dump(meta["charset"], f)
    (output_root / "meta.json").write_text(
        json.dumps({k: v for k, v in meta.items() if k != "charset"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def split_name(index, num_images, valid_ratio, test_ratio):
    valid_start = int(num_images * (1.0 - valid_ratio - test_ratio))
    test_start = int(num_images * (1.0 - test_ratio))
    if index >= test_start:
        return "test"
    if index >= valid_start:
        return "valid"
    return "train"


def prepare_dirs(output_root, raw_root, overwrite):
    output_root = Path(output_root)
    raw_root = Path(raw_root)
    targets = [output_root / "lines", raw_root / "label_textline", raw_root / "label_char"]
    if not overwrite:
        existing = [p for p in targets if p.exists() and any(p.iterdir())]
        if existing:
            raise FileExistsError("Non-empty output dirs: " + ", ".join(str(p) for p in existing))
    for p in targets:
        p.mkdir(parents=True, exist_ok=True)
    return output_root, raw_root


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_root, raw_root = prepare_dirs(args.output_root, args.raw_root, args.overwrite)
    charset = load_charset(args.charset)
    glyph_index = load_glyph_index(args.glyph_bank)
    texts, char_counter = load_source_texts(args.source_labels)
    font_paths = usable_fonts(args.font_paths)
    available = {c for c in charset if c in glyph_index}
    if font_paths:
        # Allow font fallback for charset chars without real crops.
        available |= {c for c in charset if random.random() < 1.0 and any(font_supports(p, c) for p in font_paths)}
    if not available:
        raise RuntimeError("No available characters from glyph bank/fonts.")
    char_weights = Counter({c: max(1, char_counter.get(c, 1)) for c in available})
    buckets = parse_length_buckets(args.length_buckets)
    height_range = parse_range(args.target_char_height, int)
    gap_lo, gap_hi = parse_range(args.gap_ratio, float)
    tint_range = parse_range(args.paper_tint, int)

    labels = {"ground_truth": {"train": [], "valid": [], "test": []}}
    rendered_counter = Counter()
    length_counter = Counter()
    coverage_counts = Counter()
    glyph_use = 0
    font_use = 0

    for index in range(args.num_images):
        target_len = sample_length(buckets)
        text = sample_text(
            target_len,
            texts,
            char_weights,
            available,
            coverage_counts,
            args.coverage_min_count,
            args.real_text_ratio,
        )
        target_h = random.randint(*height_range)
        margin_x = random.randint(args.min_margin, args.max_margin)
        margin_y = random.randint(args.min_margin, args.max_margin)

        glyphs = []
        for char in text:
            glyph = None
            if char in glyph_index and random.random() > args.fallback_font_ratio:
                glyph = load_glyph(args.glyph_bank, random.choice(glyph_index[char]), target_h)
                glyph_use += 1 if glyph is not None else 0
            if glyph is None and font_paths:
                supported = [p for p in font_paths if font_supports(p, char, target_h)]
                if supported:
                    glyph = render_font_char(char, supported, target_h)
                    font_use += 1 if glyph is not None else 0
            if glyph is not None:
                glyphs.append((char, glyph))
        if not glyphs:
            continue

        gaps = [int(round(random.uniform(gap_lo, gap_hi) * target_h)) for _ in range(max(0, len(glyphs) - 1))]
        width = max(g.width for _, g in glyphs) + 2 * margin_x + random.randint(0, max(2, target_h // 3))
        height = margin_y * 2 + sum(g.height for _, g in glyphs) + sum(gaps)
        if height > args.max_height:
            # Keep the beginning of the column, matching line-crop truncation behavior.
            kept = []
            h_acc = margin_y * 2
            for i, item in enumerate(glyphs):
                extra = item[1].height + (gaps[i] if i < len(gaps) else 0)
                if h_acc + extra > args.max_height:
                    break
                kept.append(item)
                h_acc += extra
            glyphs = kept
            gaps = gaps[: max(0, len(glyphs) - 1)]
            height = max(target_h + 2 * margin_y, h_acc)
        if not glyphs:
            continue
        height = max(height, target_h + 2 * margin_y)
        width = max(width, 24)

        canvas = make_paper_background(width, height, tint_range).convert("RGBA")
        boxes = []
        y = margin_y
        for i, (char, glyph) in enumerate(glyphs):
            x = int((width - glyph.width) / 2 + random.uniform(-2.0, 2.0))
            box = paste_rgba(canvas, glyph, max(0, x), max(0, y))
            if box is not None:
                x1, y1, x2, y2 = box
                boxes.append((char, max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
            y += glyph.height + (gaps[i] if i < len(gaps) else 0)
        if not boxes:
            continue

        rendered_text = "".join(c for c, *_ in boxes)
        image = augment(canvas.convert("RGB"))
        page_id = f"synth{index:06d}"
        sample_id = f"{page_id}_000"
        image.save(output_root / "lines" / f"{sample_id}.jpg", quality=args.jpeg_quality)
        (raw_root / "label_textline" / f"{page_id}.txt").write_text(
            f"{rendered_text},0,0,{width},0,{width},{height},0,{height}\n",
            encoding="utf-8",
        )
        with (raw_root / "label_char" / f"{page_id}.txt").open("w", encoding="utf-8") as f:
            for char, x1, y1, x2, y2 in boxes:
                f.write(f"{char} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")

        split = split_name(index, args.num_images, args.valid_ratio, args.test_ratio)
        labels["ground_truth"][split].append({"id": sample_id, "text": rendered_text, "direction": "vertical"})
        rendered_counter.update(rendered_text)
        coverage_counts.update(rendered_text)
        length_counter[min(len(rendered_text), 40)] += 1
        if (index + 1) % 500 == 0:
            print(f"generated {index + 1}/{args.num_images}", flush=True)
        if args.save_every > 0 and (index + 1) % args.save_every == 0:
            write_outputs(
                output_root,
                raw_root,
                labels,
                {
                    "charset": charset,
                    "num_images": args.num_images,
                    "generated_images": sum(len(v) for v in labels["ground_truth"].values()),
                    "splits": {k: len(v) for k, v in labels["ground_truth"].items()},
                    "glyph_bank": args.glyph_bank,
                    "glyph_use": glyph_use,
                    "font_use": font_use,
                    "rendered_charset_size": len(rendered_counter),
                    "min_rendered_char_count": min(rendered_counter.values()) if rendered_counter else 0,
                    "length_hist_capped40": dict(sorted(length_counter.items())),
                },
            )

    meta = {
        "charset": charset,
        "num_images": args.num_images,
        "generated_images": sum(len(v) for v in labels["ground_truth"].values()),
        "splits": {k: len(v) for k, v in labels["ground_truth"].items()},
        "glyph_bank": args.glyph_bank,
        "glyph_use": glyph_use,
        "font_use": font_use,
        "rendered_charset_size": len(rendered_counter),
        "min_rendered_char_count": min(rendered_counter.values()) if rendered_counter else 0,
        "length_hist_capped40": dict(sorted(length_counter.items())),
        "args": vars(args),
    }
    write_outputs(output_root, raw_root, labels, meta)
    print(json.dumps({k: v for k, v in meta.items() if k != "charset"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
