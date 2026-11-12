import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser("Visualize MMOCR text recognition predictions")
    parser.add_argument("--cases", required=True, help="JSONL produced by eval_mmocr_textrecog_predictions.py")
    parser.add_argument("--ann", required=True, help="MMOCR textrecog annotation JSON")
    parser.add_argument("--data_root", required=True, help="MMOCR data root")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--font", default="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    parser.add_argument("--rows", type=int, default=24)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_ann(path, data_root):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for item in data["data_list"]:
        img_path = Path(data_root) / item["img_path"]
        text = item["instances"][0]["text"]
        records.append({"img_path": img_path, "text": text})
    return records


def fit_image(img, max_w, max_h):
    img = img.convert("RGB")
    w, h = img.size
    scale = min(max_w / max(w, 1), max_h / max(h, 1), 1.0)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def draw_wrapped(draw, xy, text, font, fill, max_width, line_gap=4):
    x, y = xy
    lines = []
    current = ""
    for ch in text:
        candidate = current + ch
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = ch
    if current:
        lines.append(current)
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y = bbox[3] + line_gap
    return y


def make_sheet(name, cases, records, out_dir, font_path, rows):
    selected_cases = cases[:rows]
    font = ImageFont.truetype(font_path, 22)
    small_font = ImageFont.truetype(font_path, 18)
    title_font = ImageFont.truetype(font_path, 26)

    thumb_w = 140
    row_h = 360
    text_w = 1060
    pad = 18
    width = thumb_w + text_w + pad * 3
    height = row_h * len(selected_cases) + pad * 2 + 42
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, pad), name, font=title_font, fill=(20, 20, 20))

    y0 = pad + 42
    for row_i, case in enumerate(selected_cases):
        y = y0 + row_i * row_h
        idx = case["idx"]
        record = records[idx]
        img_box = (pad, y + pad)
        try:
            thumb = fit_image(Image.open(record["img_path"]), thumb_w, row_h - 2 * pad)
        except Exception:
            thumb = Image.new("RGB", (thumb_w, row_h - 2 * pad), (240, 240, 240))
        thumb_x = img_box[0] + (thumb_w - thumb.size[0]) // 2
        thumb_y = img_box[1] + (row_h - 2 * pad - thumb.size[1]) // 2
        canvas.paste(thumb, (thumb_x, thumb_y))
        draw.rectangle((pad, y + pad, pad + thumb_w, y + row_h - pad), outline=(210, 210, 210))

        tx = pad * 2 + thumb_w
        ty = y + pad
        meta = (
            f"idx={idx}  gt_len={case['gt_len']}  pred_len={case['pred_len']}  "
            f"CER={case['cer']:.3f}  I/D/S={case['ins']}/{case['del']}/{case['sub']}"
        )
        draw.text((tx, ty), meta, font=small_font, fill=(80, 80, 80))
        ty += 30
        ty = draw_wrapped(draw, (tx, ty), "GT:   " + case["gt"], font, (0, 90, 0), text_w)
        ty += 8
        draw_wrapped(draw, (tx, ty), "PRED: " + case["pred"], font, (170, 20, 20), text_w)
        draw.line((pad, y + row_h - 1, width - pad, y + row_h - 1), fill=(230, 230, 230))

    out_path = Path(out_dir) / f"{name}.png"
    canvas.save(out_path)
    return out_path


def main():
    args = parse_args()
    cases = read_jsonl(args.cases)
    records = load_ann(args.ann, args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exact = sorted(cases, key=lambda item: item["cer"])
    sheets = {
        "top_errors": cases,
        "len1_errors": [row for row in cases if row["gt_len"] == 1],
        "len2_errors": [row for row in cases if row["gt_len"] == 2],
        "good_examples": exact[: args.rows],
    }
    written = []
    for name, selected in sheets.items():
        if selected:
            written.append(make_sheet(name, selected, records, out_dir, args.font, args.rows))

    index = {
        "cases": args.cases,
        "ann": args.ann,
        "data_root": args.data_root,
        "outputs": [str(path) for path in written],
    }
    (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
