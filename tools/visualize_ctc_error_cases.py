import argparse
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import build_dataset
from util.slconfig import DictAction, SLConfig


def parse_args():
    parser = argparse.ArgumentParser("Visualize and summarize CTC error cases")
    parser.add_argument("--config_file", "-c", required=True)
    parser.add_argument("--dataset_file", default="mth1000")
    parser.add_argument("--split", default="test", choices=["train", "valid", "val", "test"])
    parser.add_argument("--cases", required=True, help="JSONL from tools/analyze_ctc_errors.py")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=80)
    parser.add_argument("--short_k", type=int, default=80)
    parser.add_argument("--copy_images", action="store_true")
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="Override config values, same format as finetuning.py --options.",
    )
    return parser.parse_args()


def load_cfg_to_args(cli):
    cfg = SLConfig.fromfile(cli.config_file)
    if cli.options is not None:
        cfg.merge_from_dict(cli.options)
    cfg_dict = cfg._cfg_dict.to_dict()
    base = vars(cli).copy()
    for key, value in cfg_dict.items():
        if key not in base:
            base[key] = value
    defaults = {
        "mode_chr": True,
        "amp": False,
        "eval": True,
        "distributed": False,
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "fix_size": False,
        "coco_path": "data/COCO2017",
    }
    for key, value in defaults.items():
        if key not in base:
            base[key] = value
    return argparse.Namespace(**base)


def read_cases(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def edit_alignment(gt, pred):
    n, m = len(gt), len(pred)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
        op[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        op[0][j] = "ins"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gt[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                op[i][j] = "eq"
            else:
                candidates = [
                    (dp[i - 1][j] + 1, "del"),
                    (dp[i][j - 1] + 1, "ins"),
                    (dp[i - 1][j - 1] + 1, "sub"),
                ]
                dp[i][j], op[i][j] = min(candidates, key=lambda item: item[0])

    i, j = n, m
    steps = []
    while i > 0 or j > 0:
        step = op[i][j]
        if step == "eq":
            steps.append(("eq", gt[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif step == "sub":
            steps.append(("sub", gt[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif step == "del":
            steps.append(("del", gt[i - 1], ""))
            i -= 1
        elif step == "ins":
            steps.append(("ins", "", pred[j - 1]))
            j -= 1
        else:
            break
    steps.reverse()
    return steps


def error_type(row):
    if row["gt"] == row["pred"]:
        return "correct"
    if row["pred_len"] == 0:
        return "empty_pred"
    ins, dels, subs = int(row["ins"]), int(row["del"]), int(row["sub"])
    parts = []
    if dels:
        parts.append(("deletion", dels))
    if subs:
        parts.append(("substitution", subs))
    if ins:
        parts.append(("insertion", ins))
    if len(parts) == 1:
        return parts[0][0]
    parts.sort(key=lambda item: item[1], reverse=True)
    if len(parts) >= 2 and parts[0][1] == parts[1][1]:
        return "mixed"
    return f"{parts[0][0]}_dominant"


def case_image_path(dataset, idx):
    sample = dataset.samples[idx]
    image_id = sample["id"]
    return Path(dataset.images_path) / f"{image_id}.{dataset.im_extension}"


def resize_for_contact(image_path, width=180, height=260):
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((width, height))
    canvas = Image.new("RGB", (width, height), "white")
    x = (width - img.width) // 2
    y = (height - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def make_contact_sheet(rows, dataset, out_path, title, max_items=40, cols=5):
    rows = rows[:max_items]
    if not rows:
        return None
    cell_w, cell_h = 220, 320
    header_h = 22
    rows_n = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows_n * cell_h), "white")
    draw = ImageDraw.Draw(sheet)
    for pos, row in enumerate(rows):
        x0 = (pos % cols) * cell_w
        y0 = (pos // cols) * cell_h
        img_path = case_image_path(dataset, int(row["idx"]))
        try:
            tile = resize_for_contact(img_path, width=cell_w, height=cell_h - header_h)
        except Exception:
            tile = Image.new("RGB", (cell_w, cell_h - header_h), "lightgray")
        sheet.paste(tile, (x0, y0 + header_h))
        label = f"idx={row['idx']} L={row['gt_len']} cer={row['cer']:.2f}"
        draw.text((x0 + 4, y0 + 4), label, fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def save_case_images(rows, dataset, out_dir, prefix, limit):
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for rank, row in enumerate(rows[:limit], start=1):
        src = case_image_path(dataset, int(row["idx"]))
        safe = f"{prefix}_{rank:03d}_idx{int(row['idx']):05d}_L{row['gt_len']}_cer{row['cer']:.2f}{src.suffix}"
        dst = out_dir / safe
        if src.exists():
            shutil.copyfile(src, dst)
            saved.append((row, dst))
    return saved


def pct(value):
    return f"{100.0 * value:.2f}%"


def main():
    cli = parse_args()
    args = load_cfg_to_args(cli)
    split = "valid" if cli.split == "val" else cli.split
    dataset = build_dataset(image_set=split, args=args)
    rows = read_cases(cli.cases)
    out_dir = Path(cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        row["error_type"] = error_type(row)

    errors = [row for row in rows if row["gt"] != row["pred"]]
    correct = len(rows) - len(errors)
    by_type = Counter(row["error_type"] for row in rows)
    by_len = defaultdict(list)
    for row in rows:
        if row["gt_len"] == 1:
            key = "1"
        elif row["gt_len"] == 2:
            key = "2"
        elif row["gt_len"] <= 5:
            key = "3-5"
        elif row["gt_len"] <= 10:
            key = "6-10"
        else:
            key = "11+"
        by_len[key].append(row)

    sub_pairs = Counter()
    deleted_chars = Counter()
    inserted_chars = Counter()
    for row in errors:
        for op, gt_ch, pred_ch in edit_alignment(row["gt"], row["pred"]):
            if op == "sub":
                sub_pairs[(gt_ch, pred_ch)] += 1
            elif op == "del":
                deleted_chars[gt_ch] += 1
            elif op == "ins":
                inserted_chars[pred_ch] += 1

    worst = sorted(errors, key=lambda row: (row["cer"], row["gt_len"]), reverse=True)
    short_errors = sorted(
        [row for row in errors if row["gt_len"] <= 2],
        key=lambda row: (row["cer"], -row["gt_len"]),
        reverse=True,
    )
    len11_errors = sorted(
        [row for row in errors if row["gt_len"] >= 11],
        key=lambda row: row["cer"],
        reverse=True,
    )

    images_dir = out_dir / "case_images"
    saved_worst = save_case_images(worst, dataset, images_dir, "worst", cli.top_k) if cli.copy_images else []
    saved_short = save_case_images(short_errors, dataset, images_dir, "short", cli.short_k) if cli.copy_images else []

    make_contact_sheet(worst, dataset, out_dir / "worst_contact_sheet.jpg", "worst", max_items=40)
    make_contact_sheet(short_errors, dataset, out_dir / "short_contact_sheet.jpg", "short", max_items=40)
    make_contact_sheet(len11_errors, dataset, out_dir / "len11_error_contact_sheet.jpg", "len11", max_items=40)

    summary = {
        "cases": len(rows),
        "correct": correct,
        "errors": len(errors),
        "exact_match": correct / max(len(rows), 1),
        "error_type_counts": dict(by_type),
        "length_bins": {
            key: {
                "samples": len(items),
                "errors": sum(1 for item in items if item["gt"] != item["pred"]),
                "exact_match": sum(1 for item in items if item["gt"] == item["pred"]) / max(len(items), 1),
                "avg_cer": sum(float(item["cer"]) for item in items) / max(len(items), 1),
            }
            for key, items in by_len.items()
        },
        "top_substitutions": [
            {"gt": gt, "pred": pred, "count": count}
            for (gt, pred), count in sub_pairs.most_common(30)
        ],
        "top_deletions": [
            {"gt": gt, "count": count}
            for gt, count in deleted_chars.most_common(30)
        ],
        "top_insertions": [
            {"pred": pred, "count": count}
            for pred, count in inserted_chars.most_common(30)
        ],
    }
    (out_dir / "error_visual_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = []
    report.append("# CTC Error Visualization Report")
    report.append("")
    report.append(f"- cases: `{len(rows)}`")
    report.append(f"- exact match: `{pct(summary['exact_match'])}`")
    report.append(f"- error cases: `{len(errors)}`")
    report.append("")
    report.append("## Error Types")
    for key, count in by_type.most_common():
        report.append(f"- `{key}`: `{count}` ({pct(count / max(len(rows), 1))})")
    report.append("")
    report.append("## Length Bins")
    for key in ["1", "2", "3-5", "6-10", "11+"]:
        if key not in summary["length_bins"]:
            continue
        item = summary["length_bins"][key]
        report.append(
            f"- `{key}`: samples `{item['samples']}`, errors `{item['errors']}`, "
            f"exact `{pct(item['exact_match'])}`, avg CER `{item['avg_cer']:.4f}`"
        )
    report.append("")
    report.append("## Contact Sheets")
    report.append("- [Worst errors](worst_contact_sheet.jpg)")
    report.append("- [Short errors](short_contact_sheet.jpg)")
    report.append("- [Len>=11 errors](len11_error_contact_sheet.jpg)")
    report.append("")
    report.append("## Top Substitutions")
    for item in summary["top_substitutions"][:20]:
        report.append(f"- `{item['gt']}` -> `{item['pred']}`: `{item['count']}`")
    report.append("")
    report.append("## Top Deletions")
    for item in summary["top_deletions"][:20]:
        report.append(f"- `{item['gt']}`: `{item['count']}`")
    report.append("")
    report.append("## Top Insertions")
    for item in summary["top_insertions"][:20]:
        report.append(f"- `{item['pred']}`: `{item['count']}`")
    report.append("")
    report.append("## Worst Cases")
    for row in worst[: cli.top_k]:
        report.append(
            f"- idx `{row['idx']}`, len `{row['gt_len']}`, type `{row['error_type']}`, "
            f"CER `{row['cer']:.3f}`"
        )
        report.append(f"  GT: `{row['gt']}`")
        report.append(f"  Pred: `{row['pred']}`")
    report.append("")
    report.append("## Short Error Cases")
    for row in short_errors[: cli.short_k]:
        report.append(
            f"- idx `{row['idx']}`, len `{row['gt_len']}`, type `{row['error_type']}`, "
            f"CER `{row['cer']:.3f}`, GT `{row['gt']}`, Pred `{row['pred']}`"
        )

    (out_dir / "error_visual_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved report -> {out_dir / 'error_visual_report.md'}")
    print(f"Saved contact sheets -> {out_dir}")
    if cli.copy_images:
        print(f"Copied {len(saved_worst) + len(saved_short)} case images -> {images_dir}")


if __name__ == "__main__":
    main()
