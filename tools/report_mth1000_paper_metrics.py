import argparse
import json
from pathlib import Path


def load_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def pick_best_epoch(rows):
    if not rows:
        raise ValueError("log file has no JSON rows")
    return min(rows, key=lambda row: float(row["test_cer_oracle_direction"]))


def build_report(best_row, ctc_summary, run_dir, ctc_summary_path):
    by_bin = ctc_summary.get("by_gt_len_bin", {})

    report = {
        "run_dir": str(run_dir),
        "best_epoch": int(best_row["epoch"]),
        "best_test_cer_oracle_direction": float(best_row["test_cer_oracle_direction"]),
        "best_test_ar_assuming_1_minus_cer": 1.0 - float(best_row["test_cer_oracle_direction"]),
        "best_test_wer_oracle_direction": float(best_row["test_wer_oracle_direction"]),
        "best_test_blank_pred_ratio": float(best_row["test_blank_pred_ratio_unscaled"]),
        "ctc_summary_path": str(ctc_summary_path),
        "ctc_summary_cer_micro": float(ctc_summary["cer_micro"]),
        "ctc_summary_ar_assuming_1_minus_cer": 1.0 - float(ctc_summary["cer_micro"]),
        "ctc_summary_empty_pred_rate": float(ctc_summary["empty_pred_rate"]),
        "ctc_summary_pred_gt_len_ratio": float(ctc_summary["pred_gt_len_ratio"]),
        "len_bins": {
            "1": by_bin.get("1", {}),
            "2": by_bin.get("2", {}),
            "3-5": by_bin.get("3-5", {}),
            "6-10": by_bin.get("6-10", {}),
            "11+": by_bin.get("11+", {}),
        },
        "ar_formula_status": "unverified_vs_mthv2_papers",
    }
    return report


def render_markdown(report):
    lines = []
    lines.append("# MTH1000 Paper Metrics Report")
    lines.append("")
    lines.append(f"- run dir: `{report['run_dir']}`")
    lines.append(f"- best epoch from `log.txt`: `{report['best_epoch']}`")
    lines.append(
        f"- AR interpretation status: `{report['ar_formula_status']}`"
    )
    lines.append("")
    lines.append("## Main Table")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | ---: |")
    lines.append(
        f"| best test CER (oracle direction) | {report['best_test_cer_oracle_direction']:.6f} |"
    )
    lines.append(
        f"| best test AR assuming 1-CER | {report['best_test_ar_assuming_1_minus_cer']:.6f} |"
    )
    lines.append(
        f"| best test WER (oracle direction) | {report['best_test_wer_oracle_direction']:.6f} |"
    )
    lines.append(
        f"| best test blank pred ratio | {report['best_test_blank_pred_ratio']:.6f} |"
    )
    lines.append(f"| CTC summary CER micro | {report['ctc_summary_cer_micro']:.6f} |")
    lines.append(
        f"| CTC summary AR assuming 1-CER | {report['ctc_summary_ar_assuming_1_minus_cer']:.6f} |"
    )
    lines.append(
        f"| empty prediction rate | {report['ctc_summary_empty_pred_rate']:.6f} |"
    )
    lines.append(
        f"| pred/GT length ratio | {report['ctc_summary_pred_gt_len_ratio']:.6f} |"
    )
    lines.append("")
    lines.append("## Length Buckets")
    lines.append("")
    lines.append("| bin | samples | CER | empty rate | pred/GT len ratio |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")

    for key in ["1", "2", "3-5", "6-10", "11+"]:
        row = report["len_bins"].get(key, {})
        if not row:
            continue
        lines.append(
            f"| {key} | {int(row['samples'])} | {float(row['cer_micro']):.6f} | "
            f"{float(row['empty_pred_rate']):.6f} | {float(row['pred_gt_len_ratio']):.6f} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(
        "- `best test CER` comes from `log.txt` epoch selection; bucket breakdown comes from the supplied CTC summary JSON."
    )
    lines.append(
        "- `AR = 1 - CER` is only a temporary interpretation for paper planning and must be verified against the target paper's exact metric definition."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser("Build a fixed paper-facing metrics table for MTH1000 runs")
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--ctc_summary_json", required=True, type=str)
    parser.add_argument("--output_json", default="", type=str)
    parser.add_argument("--output_md", default="", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    log_rows = load_jsonl(run_dir / "log.txt")
    best_row = pick_best_epoch(log_rows)
    ctc_summary_path = Path(args.ctc_summary_json)
    ctc_summary = json.loads(ctc_summary_path.read_text(encoding="utf-8"))
    report = build_report(best_row, ctc_summary, run_dir, ctc_summary_path)

    output_json = Path(args.output_json) if args.output_json else run_dir / "paper_metrics_report.json"
    output_md = Path(args.output_md) if args.output_md else run_dir / "paper_metrics_report.md"
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved JSON -> {output_json}")
    print(f"Saved Markdown -> {output_md}")


if __name__ == "__main__":
    main()
