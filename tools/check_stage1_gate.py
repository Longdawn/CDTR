import argparse
import json
from pathlib import Path


def load_summary(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_gate(candidate, baseline, blank_max_delta, pred_min_ratio, pred_max_ratio, min_real_gt_box_rate):
    reasons = []

    cand_blank = float(candidate["avg_blank_ratio_full"])
    base_blank = float(baseline["avg_blank_ratio_full"])
    cand_pred = float(candidate["avg_pred_nonblank_count"])
    base_pred = float(baseline["avg_pred_nonblank_count"])
    cand_real_box = float(candidate.get("real_gt_box_rate", 0.0))

    if cand_blank > base_blank + blank_max_delta:
        reasons.append(
            f"blank_ratio too high: candidate={cand_blank:.4f}, baseline={base_blank:.4f}, limit={base_blank + blank_max_delta:.4f}"
        )

    if cand_pred < base_pred * pred_min_ratio:
        reasons.append(
            f"pred_nonblank_count too low: candidate={cand_pred:.2f}, baseline={base_pred:.2f}, limit={base_pred * pred_min_ratio:.2f}"
        )

    if cand_pred > base_pred * pred_max_ratio:
        reasons.append(
            f"pred_nonblank_count too high: candidate={cand_pred:.2f}, baseline={base_pred:.2f}, limit={base_pred * pred_max_ratio:.2f}"
        )

    if cand_real_box < min_real_gt_box_rate:
        reasons.append(
            f"real_gt_box_rate too low: candidate={cand_real_box:.4f}, required={min_real_gt_box_rate:.4f}"
        )

    return {
        "pass": len(reasons) == 0,
        "candidate": candidate,
        "baseline": baseline,
        "thresholds": {
            "blank_max_delta": blank_max_delta,
            "pred_min_ratio": pred_min_ratio,
            "pred_max_ratio": pred_max_ratio,
            "min_real_gt_box_rate": min_real_gt_box_rate,
        },
        "reasons": reasons,
    }


def render_markdown(result):
    lines = []
    lines.append("# Stage-1 Gate Check")
    lines.append("")
    lines.append(f"- pass: `{result['pass']}`")
    lines.append("")
    lines.append("| metric | candidate | baseline |")
    lines.append("| --- | ---: | ---: |")
    lines.append(
        f"| avg_blank_ratio_full | {float(result['candidate']['avg_blank_ratio_full']):.6f} | {float(result['baseline']['avg_blank_ratio_full']):.6f} |"
    )
    lines.append(
        f"| avg_pred_nonblank_count | {float(result['candidate']['avg_pred_nonblank_count']):.6f} | {float(result['baseline']['avg_pred_nonblank_count']):.6f} |"
    )
    lines.append(
        f"| avg_gt_len | {float(result['candidate']['avg_gt_len']):.6f} | {float(result['baseline']['avg_gt_len']):.6f} |"
    )
    lines.append(
        f"| real_gt_box_rate | {float(result['candidate'].get('real_gt_box_rate', 0.0)):.6f} | {float(result['baseline'].get('real_gt_box_rate', 0.0)):.6f} |"
    )
    lines.append("")
    lines.append("## Reasons")
    lines.append("")
    if result["reasons"]:
        for reason in result["reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- Candidate passed all thresholds.")
    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser("Check whether a stage-1 box diagnostic can proceed downstream")
    parser.add_argument("--candidate_summary", required=True, type=str)
    parser.add_argument("--baseline_summary", required=True, type=str)
    parser.add_argument("--blank_max_delta", default=0.01, type=float)
    parser.add_argument("--pred_min_ratio", default=0.8, type=float)
    parser.add_argument("--pred_max_ratio", default=1.5, type=float)
    parser.add_argument("--min_real_gt_box_rate", default=0.9, type=float)
    parser.add_argument("--output_json", default="", type=str)
    parser.add_argument("--output_md", default="", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    candidate = load_summary(args.candidate_summary)
    baseline = load_summary(args.baseline_summary)
    result = evaluate_gate(
        candidate,
        baseline,
        args.blank_max_delta,
        args.pred_min_ratio,
        args.pred_max_ratio,
        args.min_real_gt_box_rate,
    )

    candidate_dir = Path(args.candidate_summary).resolve().parent
    output_json = Path(args.output_json) if args.output_json else candidate_dir / "stage1_gate_check.json"
    output_md = Path(args.output_md) if args.output_md else candidate_dir / "stage1_gate_check.md"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(result), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved JSON -> {output_json}")
    print(f"Saved Markdown -> {output_md}")


if __name__ == "__main__":
    main()
