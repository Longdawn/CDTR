import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


TOPKS = (1, 2, 4)


def parse_args():
    parser = argparse.ArgumentParser("Diagnose fixed-bias short CTC errors using existing top-k diagnostic files")
    parser.add_argument("--fixed_bias_cases", type=str, required=True)
    parser.add_argument("--fixed_bias_summary", type=str, required=True)
    parser.add_argument("--short_diag_cases", type=str, required=True)
    parser.add_argument("--short_diag_summary", type=str, required=True)
    parser.add_argument("--output_summary", type=str, required=True)
    parser.add_argument("--output_cases", type=str, required=True)
    parser.add_argument("--output_report", type=str, required=True)
    parser.add_argument("--strict_align", action="store_true")
    return parser.parse_args()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def classify_error_type(case):
    if float(case.get("cer", 0.0)) <= 0.0:
        return "correct"

    gt_len = int(case["gt_len"])
    pred_len = int(case["pred_len"])
    subs = int(case.get("sub", 0))

    if gt_len == 1:
        if pred_len == 0:
            return "empty"
        if pred_len == 1:
            return "wrong_char"
        return "over_decode"

    if gt_len == 2:
        if pred_len == 0:
            return "empty"
        if pred_len == 1:
            return "short_output"
        if pred_len == 2 and subs == 1:
            return "one_correct_one_wrong"
        if pred_len == 2:
            return "both_wrong"
        return "over_decode"

    return "other"


def unique_top_labels(top_queries, topk):
    labels = []
    used = set()
    for query in top_queries:
        label = int(query["best_label"])
        if label in used:
            continue
        labels.append(label)
        used.add(label)
        if len(labels) >= int(topk):
            break
    return labels


def compute_topk_flags(diag_case, topks=TOPKS):
    gt_labels = [int(item["label"]) for item in diag_case.get("gt_best", [])]
    gt_set = set(gt_labels)
    top_queries = diag_case.get("top_queries", [])
    results = {}
    for topk in topks:
        labels = unique_top_labels(top_queries, topk)
        label_set = set(labels)
        missing = len(gt_set - label_set)
        results[f"top{topk}"] = {
            "labels": labels,
            "any_gt_in_topk": bool(gt_set & label_set),
            "both_gt_in_topk": missing == 0 and bool(gt_set),
            "missing_gt_count": missing,
        }
    return results


def align_cases(fixed_cases, diag_cases, strict=False):
    diag_by_idx = {int(row["idx"]): row for row in diag_cases}
    aligned = []
    unmatched = []
    for case in fixed_cases:
        idx = int(case["idx"])
        diag_case = diag_by_idx.get(idx)
        if diag_case is None:
            unmatched.append(idx)
            continue
        aligned.append((case, diag_case))
    if strict and unmatched:
        raise ValueError(f"unmatched idx entries: {unmatched[:10]}")
    return aligned, unmatched


def _char_overlap(pred_text, gt_text):
    gt_chars = list(gt_text)
    return any(ch in gt_chars for ch in pred_text)


def assign_rescue_bucket(case, error_type, topk_flags):
    gt_len = int(case["gt_len"])
    pred_text = case.get("pred", "")
    gt_text = case.get("gt", "")

    result = {
        "bucket": "not_decode_rescuable",
        "eligible_empty_top1_rescue": False,
        "eligible_empty_top2_rescue": False,
        "pred_char_hits_gt": False,
        "missing_gt_in_top1": False,
        "missing_gt_in_top2": False,
        "missing_gt_in_top4": False,
    }

    result["missing_gt_in_top1"] = topk_flags["top1"]["missing_gt_count"] == 0
    result["missing_gt_in_top2"] = topk_flags["top2"]["missing_gt_count"] == 0
    result["missing_gt_in_top4"] = topk_flags["top4"]["missing_gt_count"] == 0

    if gt_len == 1:
        if error_type == "empty":
            if topk_flags["top1"]["any_gt_in_topk"]:
                result["bucket"] = "fill_only_rescue"
                result["eligible_empty_top1_rescue"] = True
            elif topk_flags["top2"]["any_gt_in_topk"]:
                result["bucket"] = "fill_only_rescue"
                result["eligible_empty_top2_rescue"] = True
            elif topk_flags["top4"]["any_gt_in_topk"]:
                result["bucket"] = "reranking_candidate"
            return result

        if error_type == "wrong_char":
            if topk_flags["top4"]["any_gt_in_topk"]:
                result["bucket"] = "reranking_candidate"
            return result

        if error_type == "over_decode":
            if topk_flags["top4"]["any_gt_in_topk"]:
                result["bucket"] = "reranking_candidate"
            return result

        return result

    if gt_len == 2:
        if error_type == "empty":
            if topk_flags["top2"]["both_gt_in_topk"] or topk_flags["top4"]["both_gt_in_topk"]:
                result["bucket"] = "fill_only_rescue"
            elif topk_flags["top4"]["any_gt_in_topk"]:
                result["bucket"] = "reranking_candidate"
            return result

        if error_type == "short_output":
            result["pred_char_hits_gt"] = _char_overlap(pred_text, gt_text)
            if result["pred_char_hits_gt"] and (topk_flags["top2"]["both_gt_in_topk"] or topk_flags["top4"]["both_gt_in_topk"]):
                result["bucket"] = "append_rescue"
            elif topk_flags["top2"]["both_gt_in_topk"] or topk_flags["top4"]["both_gt_in_topk"]:
                result["bucket"] = "fill_only_rescue"
            elif topk_flags["top4"]["any_gt_in_topk"]:
                result["bucket"] = "reranking_candidate"
            return result

        if error_type in {"one_correct_one_wrong", "over_decode"}:
            if topk_flags["top4"]["any_gt_in_topk"]:
                result["bucket"] = "reranking_candidate"
            return result

        if error_type == "both_wrong":
            if topk_flags["top4"]["any_gt_in_topk"]:
                result["bucket"] = "reranking_candidate"
            return result

    return result


def annotate_case(case, diag_case):
    error_type = classify_error_type(case)
    topk_flags = compute_topk_flags(diag_case)
    rescue = assign_rescue_bucket(case, error_type, topk_flags)
    return {
        "idx": int(case["idx"]),
        "gt_len": int(case["gt_len"]),
        "pred_len": int(case["pred_len"]),
        "gt": case.get("gt", ""),
        "pred": case.get("pred", ""),
        "cer": float(case["cer"]),
        "ins": int(case.get("ins", 0)),
        "del": int(case.get("del", 0)),
        "sub": int(case.get("sub", 0)),
        "error_type": error_type,
        "gt_labels": [int(item["label"]) for item in diag_case.get("gt_best", [])],
        "topk": topk_flags,
        "top1_margin_blank_minus_nonblank": diag_case.get("top1_margin_blank_minus_nonblank"),
        "gt_best_mean_margin_blank_minus_gt": diag_case.get("gt_best_mean_margin_blank_minus_gt"),
        **rescue,
    }


def _new_stats():
    return {
        "count": 0,
        "top1_any": 0,
        "top2_any": 0,
        "top4_any": 0,
        "top1_both": 0,
        "top2_both": 0,
        "top4_both": 0,
        "missing_top1_total": 0,
        "missing_top2_total": 0,
        "missing_top4_total": 0,
        "bucket_counts": Counter(),
        "pred_char_hits_gt_count": 0,
        "append_rescue_count": 0,
        "fill_only_count": 0,
        "reranking_count": 0,
        "not_decode_count": 0,
    }


def _stats_to_summary(stats):
    count = stats["count"]
    if count == 0:
        return {
            "count": 0,
            "bucket_counts": {},
        }
    return {
        "count": count,
        "top1_any_gt_rate": stats["top1_any"] / count,
        "top2_any_gt_rate": stats["top2_any"] / count,
        "top4_any_gt_rate": stats["top4_any"] / count,
        "top1_both_gt_rate": stats["top1_both"] / count,
        "top2_both_gt_rate": stats["top2_both"] / count,
        "top4_both_gt_rate": stats["top4_both"] / count,
        "avg_missing_gt_top1": stats["missing_top1_total"] / count,
        "avg_missing_gt_top2": stats["missing_top2_total"] / count,
        "avg_missing_gt_top4": stats["missing_top4_total"] / count,
        "pred_char_hits_gt_rate": stats["pred_char_hits_gt_count"] / count,
        "append_rescue_rate": stats["append_rescue_count"] / count,
        "fill_only_rescue_rate": stats["fill_only_count"] / count,
        "reranking_candidate_rate": stats["reranking_count"] / count,
        "not_decode_rescuable_rate": stats["not_decode_count"] / count,
        "bucket_counts": dict(stats["bucket_counts"]),
    }


def summarize_annotations(annotations, fixed_bias_summary, short_diag_summary, unmatched_count):
    len1_types = ("empty", "wrong_char", "over_decode")
    len2_types = ("empty", "short_output", "one_correct_one_wrong", "both_wrong", "over_decode")

    type_counts = {"len1": Counter(), "len2": Counter()}
    type_stats = {"len1": {name: _new_stats() for name in len1_types}, "len2": {name: _new_stats() for name in len2_types}}
    bucket_counts = {"len1": Counter(), "len2": Counter()}

    len1_errors = 0
    len2_errors = 0
    for row in annotations:
        gt_len = row["gt_len"]
        top1 = row["topk"]["top1"]
        top2 = row["topk"]["top2"]
        top4 = row["topk"]["top4"]
        if gt_len == 1:
            len1_errors += 1
            group = "len1"
        elif gt_len == 2:
            len2_errors += 1
            group = "len2"
        else:
            continue

        error_type = row["error_type"]
        type_counts[group][error_type] += 1
        bucket_counts[group][row["bucket"]] += 1
        stats = type_stats[group][error_type]
        stats["count"] += 1
        stats["top1_any"] += int(top1["any_gt_in_topk"])
        stats["top2_any"] += int(top2["any_gt_in_topk"])
        stats["top4_any"] += int(top4["any_gt_in_topk"])
        stats["top1_both"] += int(top1["both_gt_in_topk"])
        stats["top2_both"] += int(top2["both_gt_in_topk"])
        stats["top4_both"] += int(top4["both_gt_in_topk"])
        stats["missing_top1_total"] += int(top1["missing_gt_count"])
        stats["missing_top2_total"] += int(top2["missing_gt_count"])
        stats["missing_top4_total"] += int(top4["missing_gt_count"])
        stats["pred_char_hits_gt_count"] += int(row["pred_char_hits_gt"])
        stats["bucket_counts"][row["bucket"]] += 1
        if row["bucket"] == "append_rescue":
            stats["append_rescue_count"] += 1
        elif row["bucket"] == "fill_only_rescue":
            stats["fill_only_count"] += 1
        elif row["bucket"] == "reranking_candidate":
            stats["reranking_count"] += 1
        else:
            stats["not_decode_count"] += 1

    len2_short_output = type_stats["len2"]["short_output"]
    short_output_count = len2_short_output["count"]
    append_upper_bound_top2 = len2_short_output["append_rescue_count"] / short_output_count if short_output_count else 0.0
    reranking_rate_len2 = (
        bucket_counts["len2"]["reranking_candidate"] / len2_errors if len2_errors else 0.0
    )
    not_decode_rate_len1 = (
        bucket_counts["len1"]["not_decode_rescuable"] / len1_errors if len1_errors else 0.0
    )
    recommendation = "continue_short_rescue" if append_upper_bound_top2 >= 0.15 else "prefer_training_side_ablation"
    if append_upper_bound_top2 < 0.15 and reranking_rate_len2 >= 0.35:
        recommendation = "add_topk_reranking_diagnostic_first"

    return {
        "inputs": {
            "fixed_bias_summary_samples": fixed_bias_summary.get("samples"),
            "short_diag_samples": short_diag_summary.get("all_short", {}).get("samples"),
        },
        "counts": {
            "len1_errors": len1_errors,
            "len2_errors": len2_errors,
            "unmatched_short_diag_cases": unmatched_count,
        },
        "len1": {
            "error_type_counts": dict(type_counts["len1"]),
            "rescue_bucket_counts": dict(bucket_counts["len1"]),
            "by_error_type": {
                name: _stats_to_summary(type_stats["len1"][name]) for name in len1_types
            },
        },
        "len2": {
            "error_type_counts": dict(type_counts["len2"]),
            "rescue_bucket_counts": dict(bucket_counts["len2"]),
            "by_error_type": {
                name: _stats_to_summary(type_stats["len2"][name]) for name in len2_types
            },
            "short_output_focus": {
                "count": short_output_count,
                "pred_char_hits_gt_rate": (
                    len2_short_output["pred_char_hits_gt_count"] / short_output_count if short_output_count else 0.0
                ),
                "both_gt_in_top1_rate": (
                    len2_short_output["top1_both"] / short_output_count if short_output_count else 0.0
                ),
                "both_gt_in_top2_rate": (
                    len2_short_output["top2_both"] / short_output_count if short_output_count else 0.0
                ),
                "both_gt_in_top4_rate": (
                    len2_short_output["top4_both"] / short_output_count if short_output_count else 0.0
                ),
                "append_rescue_upper_bound_top2": append_upper_bound_top2,
            },
        },
        "headline": {
            "fixed_bias_cer_micro": fixed_bias_summary.get("cer_micro"),
            "fixed_bias_len1_cer": fixed_bias_summary.get("by_gt_len_bin", {}).get("1", {}).get("cer_micro"),
            "fixed_bias_len2_cer": fixed_bias_summary.get("by_gt_len_bin", {}).get("2", {}).get("cer_micro"),
            "len2_short_output_append_rescue_upper_bound_top2": append_upper_bound_top2,
            "len2_reranking_candidate_rate": reranking_rate_len2,
            "len1_not_decode_rescuable_rate": not_decode_rate_len1,
            "recommendation": recommendation,
        },
    }


def build_markdown_report(summary):
    h = summary["headline"]
    c = summary["counts"]
    len1 = summary.get("len1", {})
    len2 = summary.get("len2", {})
    short_output = len2.get("short_output_focus", {})
    lines = [
        "# Short Top-k Diagnostic Report",
        "",
        "## Headline",
        "",
        f"- fixed-bias len=1 errors: `{c.get('len1_errors', 0)}`",
        f"- fixed-bias len=2 errors: `{c.get('len2_errors', 0)}`",
        f"- unmatched short diagnostic cases: `{c.get('unmatched_short_diag_cases', 0)}`",
        f"- len=2 short_output append upper bound (top-2): `{float(short_output.get('append_rescue_upper_bound_top2', h.get('len2_short_output_append_rescue_upper_bound_top2', 0.0))):.4f}`",
        f"- recommendation: `{h.get('recommendation', 'unknown')}`",
        "",
        "## len=1",
        "",
        f"- error types: `{json.dumps(len1.get('error_type_counts', {}), ensure_ascii=False)}`",
        f"- rescue buckets: `{json.dumps(len1.get('rescue_bucket_counts', {}), ensure_ascii=False)}`",
        "",
        "## len=2",
        "",
        f"- error types: `{json.dumps(len2.get('error_type_counts', {}), ensure_ascii=False)}`",
        f"- rescue buckets: `{json.dumps(len2.get('rescue_bucket_counts', {}), ensure_ascii=False)}`",
        "",
        "## len=2 short_output focus",
        "",
        f"- current one-char prediction already hits GT: `{float(short_output.get('pred_char_hits_gt_rate', 0.0)):.4f}`",
        f"- both GT in top-2 unique nonblank: `{float(short_output.get('both_gt_in_top2_rate', 0.0)):.4f}`",
        f"- append-style rescue theoretical upper bound: `{float(short_output.get('append_rescue_upper_bound_top2', h.get('len2_short_output_append_rescue_upper_bound_top2', 0.0))):.4f}`",
        "",
        "## Recommendation",
        "",
    ]
    if h.get("recommendation") == "continue_short_rescue":
        lines.append("- Continue decode-side short rescue, starting with append-style handling for len=2 short_output.")
    elif h.get("recommendation") == "add_topk_reranking_diagnostic_first":
        lines.append("- Add a narrower reranking-oriented diagnostic before any decode-side intervention.")
    else:
        lines.append("- Decode-side upside looks limited; prefer training-side ablation next.")
    lines.append("")
    return "\n".join(lines)


def main():
    cli = parse_args()
    fixed_cases = load_jsonl(cli.fixed_bias_cases)
    fixed_summary = load_json(cli.fixed_bias_summary)
    short_diag_cases = load_jsonl(cli.short_diag_cases)
    short_diag_summary = load_json(cli.short_diag_summary)

    target_fixed_cases = [
        case
        for case in fixed_cases
        if int(case["gt_len"]) in (1, 2) and float(case.get("cer", 0.0)) > 0.0
    ]

    aligned, unmatched = align_cases(target_fixed_cases, short_diag_cases, strict=cli.strict_align)
    annotations = []
    for case, diag_case in aligned:
        annotations.append(annotate_case(case, diag_case))

    summary = summarize_annotations(
        annotations=annotations,
        fixed_bias_summary=fixed_summary,
        short_diag_summary=short_diag_summary,
        unmatched_count=len(unmatched),
    )
    report = build_markdown_report(summary)

    out_summary = Path(cli.output_summary)
    out_cases = Path(cli.output_cases)
    out_report = Path(cli.output_report)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_cases.open("w", encoding="utf-8") as f:
        for row in annotations:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_report.write_text(report, encoding="utf-8")

    print("Saved summary ->", out_summary)
    print("Saved cases ->", out_cases)
    print("Saved report ->", out_report)
    print("Recommendation:", summary["headline"]["recommendation"])
    print("len2 short_output append upper bound (top2):", f"{summary['headline']['len2_short_output_append_rescue_upper_bound_top2']:.4f}")


if __name__ == "__main__":
    main()
