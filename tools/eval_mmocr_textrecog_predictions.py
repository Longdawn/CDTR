import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import mmengine

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from util.ctc_metrics import CtcTotals, LENGTH_BIN_KEYS, length_bin


def levenshtein_ops_any(a, b):
    a = list(a)
    b = list(b)
    n, m = len(a), len(b)
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
            if a[i - 1] == b[j - 1]:
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
    insertions = deletions = substitutions = 0
    while i > 0 or j > 0:
        step = op[i][j]
        if step == "eq":
            i -= 1
            j -= 1
        elif step == "sub":
            substitutions += 1
            i -= 1
            j -= 1
        elif step == "del":
            deletions += 1
            i -= 1
        elif step == "ins":
            insertions += 1
            j -= 1
        else:
            break

    return dp[n][m], insertions, deletions, substitutions


def parse_args():
    parser = argparse.ArgumentParser("Evaluate dumped MMOCR text recognition predictions")
    parser.add_argument("--predictions", required=True, help="MMOCR DumpResults pkl file")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_cases", required=True)
    return parser.parse_args()


def get_text_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        item = value.item
        if isinstance(item, str):
            return item
    if isinstance(value, dict):
        for key in ("item", "text"):
            if key in value:
                return get_text_value(value[key])
    return str(value)


def extract_texts(sample):
    gt = ""
    pred = ""

    if hasattr(sample, "gt_text"):
        gt = get_text_value(sample.gt_text)
    elif isinstance(sample, dict) and "gt_text" in sample:
        gt = get_text_value(sample["gt_text"])

    if hasattr(sample, "pred_text"):
        pred = get_text_value(sample.pred_text)
    elif isinstance(sample, dict) and "pred_text" in sample:
        pred = get_text_value(sample["pred_text"])

    return gt, pred


def main():
    args = parse_args()
    predictions = mmengine.load(args.predictions)

    totals = CtcTotals()
    by_len_bin = defaultdict(CtcTotals)
    by_len_exact = defaultdict(CtcTotals)
    cases = []

    for idx, sample in enumerate(predictions):
        gt, pred = extract_texts(sample)
        gt_chars = list(gt)
        pred_chars = list(pred)
        dist, ins, dels, subs = levenshtein_ops_any(gt_chars, pred_chars)
        gt_len = len(gt_chars)
        pred_len = len(pred_chars)
        totals.add(gt_len, pred_len, dist, ins, dels, subs)
        by_len_bin[length_bin(gt_len)].add(gt_len, pred_len, dist, ins, dels, subs)
        by_len_exact[str(gt_len)].add(gt_len, pred_len, dist, ins, dels, subs)
        cases.append({
            "idx": idx,
            "gt_len": gt_len,
            "pred_len": pred_len,
            "cer": dist / max(gt_len, 1),
            "ins": ins,
            "del": dels,
            "sub": subs,
            "gt": gt,
            "pred": pred,
        })

    summary = totals.to_summary()
    summary["micro_ar"] = 1.0 - summary["cer_micro"]
    summary["micro_cr"] = 1.0 - (totals.dels + totals.subs) / max(totals.gt_chars, 1)
    summary.update({
        "predictions": args.predictions,
        "samples": len(predictions),
        "by_gt_len_bin": {
            key: by_len_bin[key].to_summary()
            for key in LENGTH_BIN_KEYS
            if by_len_bin[key].n > 0
        },
        "by_gt_len_exact": {
            key: value.to_summary()
            for key, value in sorted(by_len_exact.items(), key=lambda item: int(item[0]))
        },
    })

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    out_cases = Path(args.output_cases)
    out_cases.parent.mkdir(parents=True, exist_ok=True)
    with out_cases.open("w", encoding="utf-8") as f:
        for row in sorted(cases, key=lambda item: item["cer"], reverse=True):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
