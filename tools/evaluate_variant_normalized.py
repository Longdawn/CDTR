import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Evaluate OCR cases after variant-character normalization")
    parser.add_argument("--cases", required=True, help="JSONL cases from tools/analyze_ctc_errors.py")
    parser.add_argument(
        "--variant_map",
        default="data/variant_normalization_conservative.json",
        help="JSON mapping from variant char to canonical char.",
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_fixed_cases", default="")
    return parser.parse_args()


def read_cases(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_variant_map(path):
    with Path(path).open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {str(k): str(v) for k, v in mapping.items()}


def normalize_text(text, mapping):
    return "".join(mapping.get(ch, ch) for ch in text)


def edit_ops(gt, pred):
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
    ins = dels = subs = 0
    sub_pairs = []
    while i > 0 or j > 0:
        step = op[i][j]
        if step == "eq":
            i -= 1
            j -= 1
        elif step == "sub":
            subs += 1
            sub_pairs.append((gt[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif step == "del":
            dels += 1
            i -= 1
        elif step == "ins":
            ins += 1
            j -= 1
        else:
            break
    return dp[n][m], ins, dels, subs, list(reversed(sub_pairs))


def length_bin(gt_len):
    if gt_len == 1:
        return "1"
    if gt_len == 2:
        return "2"
    if gt_len <= 5:
        return "3-5"
    if gt_len <= 10:
        return "6-10"
    return "11+"


class Totals:
    def __init__(self):
        self.n = 0
        self.gt_chars = 0
        self.pred_chars = 0
        self.dist = 0
        self.ins = 0
        self.dels = 0
        self.subs = 0
        self.exact = 0
        self.ar_sum = 0.0
        self.cr_sum = 0.0

    def add(self, gt, pred):
        dist, ins, dels, subs, sub_pairs = edit_ops(gt, pred)
        gt_len = max(len(gt), 1)
        self.n += 1
        self.gt_chars += gt_len
        self.pred_chars += len(pred)
        self.dist += dist
        self.ins += ins
        self.dels += dels
        self.subs += subs
        self.exact += int(gt == pred)
        self.ar_sum += 1.0 - dist / gt_len
        self.cr_sum += (len(gt) - (dels + subs)) / gt_len
        return dist, ins, dels, subs, sub_pairs

    def summary(self):
        return {
            "samples": self.n,
            "cer_micro": self.dist / max(self.gt_chars, 1),
            "ar_micro": 1.0 - self.dist / max(self.gt_chars, 1),
            "ar_macro": self.ar_sum / max(self.n, 1),
            "cr_macro": self.cr_sum / max(self.n, 1),
            "exact_match": self.exact / max(self.n, 1),
            "avg_gt_len": self.gt_chars / max(self.n, 1),
            "avg_pred_len": self.pred_chars / max(self.n, 1),
            "ins_rate": self.ins / max(self.gt_chars, 1),
            "del_rate": self.dels / max(self.gt_chars, 1),
            "sub_rate": self.subs / max(self.gt_chars, 1),
        }


def main():
    args = parse_args()
    rows = read_cases(args.cases)
    variant_map = load_variant_map(args.variant_map)

    strict = Totals()
    normalized = Totals()
    by_len = defaultdict(Totals)
    fixed_cases = []
    fixed_sub_pairs = Counter()

    for row in rows:
        gt = row["gt"]
        pred = row["pred"]
        strict.add(gt, pred)

        norm_gt = normalize_text(gt, variant_map)
        norm_pred = normalize_text(pred, variant_map)
        norm_dist, norm_ins, norm_del, norm_sub, _ = normalized.add(norm_gt, norm_pred)
        by_len[length_bin(len(gt))].add(norm_gt, norm_pred)

        strict_dist, _, _, _, strict_sub_pairs = edit_ops(gt, pred)
        if strict_dist > 0 and norm_dist < strict_dist:
            for gt_ch, pred_ch in strict_sub_pairs:
                if gt_ch != pred_ch and variant_map.get(gt_ch, gt_ch) == variant_map.get(pred_ch, pred_ch):
                    fixed_sub_pairs[(gt_ch, pred_ch)] += 1
            fixed_cases.append(
                {
                    "idx": row["idx"],
                    "gt_len": row["gt_len"],
                    "strict_cer": strict_dist / max(len(gt), 1),
                    "normalized_cer": norm_dist / max(len(norm_gt), 1),
                    "gt": gt,
                    "pred": pred,
                    "normalized_gt": norm_gt,
                    "normalized_pred": norm_pred,
                }
            )

    strict_summary = strict.summary()
    norm_summary = normalized.summary()
    output = {
        "cases": len(rows),
        "variant_map": args.variant_map,
        "strict": strict_summary,
        "normalized": norm_summary,
        "delta": {
            "cer_micro": norm_summary["cer_micro"] - strict_summary["cer_micro"],
            "ar_micro": norm_summary["ar_micro"] - strict_summary["ar_micro"],
            "ar_macro": norm_summary["ar_macro"] - strict_summary["ar_macro"],
            "cr_macro": norm_summary["cr_macro"] - strict_summary["cr_macro"],
            "exact_match": norm_summary["exact_match"] - strict_summary["exact_match"],
        },
        "by_gt_len_bin_normalized": {
            key: by_len[key].summary()
            for key in ["1", "2", "3-5", "6-10", "11+"]
            if by_len[key].n > 0
        },
        "fixed_case_count": len(fixed_cases),
        "top_fixed_substitutions": [
            {"gt": gt, "pred": pred, "count": count}
            for (gt, pred), count in fixed_sub_pairs.most_common(50)
        ],
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.output_fixed_cases:
        out_cases = Path(args.output_fixed_cases)
        out_cases.parent.mkdir(parents=True, exist_ok=True)
        with out_cases.open("w", encoding="utf-8") as f:
            for row in sorted(fixed_cases, key=lambda item: item["strict_cer"], reverse=True):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Saved normalized evaluation -> {out_json}")


if __name__ == "__main__":
    main()
