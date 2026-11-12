import argparse
import csv
import json
import sys
from pathlib import Path


LENGTH_BUCKETS = ("1", "2", "3-5", "6-10", "11+")
CSV_FIELDS = (
    "dataset",
    "split",
    "decode",
    "blank_bias",
    "nonblank_bias",
    "samples",
    "cer_micro",
    "ar_micro",
    "cr_micro",
    "empty_pred_rate",
    "pred_gt_len_ratio",
    "len1_cer",
    "len2_cer",
    "len3_5_cer",
    "len6_10_cer",
    "len11p_cer",
    "checkpoint",
    "result_file",
)


def parse_args():
    parser = argparse.ArgumentParser(
        "Format a DTLR CTC micro-metrics JSON as paper-table rows."
    )
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--decode", required=True, choices=["clean", "bias"])
    parser.add_argument("--blank_bias", type=float, default=0.0)
    parser.add_argument("--nonblank_bias", type=float, default=0.0)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--result_file", default=None)
    parser.add_argument(
        "--table",
        choices=["main-md", "selection-md", "csv"],
        default="main-md",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="When input_json is a sweep list, select this zero-based rank.",
    )
    parser.add_argument("--notes", default="")
    parser.add_argument("--header", action="store_true")
    return parser.parse_args()


def load_summary(path, rank):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        if not data:
            raise ValueError(f"{path} is an empty sweep result list")
        if rank < 0 or rank >= len(data):
            raise ValueError(f"rank {rank} out of range for {len(data)} rows")
        return data[rank]
    return data


def ratio(row, key, default=None):
    value = row.get(key, default)
    return "" if value is None else value


def micro_ar(row):
    if "micro_ar" in row:
        return row["micro_ar"]
    if "cer_micro" in row:
        return 1.0 - row["cer_micro"]
    return 1.0 - row["micro_cer"]


def micro_cr(row):
    if "micro_cr" in row:
        return row["micro_cr"]
    return 1.0 - row["del_rate"] - row["sub_rate"]


def bucket_cer(row, bucket):
    buckets = row.get("by_gt_len_bin", {})
    if bucket not in buckets:
        return ""
    return buckets[bucket].get("cer_micro", "")


def pct(value):
    if value == "":
        return "-"
    return f"{100.0 * float(value):.2f}"


def dec(value):
    if value == "":
        return "-"
    return f"{float(value):.3f}"


def bias_label(blank_bias, nonblank_bias):
    return f"{format_bias(blank_bias)}/{format_bias(nonblank_bias)}"


def format_bias(value):
    value = float(value)
    if abs(value) < 1e-12:
        return "0"
    if value.is_integer():
        return f"{value:.1f}"
    return f"{value:g}"


def as_csv_row(args, row):
    return {
        "dataset": args.dataset,
        "split": args.split,
        "decode": args.decode,
        "blank_bias": format_bias(args.blank_bias),
        "nonblank_bias": format_bias(args.nonblank_bias),
        "samples": int(row["samples"]),
        "cer_micro": ratio(row, "micro_cer", row.get("cer_micro")),
        "ar_micro": micro_ar(row),
        "cr_micro": micro_cr(row),
        "empty_pred_rate": ratio(row, "empty_pred_rate"),
        "pred_gt_len_ratio": ratio(row, "pred_gt_len_ratio"),
        "len1_cer": bucket_cer(row, "1"),
        "len2_cer": bucket_cer(row, "2"),
        "len3_5_cer": bucket_cer(row, "3-5"),
        "len6_10_cer": bucket_cer(row, "6-10"),
        "len11p_cer": bucket_cer(row, "11+"),
        "checkpoint": args.checkpoint,
        "result_file": args.result_file or args.input_json,
    }


def print_csv(args, row):
    writer = csv.DictWriter(sys.stdout, fieldnames=CSV_FIELDS)
    if args.header:
        writer.writeheader()
    writer.writerow(as_csv_row(args, row))


def print_main_md(args, row):
    fields = [
        args.dataset,
        args.split,
        args.decode,
        bias_label(args.blank_bias, args.nonblank_bias),
        str(int(row["samples"])),
        pct(ratio(row, "micro_cer", row.get("cer_micro"))),
        pct(micro_ar(row)),
        pct(micro_cr(row)),
        pct(ratio(row, "empty_pred_rate")),
        dec(ratio(row, "pred_gt_len_ratio")),
        pct(bucket_cer(row, "1")),
        pct(bucket_cer(row, "2")),
        pct(bucket_cer(row, "3-5")),
        pct(bucket_cer(row, "6-10")),
        pct(bucket_cer(row, "11+")),
        f"`{args.checkpoint}`",
        f"`{args.result_file or args.input_json}`",
    ]
    print("| " + " | ".join(fields) + " |")


def print_selection_md(args, row):
    fields = [
        args.dataset,
        args.split,
        args.decode,
        bias_label(args.blank_bias, args.nonblank_bias),
        str(int(row["samples"])),
        pct(ratio(row, "micro_cer", row.get("cer_micro"))),
        pct(micro_ar(row)),
        pct(micro_cr(row)),
        args.notes,
        f"`{args.result_file or args.input_json}`",
    ]
    print("| " + " | ".join(fields) + " |")


def main():
    args = parse_args()
    input_path = Path(args.input_json)
    row = load_summary(input_path, args.rank)
    if args.table == "csv":
        print_csv(args, row)
    elif args.table == "selection-md":
        print_selection_md(args, row)
    else:
        print_main_md(args, row)


if __name__ == "__main__":
    main()
