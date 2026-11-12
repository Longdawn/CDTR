#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


DEFAULT_RUNS = [
    "logs/mth1200pre_mth1000_step1_0511-1441",
    "logs/mth1200pre_mth1000ft_full_0511-1717",
    "logs/mth1200pre_mth1000ft_full_lr1e6_0511-1950",
]

METRIC_KEYS = {
    "cer": ("test_cer_oracle_direction", "test_cer"),
    "wer": ("test_wer_oracle_direction", "test_wer"),
    "blank": ("test_blank_pred_ratio_unscaled",),
    "test_loss": ("test_loss", "test_loss_CTC"),
    "train_loss": ("train_loss", "train_loss_CTC"),
    "lr": ("train_lr",),
}


def pick(row, names, default=None):
    for name in names:
        if name in row:
            return row[name]
    return default


def load_rows(run_dir):
    log_path = run_dir / "log.txt"
    if not log_path.exists():
        return []

    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def best_row(rows):
    candidates = [(pick(row, METRIC_KEYS["cer"]), row) for row in rows]
    candidates = [(cer, row) for cer, row in candidates if cer is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def trend(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return "n/a"
    first, last = values[0], values[-1]
    best = min(values)
    if best == last and last < first:
        return "improving"
    if best < last and last > best * 1.01:
        return "regressed_after_best"
    if last < first:
        return "slightly_improving"
    return "flat_or_worse"


def fmt(value, digits=6):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def summarize_run(run_dir):
    rows = load_rows(run_dir)
    if not rows:
        return {
            "run": str(run_dir),
            "status": "missing_or_empty_log",
            "rows": [],
        }

    best = best_row(rows)
    last = rows[-1]
    cers = [pick(row, METRIC_KEYS["cer"]) for row in rows]
    test_losses = [pick(row, METRIC_KEYS["test_loss"]) for row in rows]
    train_losses = [pick(row, METRIC_KEYS["train_loss"]) for row in rows]
    blanks = [pick(row, METRIC_KEYS["blank"]) for row in rows]

    return {
        "run": str(run_dir),
        "status": "ok",
        "rows": rows,
        "epochs": len(rows),
        "best_epoch": best.get("epoch") if best else None,
        "best_cer": pick(best, METRIC_KEYS["cer"]) if best else None,
        "best_wer": pick(best, METRIC_KEYS["wer"]) if best else None,
        "best_test_loss": pick(best, METRIC_KEYS["test_loss"]) if best else None,
        "best_blank": pick(best, METRIC_KEYS["blank"]) if best else None,
        "last_epoch": last.get("epoch"),
        "last_cer": pick(last, METRIC_KEYS["cer"]),
        "last_wer": pick(last, METRIC_KEYS["wer"]),
        "last_test_loss": pick(last, METRIC_KEYS["test_loss"]),
        "last_train_loss": pick(last, METRIC_KEYS["train_loss"]),
        "last_blank": pick(last, METRIC_KEYS["blank"]),
        "lr": pick(last, METRIC_KEYS["lr"]),
        "cer_trend": trend(cers),
        "test_loss_trend": trend(test_losses),
        "train_loss_trend": trend(train_losses),
        "blank_min": min([v for v in blanks if v is not None], default=None),
        "blank_max": max([v for v in blanks if v is not None], default=None),
    }


def infer_causes(summaries):
    ok = [s for s in summaries if s["status"] == "ok"]
    if not ok:
        return ["No valid run logs were found."]

    best = min(ok, key=lambda s: s["best_cer"] if s["best_cer"] is not None else float("inf"))
    notes = [
        f"Best observed checkpoint is from `{best['run']}` at epoch {best['best_epoch']} with CER {fmt(best['best_cer'])}.",
    ]

    head_runs = [s for s in ok if "step1" in s["run"] or "head" in s["run"]]
    full_runs = [s for s in ok if "full" in s["run"] and "lr1e6" not in s["run"]]
    low_lr_runs = [s for s in ok if "lr1e6" in s["run"] or "lr1e-6" in s["run"]]

    if head_runs and full_runs:
        head_best = min(head_runs, key=lambda s: s["best_cer"])
        full_best = min(full_runs, key=lambda s: s["best_cer"])
        gain = head_best["best_cer"] - full_best["best_cer"]
        notes.append(
            "End-to-end full finetuning is useful: "
            f"head-only best CER {fmt(head_best['best_cer'])} -> full best CER {fmt(full_best['best_cer'])} "
            f"(absolute gain {fmt(gain)})."
        )

    if low_lr_runs and full_runs:
        full_best = min(full_runs, key=lambda s: s["best_cer"])
        low_best = min(low_lr_runs, key=lambda s: s["best_cer"])
        if low_best["best_cer"] >= full_best["best_cer"]:
            notes.append(
                "The lr=1e-6 continuation did not improve the best checkpoint: "
                f"{fmt(low_best['best_cer'])} vs previous full best {fmt(full_best['best_cer'])}."
            )

    for s in ok:
        if s["cer_trend"] == "regressed_after_best" and s["test_loss_trend"] in {"improving", "slightly_improving"}:
            notes.append(
                f"`{s['run']}` shows CTC-loss/CER mismatch: test loss keeps improving while CER regresses after epoch {s['best_epoch']}."
            )
            break

    blank_ranges = [(s["blank_min"], s["blank_max"]) for s in ok if s["blank_min"] is not None and s["blank_max"] is not None]
    if blank_ranges:
        global_min = min(v[0] for v in blank_ranges)
        global_max = max(v[1] for v in blank_ranges)
        if global_min > 0.98 and (global_max - global_min) < 0.005:
            notes.append(
                "Blank ratio stays very high and almost unchanged "
                f"({fmt(global_min, 4)}-{fmt(global_max, 4)}), so lr-only continuation is not changing the blank/length behavior."
            )

    notes.append(
        "Most likely bottleneck is no longer raw CTC convergence. The next useful checks are box quality, predicted length/nonblank count, character-confusion distribution, and whether stage-1 detection pretraining produces per-character boxes."
    )
    return notes


def write_report(summaries):
    lines = []
    lines.append("# MTH Finetuning Run Analysis")
    lines.append("")
    lines.append("| run | status | epochs | lr | best epoch | best CER | best WER | best test loss | last CER | last test loss | blank range | CER trend |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for s in summaries:
        blank_range = "-"
        if s.get("blank_min") is not None and s.get("blank_max") is not None:
            blank_range = f"{fmt(s['blank_min'], 4)}-{fmt(s['blank_max'], 4)}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{s['run']}`",
                    s.get("status", "-"),
                    fmt(s.get("epochs")),
                    fmt(s.get("lr")),
                    fmt(s.get("best_epoch")),
                    fmt(s.get("best_cer")),
                    fmt(s.get("best_wer")),
                    fmt(s.get("best_test_loss")),
                    fmt(s.get("last_cer")),
                    fmt(s.get("last_test_loss")),
                    blank_range,
                    s.get("cer_trend", "-"),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Diagnosis")
    for note in infer_causes(summaries):
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Recommended Next Steps")
    lines.append("- Use the best checkpoint for reporting/evaluation, not the last checkpoint.")
    lines.append("- Stop lr-only continuation if it does not beat the previous best CER after 2-3 validation points.")
    lines.append("- Run box visualization and predicted-length analysis on the best checkpoint before more finetuning.")
    lines.append("- If boxes are still long vertical columns or poorly localized, improve stage-1 detection pretraining before more CTC tuning.")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize MTH DTLR finetuning logs and diagnose likely bottlenecks.")
    parser.add_argument("runs", nargs="*", default=DEFAULT_RUNS, help="Run directories containing log.txt")
    parser.add_argument("--output", "-o", default="", help="Optional markdown report path")
    return parser.parse_args()


def main():
    args = parse_args()
    summaries = [summarize_run(Path(run)) for run in args.runs]
    report = write_report(summaries)
    print(report)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
