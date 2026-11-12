import argparse
import glob
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Merge chunked CTC micro CER/AR/CR JSON summaries")
    parser.add_argument("--inputs", nargs="+", required=True, help="JSON files or glob patterns")
    parser.add_argument("--output_json", required=True)
    return parser.parse_args()


def expand_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            unique.append(Path(path))
            seen.add(path)
    return unique


def main():
    cli = parse_args()
    paths = expand_inputs(cli.inputs)
    if not paths:
        raise ValueError("no input chunk files")

    chunks = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            chunks.append(json.load(f))

    samples = sum(int(chunk["evaluated_samples"]) for chunk in chunks)
    total_chars = sum(int(chunk["total_chars"]) for chunk in chunks)
    insertions = sum(int(chunk["insertions"]) for chunk in chunks)
    deletions = sum(int(chunk["deletions"]) for chunk in chunks)
    substitutions = sum(int(chunk["substitutions"]) for chunk in chunks)
    pred_chars = sum(round(float(chunk["avg_pred_len"]) * int(chunk["evaluated_samples"])) for chunk in chunks)
    empty_pred = sum(round(float(chunk["empty_pred_rate"]) * int(chunk["evaluated_samples"])) for chunk in chunks)
    dist = insertions + deletions + substitutions

    first = chunks[0]
    summary = {
        "dataset": first.get("dataset"),
        "split": first.get("split"),
        "checkpoint": first.get("checkpoint"),
        "config": first.get("config"),
        "dataset_size": first.get("dataset_size"),
        "evaluated_samples": samples,
        "chunks": [
            {
                "path": str(path),
                "start_index": chunk.get("start_index"),
                "end_index": chunk.get("end_index"),
                "evaluated_samples": chunk.get("evaluated_samples"),
            }
            for path, chunk in zip(paths, chunks)
        ],
        "total_chars": total_chars,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "cer_micro": dist / max(total_chars, 1),
        "micro_cer": dist / max(total_chars, 1),
        "micro_ar": 1.0 - dist / max(total_chars, 1),
        "micro_cr": 1.0 - (deletions + substitutions) / max(total_chars, 1),
        "avg_gt_len": total_chars / max(samples, 1),
        "avg_pred_len": pred_chars / max(samples, 1),
        "pred_gt_len_ratio": pred_chars / max(total_chars, 1),
        "empty_pred_rate": empty_pred / max(samples, 1),
        "ins_rate": insertions / max(total_chars, 1),
        "del_rate": deletions / max(total_chars, 1),
        "sub_rate": substitutions / max(total_chars, 1),
    }

    out_path = Path(cli.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved merged summary -> {out_path}")


if __name__ == "__main__":
    main()
