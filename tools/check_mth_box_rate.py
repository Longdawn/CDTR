import argparse
import json
from types import SimpleNamespace

from datasets.MTH1000 import build_mth1000


def parse_args():
    parser = argparse.ArgumentParser(description="Check real char-box hit rate for MTH-style datasets")
    parser.add_argument("--root", required=True, help="Processed dataset root name under datasets_path")
    parser.add_argument("--raw-root", required=True, help="Raw dataset root name under datasets_path")
    parser.add_argument("--split", default="train", choices=["train", "valid", "val", "test"])
    parser.add_argument("--image-ext", default="jpg")
    parser.add_argument("--filter-direction", default="vertical", choices=["vertical", "horizontal", "all"])
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full split")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N samples; 0 disables")
    return parser.parse_args()


def main():
    cli = parse_args()
    args = SimpleNamespace(
        mth1000_root=cli.root,
        mth1000_labels_file="labels.pkl",
        mth1000_images_subdir="lines",
        mth1000_image_ext=cli.image_ext,
        mth1000_use_char_boxes=True,
        mth1000_raw_root=cli.raw_root,
        direction_source="label",
        forced_direction="vertical" if cli.filter_direction == "vertical" else "auto",
        mth1000_filter_direction=cli.filter_direction,
        data_aug_scales=[800],
        data_aug_max_size=1800,
        mode_chr=True,
    )

    dataset = build_mth1000(cli.split, args)
    limit = len(dataset) if cli.max_samples <= 0 else min(len(dataset), cli.max_samples)
    for i in range(limit):
        try:
            _ = dataset[i]
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "error_at_index": i,
                        "sample_id": dataset.samples[i].get("id"),
                        "error": repr(exc),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            raise
        if cli.progress_every > 0 and (i + 1) % cli.progress_every == 0:
            print(f"processed {i + 1}/{limit}", flush=True)

    stats = dict(dataset._char_box_stats)
    used_real = stats.get("used_real_boxes", 0)
    used_dummy = stats.get("used_dummy_boxes", 0)
    total = used_real + used_dummy
    report = {
        "root": cli.root,
        "raw_root": cli.raw_root,
        "split": cli.split,
        "filter_direction": cli.filter_direction,
        "num_samples_checked": limit,
        "used_real_boxes": used_real,
        "used_dummy_boxes": used_dummy,
        "real_box_rate": (used_real / total) if total else None,
        "stats": stats,
        "fallback_examples": dataset._char_box_fallback_examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
