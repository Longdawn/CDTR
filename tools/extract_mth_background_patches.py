import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract mostly-blank paper/background patches from TKHMTH images."
    )
    parser.add_argument(
        "--image-roots",
        nargs="+",
        default=[
            "data/TKHMTH2200/MTH1000/img",
            "data/TKHMTH2200/MTH1200/img",
            "data/TKHMTH2200/TKH/img",
        ],
    )
    parser.add_argument("--output-dir", default="data/synth_bg_patches")
    parser.add_argument("--num-patches", type=int, default=2000)
    parser.add_argument("--patch-width", type=int, default=192)
    parser.add_argument("--patch-height", type=int, default=768)
    parser.add_argument("--max-dark-ratio", type=float, default=0.015)
    parser.add_argument(
        "--ink-threshold",
        type=int,
        default=210,
        help="Pixels darker than this are treated as possible faint ink/text.",
    )
    parser.add_argument(
        "--max-ink-ratio",
        type=float,
        default=0.035,
        help="Reject patches with too much faint ink/text. Lower is stricter.",
    )
    parser.add_argument(
        "--max-edge-ratio",
        type=float,
        default=0.030,
        help="Reject patches with too many Canny edges, which usually means text remains.",
    )
    parser.add_argument(
        "--max-band-ink-ratio",
        type=float,
        default=0.080,
        help="Reject patches when any horizontal/vertical band has concentrated faint ink.",
    )
    parser.add_argument("--band-count", type=int, default=16)
    parser.add_argument("--min-std", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tries-per-patch", type=int, default=200)
    return parser.parse_args()


def collect_images(roots):
    paths = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for suffix in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            paths.extend(root_path.glob(suffix))
    return sorted(paths)


def max_band_ink_ratio(gray, ink_threshold, band_count):
    ratios = []
    height, width = gray.shape
    for i in range(band_count):
        h_band = gray[i * height // band_count : (i + 1) * height // band_count, :]
        v_band = gray[:, i * width // band_count : (i + 1) * width // band_count]
        ratios.append(float((h_band < ink_threshold).mean()))
        ratios.append(float((v_band < ink_threshold).mean()))
    return max(ratios) if ratios else 0.0


def is_good_patch(
    patch,
    max_dark_ratio,
    ink_threshold,
    max_ink_ratio,
    max_edge_ratio,
    max_band_ink,
    band_count,
    min_std,
):
    gray = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2GRAY)
    dark_ratio = float((gray < 145).mean())
    ink_ratio = float((gray < ink_threshold).mean())
    band_ink_ratio = max_band_ink_ratio(gray, ink_threshold, band_count)
    edges = cv2.Canny(gray, 40, 120)
    edge_ratio = float((edges > 0).mean())
    std = float(gray.std())
    mean = float(gray.mean())
    return (
        dark_ratio <= max_dark_ratio
        and ink_ratio <= max_ink_ratio
        and band_ink_ratio <= max_band_ink
        and edge_ratio <= max_edge_ratio
        and std >= min_std
        and mean >= 150
    )


def random_crop_or_resize(image, width, height):
    if image.width < width or image.height < height:
        scale = max(width / image.width, height / image.height)
        new_size = (int(image.width * scale + 0.5), int(image.height * scale + 0.5))
        image = image.resize(new_size, Image.BICUBIC)
    x = random.randint(0, image.width - width)
    y = random.randint(0, image.height - height)
    return image.crop((x, y, x + width, y + height))


def main():
    args = parse_args()
    random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(args.image_roots)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {args.image_roots}")

    saved = 0
    attempts = 0
    while saved < args.num_patches:
        attempts += 1
        if attempts > args.num_patches * args.max_tries_per_patch:
            raise RuntimeError(
                f"Only saved {saved}/{args.num_patches}; relax thresholds or add images."
            )

        image_path = random.choice(image_paths)
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        patch = random_crop_or_resize(image, args.patch_width, args.patch_height)
        if not is_good_patch(
            patch,
            args.max_dark_ratio,
            args.ink_threshold,
            args.max_ink_ratio,
            args.max_edge_ratio,
            args.max_band_ink_ratio,
            args.band_count,
            args.min_std,
        ):
            continue

        patch.save(out_dir / f"bg_{saved:06d}.jpg", quality=95)
        saved += 1
        if saved % 100 == 0:
            print(f"saved {saved}/{args.num_patches}", flush=True)

    print(
        {
            "output_dir": str(out_dir),
            "num_patches": saved,
            "attempts": attempts,
            "image_roots": args.image_roots,
        }
    )


if __name__ == "__main__":
    main()
