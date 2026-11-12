import unittest

import torch

from util.ctc_decoding import (
    apply_ctc_calibration,
    collapse_ctc,
    decode_greedy,
    ratio_from_target,
)
from util.ctc_metrics import CtcTotals, length_bin, levenshtein_ops, summarize_by_length


class CtcDecodingTest(unittest.TestCase):
    def test_collapse_ctc_removes_repeats_and_blanks(self):
        self.assertEqual(collapse_ctc([0, 2, 2, 0, 3, 3, 0, 2]), [2, 3, 2])

    def test_decode_greedy_maps_tokens_to_label_ids(self):
        pred_probs = torch.tensor(
            [
                [
                    [0.90, 0.05, 0.05, 0.00],
                    [0.10, 0.80, 0.05, 0.05],
                    [0.20, 0.70, 0.05, 0.05],
                    [0.10, 0.05, 0.80, 0.05],
                    [0.80, 0.10, 0.05, 0.05],
                    [0.10, 0.05, 0.05, 0.80],
                ]
            ],
            dtype=torch.float32,
        )
        self.assertEqual(decode_greedy(pred_probs, charset_size=3), [0, 1, 2])

    def test_decode_greedy_rejects_multi_sample_batch(self):
        pred_probs = torch.zeros((2, 3, 4), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "batch size 1|single sample"):
            decode_greedy(pred_probs, charset_size=3)

    def test_ratio_from_target_accepts_tensor_and_missing_orig_size(self):
        self.assertAlmostEqual(
            ratio_from_target({"orig_size": torch.tensor([120.0, 40.0])}),
            3.0,
        )
        self.assertAlmostEqual(ratio_from_target({}), 1.0)

    def test_apply_ctc_calibration_adds_ratio_bias_only_inside_range(self):
        pred_probs = torch.tensor([[[0.51, 0.49], [0.60, 0.40]]], dtype=torch.float32)
        outside = apply_ctc_calibration(
            pred_probs,
            target={"orig_size": torch.tensor([300.0, 100.0])},
            ratio_nonblank_bias=1.0,
            ratio_min=1.5,
            ratio_max=2.0,
        )
        inside = apply_ctc_calibration(
            pred_probs,
            target={"orig_size": torch.tensor([180.0, 100.0])},
            ratio_nonblank_bias=1.0,
            ratio_min=1.5,
            ratio_max=2.0,
        )
        self.assertEqual(outside.argmax(-1)[0].tolist(), [0, 0])
        self.assertEqual(inside.argmax(-1)[0].tolist(), [1, 1])

    def test_apply_ctc_calibration_applies_bias_only_inside_margin_gate(self):
        pred_probs = torch.tensor(
            [
                [
                    [0.60, 0.40],  # margin = 0.20, inside gate
                    [0.90, 0.10],  # margin = 0.80, outside gate
                ]
            ],
            dtype=torch.float32,
        )

        calibrated = apply_ctc_calibration(
            pred_probs,
            blank_bias=-0.6,
            nonblank_bias=0.6,
            margin_gate_min=0.1,
            margin_gate_max=0.3,
        )

        self.assertEqual(calibrated.argmax(-1)[0].tolist(), [1, 0])

    def test_apply_ctc_calibration_without_gate_keeps_global_bias_behavior(self):
        pred_probs = torch.tensor(
            [
                [
                    [0.60, 0.40],
                    [0.55, 0.45],
                ]
            ],
            dtype=torch.float32,
        )

        calibrated = apply_ctc_calibration(
            pred_probs,
            blank_bias=-0.6,
            nonblank_bias=0.6,
        )

        self.assertEqual(calibrated.argmax(-1)[0].tolist(), [1, 1])

    def test_apply_ctc_calibration_adaptive_empty_scales_bias(self):
        pred_probs = torch.tensor(
            [
                [
                    [0.55, 0.45],
                    [0.90, 0.10],
                ]
            ],
            dtype=torch.float32,
        )

        calibrated = apply_ctc_calibration(
            pred_probs,
            blank_bias=-0.6,
            nonblank_bias=0.6,
            adaptive_mode="empty",
            adaptive_min_scale=0.0,
            adaptive_max_scale=1.0,
        )

        self.assertEqual(calibrated.argmax(-1)[0].tolist(), [1, 0])

    def test_apply_ctc_calibration_adaptive_empty_preserves_nonempty_decode(self):
        pred_probs = torch.tensor(
            [
                [
                    [0.40, 0.60],
                    [0.90, 0.10],
                ]
            ],
            dtype=torch.float32,
        )

        calibrated = apply_ctc_calibration(
            pred_probs,
            blank_bias=-0.6,
            nonblank_bias=0.6,
            adaptive_mode="empty",
            adaptive_min_scale=0.0,
            adaptive_max_scale=1.0,
        )

        self.assertEqual(calibrated.argmax(-1)[0].tolist(), [1, 0])

    def test_apply_ctc_calibration_adaptive_pred_short_scales_short_decode(self):
        pred_probs = torch.tensor(
            [
                [
                    [0.55, 0.45],
                    [0.90, 0.10],
                    [0.90, 0.10],
                ]
            ],
            dtype=torch.float32,
        )

        calibrated = apply_ctc_calibration(
            pred_probs,
            blank_bias=-0.6,
            nonblank_bias=0.6,
            adaptive_mode="pred_short",
            adaptive_short_pred_max_len=1,
            adaptive_min_scale=0.0,
            adaptive_max_scale=1.0,
        )

        self.assertEqual(calibrated.argmax(-1)[0].tolist(), [1, 0, 0])

    def test_apply_ctc_calibration_adaptive_pred_short_preserves_long_decode(self):
        pred_probs = torch.tensor(
            [
                [
                    [0.40, 0.60, 0.00],
                    [0.90, 0.10, 0.00],
                    [0.40, 0.00, 0.60],
                ]
            ],
            dtype=torch.float32,
        )

        calibrated = apply_ctc_calibration(
            pred_probs,
            blank_bias=-0.6,
            nonblank_bias=0.6,
            adaptive_mode="pred_short",
            adaptive_short_pred_max_len=1,
            adaptive_min_scale=0.0,
            adaptive_max_scale=1.0,
        )

        self.assertEqual(calibrated.argmax(-1)[0].tolist(), [1, 0, 2])


class CtcMetricsTest(unittest.TestCase):
    def test_length_bin_boundaries(self):
        self.assertEqual(
            [length_bin(x) for x in [1, 2, 3, 5, 6, 10, 11]],
            ["1", "2", "3-5", "3-5", "6-10", "6-10", "11+"],
        )

    def test_levenshtein_ops_counts_insertions_deletions_and_substitutions(self):
        self.assertEqual(levenshtein_ops([1, 2, 3], [1, 4, 3, 5]), (2, 1, 0, 1))
        self.assertEqual(levenshtein_ops([1, 2, 3], [1]), (2, 0, 2, 0))

    def test_ctc_totals_summary_matches_existing_metric_names(self):
        totals = CtcTotals()
        totals.add(gt_len=2, pred_len=1, dist=1, ins=0, dels=1, subs=0)
        totals.add(gt_len=1, pred_len=0, dist=1, ins=0, dels=1, subs=0)
        summary = totals.to_summary()
        self.assertEqual(summary["samples"], 2)
        self.assertAlmostEqual(summary["cer_micro"], 2 / 3)
        self.assertAlmostEqual(summary["empty_pred_rate"], 0.5)
        self.assertAlmostEqual(summary["pred_gt_len_ratio"], 1 / 3)

    def test_summarize_by_length_uses_paper_bucket_order(self):
        rows = [
            {"gt_len": 1, "pred_len": 0, "dist": 1, "ins": 0, "dels": 1, "subs": 0},
            {"gt_len": 2, "pred_len": 2, "dist": 0, "ins": 0, "dels": 0, "subs": 0},
            {"gt_len": 12, "pred_len": 11, "dist": 1, "ins": 0, "dels": 1, "subs": 0},
        ]
        summary = summarize_by_length(rows)
        self.assertEqual(list(summary.keys()), ["1", "2", "11+"])
        self.assertEqual(summary["1"]["samples"], 1)
        self.assertEqual(summary["2"]["samples"], 1)
        self.assertEqual(summary["11+"]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
