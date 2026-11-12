import importlib
import importlib.util
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShortGlyphQueryLossTest(unittest.TestCase):
    def setUp(self):
        self.dino = importlib.import_module("models.dino.dino")

    def test_short_query_ce_selects_distinct_best_queries(self):
        char_probs = torch.tensor(
            [
                [
                    [0.80, 0.10, 0.10],
                    [0.70, 0.20, 0.10],
                    [0.05, 0.90, 0.05],
                ]
            ],
            dtype=torch.float32,
        )
        targets = [{"labels": torch.tensor([0, 1], dtype=torch.long)}]

        loss, stats = self.dino.compute_short_query_ce_loss(
            char_probs,
            targets,
            max_len=2,
            gamma=0.0,
        )

        expected = (-torch.log(torch.tensor(0.80)) - torch.log(torch.tensor(0.90))) / 2.0
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        self.assertEqual(stats["eligible_samples"], 1)
        self.assertEqual(stats["selected_chars"], 2)

    def test_short_query_ce_skips_samples_longer_than_max_len(self):
        char_probs = torch.full((1, 3, 4), 0.25, dtype=torch.float32)
        targets = [{"labels": torch.tensor([0, 1, 2], dtype=torch.long)}]

        loss, stats = self.dino.compute_short_query_ce_loss(
            char_probs,
            targets,
            max_len=2,
            gamma=0.0,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(stats["eligible_samples"], 0)
        self.assertEqual(stats["selected_chars"], 0)

    def test_short_query_ce_applies_focal_gamma(self):
        char_probs = torch.tensor([[[0.25, 0.75]]], dtype=torch.float32)
        targets = [{"labels": torch.tensor([0], dtype=torch.long)}]

        loss, stats = self.dino.compute_short_query_ce_loss(
            char_probs,
            targets,
            max_len=1,
            gamma=2.0,
        )

        prob = torch.tensor(0.25)
        expected = ((1.0 - prob) ** 2.0) * -torch.log(prob)
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)
        self.assertEqual(stats["selected_chars"], 1)


class ShortGlyphQueryConfigTest(unittest.TestCase):
    def test_mthv2_sgq_config_targets_mth_combo(self):
        config = load_config(ROOT / "config" / "MTHV2_dtlr_sgq_short_ce.py")

        self.assertEqual(config.num_classes, 6700)
        self.assertEqual(config.short_gt_ce_loss_coef, 0.02)
        self.assertEqual(config.short_gt_ce_max_len, 2)
        self.assertEqual(config.short_gt_ce_gamma, 1.0)
        self.assertEqual(config.mth_combo_roots, [
            "tkhmth2200_mth1000_dtlr",
            "tkhmth2200_mth1200_dtlr",
            "tkhmth2200_tkh_dtlr",
        ])
        self.assertFalse(hasattr(config, "chdac_root"))

    def test_hdrc_sgq_config_targets_hdrc_lines(self):
        config = load_config(ROOT / "config" / "HDRC_dtlr_sgq_short_ce.py")

        self.assertEqual(config.num_classes, 7356)
        self.assertEqual(config.short_gt_ce_loss_coef, 0.02)
        self.assertEqual(config.short_gt_ce_max_len, 2)
        self.assertEqual(config.short_gt_ce_gamma, 1.0)
        self.assertEqual(config.mth1000_root, "hdrc_dtlr")
        self.assertEqual(config.mth1000_raw_root, "HDRC")
        self.assertEqual(config.mth1000_image_ext, "jpg")
        self.assertFalse(hasattr(config, "chdac_root"))


if __name__ == "__main__":
    unittest.main()
