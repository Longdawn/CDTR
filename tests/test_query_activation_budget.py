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


class QueryActivationBudgetTest(unittest.TestCase):
    def setUp(self):
        self.dino = importlib.import_module("models.dino.dino")

    def test_query_budget_loss_penalizes_over_activation(self):
        pred_logits = torch.tensor(
            [
                [
                    [4.0, -4.0],
                    [-4.0, 4.0],
                    [-4.0, -4.0],
                ]
            ],
            dtype=torch.float32,
        )
        targets = [{"labels": torch.tensor([1], dtype=torch.long)}]

        loss, stats = self.dino.compute_query_budget_loss(
            pred_logits,
            targets,
            budget_scale=1.0,
            budget_margin=0.0,
        )

        self.assertGreater(loss.item(), 0.0)
        self.assertGreater(stats["expected_count_mean"], 1.0)
        self.assertEqual(stats["target_count_mean"], 1.0)

    def test_set_criterion_emits_budget_loss_when_weight_enabled(self):
        criterion = self.dino.SetCriterion(
            num_classes=2,
            matcher=None,
            weight_dict={"loss_ce": 1.0, "loss_query_budget": 0.01},
            focal_alpha=0.25,
            losses=["labels"],
            query_budget_scale=1.0,
            query_budget_margin=0.0,
        )
        outputs = {
            "pred_logits": torch.tensor(
                [[[4.0, -4.0], [-4.0, 4.0], [-4.0, -4.0]]],
                dtype=torch.float32,
            )
        }
        targets = [{"labels": torch.tensor([1], dtype=torch.long)}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        losses = criterion.loss_labels(
            outputs,
            targets,
            indices,
            num_boxes=torch.tensor(1.0),
        )

        self.assertIn("loss_query_budget", losses)
        self.assertIn("query_budget_expected_count", losses)
        self.assertIn("query_budget_target_count", losses)
        self.assertGreater(losses["loss_query_budget"].item(), 0.0)

    def test_set_criterion_omits_budget_loss_by_default(self):
        criterion = self.dino.SetCriterion(
            num_classes=2,
            matcher=None,
            weight_dict={"loss_ce": 1.0},
            focal_alpha=0.25,
            losses=["labels"],
        )
        outputs = {
            "pred_logits": torch.tensor(
                [[[4.0, -4.0], [-4.0, 4.0], [-4.0, -4.0]]],
                dtype=torch.float32,
            )
        }
        targets = [{"labels": torch.tensor([1], dtype=torch.long)}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]

        losses = criterion.loss_labels(
            outputs,
            targets,
            indices,
            num_boxes=torch.tensor(1.0),
        )

        self.assertIn("loss_ce", losses)
        self.assertNotIn("loss_query_budget", losses)

    def test_query_budget_loss_zero_when_under_budget(self):
        pred_logits = torch.tensor(
            [
                [
                    [-4.0, -4.0],
                    [-4.0, -4.0],
                    [0.0, 4.0],
                ]
            ],
            dtype=torch.float32,
        )
        targets = [{"labels": torch.tensor([1, 1, 1], dtype=torch.long)}]

        loss, stats = self.dino.compute_query_budget_loss(
            pred_logits,
            targets,
            budget_scale=2.0,
            budget_margin=1.0,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertLess(stats["expected_count_mean"], stats["target_count_mean"])

    def test_query_budget_uses_detection_sigmoid_activation_not_blank_softmax(self):
        pred_logits = torch.full((1, 2, 3), -4.0, dtype=torch.float32)
        targets = [{"labels": torch.tensor([1], dtype=torch.long)}]

        _loss, stats = self.dino.compute_query_budget_loss(
            pred_logits,
            targets,
            budget_scale=1.0,
            budget_margin=0.0,
        )

        expected = 2.0 * torch.sigmoid(torch.tensor(-4.0)).item()
        self.assertAlmostEqual(stats["expected_count_mean"], expected, places=6)

    def test_query_budget_loss_handles_empty_target(self):
        pred_logits = torch.tensor(
            [[[4.0, -4.0], [-4.0, 4.0]]],
            dtype=torch.float32,
        )
        targets = [{"labels": torch.empty(0, dtype=torch.long)}]

        loss, stats = self.dino.compute_query_budget_loss(
            pred_logits,
            targets,
            budget_scale=2.0,
            budget_margin=1.0,
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertEqual(stats["target_count_mean"], 1.0)


class QueryActivationBudgetConfigTest(unittest.TestCase):
    def test_mthv2_stage1_query_budget_config_sets_expected_flags(self):
        config = load_config(ROOT / "config" / "MTHV2_stage1_query_budget.py")

        self.assertEqual(config.mth_combo_roots, [
            "tkhmth2200_mth1000_dtlr",
            "tkhmth2200_mth1200_dtlr",
            "tkhmth2200_tkh_dtlr",
        ])
        self.assertFalse(config.mode_chr)
        self.assertTrue(config.use_query_budget_loss)
        self.assertEqual(config.query_budget_loss_coef, 0.01)
        self.assertEqual(config.query_budget_scale, 2.0)
        self.assertEqual(config.query_budget_margin, 8.0)


if __name__ == "__main__":
    unittest.main()
