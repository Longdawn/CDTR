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


class CTCViterbiAlignmentLossTest(unittest.TestCase):
    def setUp(self):
        self.dino = importlib.import_module("models.dino.dino")

    def test_viterbi_path_follows_best_ctc_alignment(self):
        probs = torch.tensor(
            [
                [0.90, 0.05, 0.05],
                [0.05, 0.90, 0.05],
                [0.90, 0.05, 0.05],
            ],
            dtype=torch.float32,
        )
        path = self.dino._ctc_viterbi_path(torch.log(probs), [1], blank=0)

        self.assertEqual(path.tolist(), [0, 1, 0])

    def test_alignment_loss_backpropagates_through_log_probs(self):
        logits = torch.tensor(
            [
                [[2.0, 0.0, -1.0]],
                [[0.0, 2.0, -1.0]],
                [[2.0, 0.0, -1.0]],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        log_probs = logits.log_softmax(-1)
        targets = torch.tensor([[1]], dtype=torch.long)
        target_lengths = torch.tensor([1], dtype=torch.long)

        loss, stats = self.dino.compute_ctc_viterbi_alignment_loss(
            log_probs,
            targets,
            target_lengths,
            blank=0,
            blank_weight=0.0,
            nonblank_weight=1.0,
        )
        loss.backward()

        self.assertEqual(stats["selected_samples"], 1)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)


class CTCViterbiAlignmentConfigTest(unittest.TestCase):
    def test_mthv2_dctc_lite_config_sets_expected_flags(self):
        config = load_config(ROOT / "config" / "MTHV2_dtlr_dctc_lite.py")

        self.assertEqual(config.ctc_viterbi_loss_coef, 0.02)
        self.assertEqual(config.ctc_viterbi_loss_blank_weight, 0.0)
        self.assertEqual(config.ctc_viterbi_loss_nonblank_weight, 1.0)
        self.assertEqual(config.ctc_viterbi_loss_max_len, 2)


if __name__ == "__main__":
    unittest.main()
