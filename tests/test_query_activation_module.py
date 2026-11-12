import importlib
import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn

from util.misc import NestedTensor


ROOT = Path(__file__).resolve().parents[1]


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QueryActivationModuleTest(unittest.TestCase):
    def setUp(self):
        self.dino_tool = importlib.import_module("models.dino.dino")

    def test_apply_activation_gating_is_identity_when_disabled(self):
        logits = torch.tensor([[[0.2, 0.3, 0.5]]], dtype=torch.float32)
        activation_logits = torch.tensor([[[0.7]]], dtype=torch.float32)
        gated = self.dino_tool.apply_activation_gating(
            logits,
            activation_logits,
            use_activation_gating=False,
            activation_gate_blank_coef=0.3,
            activation_gate_nonblank_coef=0.4,
        )
        self.assertTrue(torch.equal(gated, logits))

    def test_apply_activation_gating_boosts_nonblank_and_suppresses_blank(self):
        logits = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float32)
        activation_logits = torch.tensor([[[2.0]]], dtype=torch.float32)
        gated = self.dino_tool.apply_activation_gating(
            logits,
            activation_logits,
            use_activation_gating=True,
            activation_gate_blank_coef=0.5,
            activation_gate_nonblank_coef=0.25,
        )
        activation = torch.sigmoid(torch.tensor(2.0))
        self.assertAlmostEqual(gated[0, 0, 0].item(), float((1.0 - activation) * 0.5), places=6)
        self.assertAlmostEqual(gated[0, 0, 1].item(), float(activation * 0.25), places=6)
        self.assertAlmostEqual(gated[0, 0, 2].item(), float(activation * 0.25), places=6)

    def test_query_count_loss_matches_transcript_length(self):
        activation_logits = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        target_lengths = torch.tensor([1.0], dtype=torch.float32)
        loss, expected = self.dino_tool.compute_query_count_loss(
            activation_logits,
            target_lengths,
            short_weight=1.0,
        )
        self.assertAlmostEqual(expected.item(), 1.0, places=6)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_query_count_loss_applies_short_weight(self):
        activation_logits = torch.tensor([[-2.0, -2.0], [0.0, 0.0]], dtype=torch.float32)
        target_lengths = torch.tensor([1.0, 4.0], dtype=torch.float32)
        loss_unweighted, _ = self.dino_tool.compute_query_count_loss(
            activation_logits,
            target_lengths,
            short_weight=1.0,
        )
        loss_weighted, _ = self.dino_tool.compute_query_count_loss(
            activation_logits,
            target_lengths,
            short_weight=2.0,
        )
        self.assertGreater(loss_weighted.item(), loss_unweighted.item())

    def test_query_activation_head_bias_uses_init_value(self):
        model = self._build_fake_dino(use_query_activation_head=True, query_activation_init_bias=-4.0)
        self.assertTrue(torch.allclose(model.query_activation_embed.bias.data, torch.full_like(model.query_activation_embed.bias.data, -4.0)))

    def test_model_without_query_activation_head_does_not_emit_pred_activation_logits(self):
        model = self._build_fake_dino(use_query_activation_head=False)
        samples = NestedTensor(torch.randn(1, 3, 8, 8), torch.zeros(1, 8, 8, dtype=torch.bool))
        outputs = model(samples)
        self.assertNotIn("pred_activation_logits", outputs)

    def _build_fake_dino(self, use_query_activation_head, query_activation_init_bias=0.0):
        class FakeDecoderLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.label_embedding = None

        class FakeDecoder(nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.layers = nn.ModuleList([FakeDecoderLayer() for _ in range(num_layers)])
                self.bbox_embed = None
                self.class_embed = None

        class FakeTransformer(nn.Module):
            def __init__(self, d_model=32, num_decoder_layers=1, num_queries=4):
                super().__init__()
                self.d_model = d_model
                self.num_decoder_layers = num_decoder_layers
                self.num_queries = num_queries
                self.decoder = FakeDecoder(num_decoder_layers)

            def forward(self, srcs, masks, input_query_bbox, poss, input_query_label, attn_mask):
                batch_size = srcs[0].shape[0]
                hs = torch.zeros(self.num_decoder_layers, batch_size, self.num_queries, self.d_model)
                reference = torch.full((self.num_decoder_layers + 1, batch_size, self.num_queries, 4), 0.5)
                return hs, reference, None, None, None

        class FakeBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_channels = [3]

            def __getitem__(self, index):
                return self

            def forward(self, samples):
                feature = NestedTensor(samples.tensors, samples.mask)
                pos = torch.zeros(samples.tensors.shape[0], 32, samples.tensors.shape[2], samples.tensors.shape[3])
                return [feature], [pos]

        return self.dino_tool.DINO(
            backbone=FakeBackbone(),
            transformer=FakeTransformer(),
            num_classes=8,
            num_queries=4,
            aux_loss=False,
            iter_update=True,
            query_dim=4,
            num_feature_levels=1,
            two_stage_type="no",
            use_query_activation_head=use_query_activation_head,
            query_activation_init_bias=query_activation_init_bias,
        )


class QueryActivationConfigSmokeTest(unittest.TestCase):
    def test_new_configs_define_expected_query_activation_flags(self):
        count_only = load_module(ROOT / "config" / "MTH1000_dtlr_csaq_count_only.py")
        gating_only = load_module(ROOT / "config" / "MTH1000_dtlr_csaq_gating_only.py")
        count_gating = load_module(ROOT / "config" / "MTH1000_dtlr_csaq_count_gating.py")

        self.assertTrue(count_only.use_query_activation_head)
        self.assertEqual(count_only.query_activation_init_bias, -4.0)
        self.assertTrue(count_only.use_query_count_loss)
        self.assertFalse(count_only.use_activation_gating)

        self.assertTrue(gating_only.use_query_activation_head)
        self.assertEqual(gating_only.query_activation_init_bias, -4.0)
        self.assertFalse(gating_only.use_query_count_loss)
        self.assertTrue(gating_only.use_activation_gating)

        self.assertTrue(count_gating.use_query_activation_head)
        self.assertEqual(count_gating.query_activation_init_bias, -4.0)
        self.assertTrue(count_gating.use_query_count_loss)
        self.assertTrue(count_gating.use_activation_gating)


if __name__ == "__main__":
    unittest.main()
