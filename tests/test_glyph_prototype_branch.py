import importlib
import importlib.util
import random
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from util.misc import NestedTensor


ROOT = Path(__file__).resolve().parents[1]


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GlyphPrototypeBranchTest(unittest.TestCase):
    def setUp(self):
        self.dino_tool = importlib.import_module("models.dino.dino")

    def test_baseline_model_does_not_emit_pred_proto_logits(self):
        model = self._build_fake_dino(use_glyph_prototype_head=False)
        samples = NestedTensor(
            torch.randn(1, 3, 8, 8), torch.zeros(1, 8, 8, dtype=torch.bool)
        )
        outputs = model(samples)
        self.assertNotIn("pred_proto_logits", outputs)

    def test_proto_model_emits_pred_proto_logits_matching_pred_logits_shape(self):
        model = self._build_fake_dino(
            use_glyph_prototype_head=True,
            glyph_proto_dim=16,
            glyph_proto_temperature=0.5,
        )
        samples = NestedTensor(
            torch.randn(2, 3, 8, 8), torch.zeros(2, 8, 8, dtype=torch.bool)
        )
        outputs = model(samples)
        self.assertIn("pred_proto_logits", outputs)
        self.assertEqual(outputs["pred_proto_logits"].shape, outputs["pred_logits"].shape)

    def test_fuse_coef_zero_preserves_base_logits(self):
        pred_logits = torch.tensor([[[0.2, -0.3], [0.5, 0.1]]], dtype=torch.float32)
        pred_proto_logits = torch.tensor([[[4.0, -5.0], [7.0, 3.0]]], dtype=torch.float32)
        fused = self.dino_tool.fuse_glyph_prototype_logits(
            pred_logits,
            pred_proto_logits,
            glyph_proto_fuse_coef=0.0,
        )
        self.assertTrue(torch.equal(fused, pred_logits))

    def test_fuse_coef_nonzero_adds_scaled_proto_logits(self):
        pred_logits = torch.tensor([[[0.2, -0.3]]], dtype=torch.float32)
        pred_proto_logits = torch.tensor([[[1.0, -2.0]]], dtype=torch.float32)
        fused = self.dino_tool.fuse_glyph_prototype_logits(
            pred_logits,
            pred_proto_logits,
            glyph_proto_fuse_coef=0.1,
        )
        expected = torch.tensor([[[0.3, -0.5]]], dtype=torch.float32)
        self.assertTrue(torch.allclose(fused, expected))

    def test_trainable_flag_freezes_prototype_table_when_disabled(self):
        model = self._build_fake_dino(
            use_glyph_prototype_head=True,
            glyph_proto_trainable=False,
        )
        self.assertFalse(model.glyph_prototype_table.weight.requires_grad)

    def test_glyph_proto_aux_loss_selects_queries_from_base_char_probs(self):
        pred_proto_logits = torch.tensor(
            [
                [
                    [-1.0, -1.0, 4.0],
                    [-1.0, 4.0, -1.0],
                    [4.0, -1.0, -1.0],
                ]
            ],
            dtype=torch.float32,
        )
        char_probs = torch.tensor(
            [
                [
                    [0.10, 0.20, 0.90],
                    [0.10, 0.90, 0.20],
                    [0.90, 0.20, 0.10],
                ]
            ],
            dtype=torch.float32,
        )
        targets = [{"labels": torch.tensor([2, 1], dtype=torch.long)}]

        loss, stats = self.dino_tool.compute_glyph_proto_aux_loss(
            pred_proto_logits,
            char_probs,
            targets,
        )

        self.assertLess(loss.item(), 0.05)
        self.assertEqual(stats["eligible_samples"], 1)
        self.assertEqual(stats["selected_chars"], 2)

    def test_prototype_head_does_not_shift_baseline_sensitive_random_init(self):
        features_dim = 32
        num_classes = 8
        new_classes = 5

        def reset_seed():
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

        def rebuild_sensitive_params(model):
            new_class_embed = nn.ModuleList(
                [nn.Linear(features_dim, new_classes, bias=True) for _ in range(len(model.class_embed))]
            )
            new_decoder_class_embed = nn.Linear(features_dim, new_classes, bias=True)
            new_enc_out_class_embed = nn.Linear(features_dim, new_classes, bias=True)
            new_label_enc = nn.Embedding(new_classes, features_dim)
            return {
                "class_embed_0_weight": new_class_embed[0].weight.detach().clone(),
                "class_embed_0_bias": new_class_embed[0].bias.detach().clone(),
                "decoder_class_embed_weight": new_decoder_class_embed.weight.detach().clone(),
                "decoder_class_embed_bias": new_decoder_class_embed.bias.detach().clone(),
                "enc_out_class_embed_weight": new_enc_out_class_embed.weight.detach().clone(),
                "enc_out_class_embed_bias": new_enc_out_class_embed.bias.detach().clone(),
                "label_enc_weight": new_label_enc.weight.detach().clone(),
            }

        reset_seed()
        baseline_model = self._build_fake_dino(use_glyph_prototype_head=False)
        baseline_params = rebuild_sensitive_params(baseline_model)

        reset_seed()
        proto_model = self._build_fake_dino(use_glyph_prototype_head=True)
        proto_params = rebuild_sensitive_params(proto_model)

        for key in baseline_params:
            self.assertTrue(
                torch.equal(baseline_params[key], proto_params[key]),
                msg=f"{key} diverged when prototype head was enabled",
            )

    def test_proto_logits_trim_to_current_class_head_size(self):
        model = self._build_fake_dino(
            use_glyph_prototype_head=True,
            glyph_proto_dim=16,
        )
        new_head = nn.Linear(model.hidden_dim, 5)
        model.class_embed = nn.ModuleList([new_head])
        model.transformer.decoder.class_embed = model.class_embed
        samples = NestedTensor(
            torch.randn(1, 3, 8, 8), torch.zeros(1, 8, 8, dtype=torch.bool)
        )
        outputs = model(samples)
        self.assertEqual(outputs["pred_logits"].shape[-1], 5)
        self.assertEqual(outputs["pred_proto_logits"].shape[-1], 5)

    def _build_fake_dino(
        self,
        use_glyph_prototype_head,
        glyph_proto_dim=8,
        glyph_proto_temperature=1.0,
        glyph_proto_trainable=True,
    ):
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
                hs = torch.randn(self.num_decoder_layers, batch_size, self.num_queries, self.d_model)
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
                pos = torch.zeros(
                    samples.tensors.shape[0],
                    32,
                    samples.tensors.shape[2],
                    samples.tensors.shape[3],
                )
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
            use_glyph_prototype_head=use_glyph_prototype_head,
            glyph_proto_dim=glyph_proto_dim,
            glyph_proto_temperature=glyph_proto_temperature,
            glyph_proto_trainable=glyph_proto_trainable,
        )


class GlyphPrototypeConfigSmokeTest(unittest.TestCase):
    def test_proto_learnable_fuse_config_sets_expected_flags(self):
        config = load_module(ROOT / "config" / "MTH1000_dtlr_proto_learnable_fuse.py")
        self.assertTrue(config.use_glyph_prototype_head)
        self.assertEqual(config.glyph_proto_dim, 256)
        self.assertEqual(config.glyph_proto_fuse_coef, 0.1)
        self.assertEqual(config.glyph_proto_temperature, 1.0)
        self.assertTrue(config.glyph_proto_trainable)

    def test_mthv2_proto_aux_config_sets_expected_flags(self):
        config = load_module(ROOT / "config" / "MTHV2_dtlr_proto_aux.py")
        self.assertTrue(config.use_glyph_prototype_head)
        self.assertEqual(config.glyph_proto_dim, 256)
        self.assertEqual(config.glyph_proto_fuse_coef, 0.0)
        self.assertEqual(config.glyph_proto_aux_loss_coef, 0.01)
        self.assertEqual(config.glyph_proto_aux_max_len, 0)


if __name__ == "__main__":
    unittest.main()
