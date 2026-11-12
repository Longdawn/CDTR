# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math
import os
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction, ms_deform_attn_core_pytorch


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


def _env_flag(name, default="0"):
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _tensor_summary(name, tensor):
    if tensor is None:
        return f"{name}=None"
    if not torch.is_tensor(tensor):
        return f"{name}=<{type(tensor).__name__}>"

    parts = [
        f"{name}.shape={tuple(tensor.shape)}",
        f"{name}.dtype={tensor.dtype}",
        f"{name}.device={tensor.device}",
    ]
    if tensor.numel() == 0:
        parts.append(f"{name}.empty=True")
        return " ".join(parts)

    finite_mask = torch.isfinite(tensor)
    finite = bool(finite_mask.all().item())
    parts.append(f"{name}.finite={finite}")
    if finite:
        parts.append(f"{name}.min={tensor.min().item():.6g}")
        parts.append(f"{name}.max={tensor.max().item():.6g}")
        parts.append(f"{name}.mean={tensor.float().mean().item():.6g}")
    else:
        if torch.is_floating_point(tensor):
            parts.append(f"{name}.nan_count={int(torch.isnan(tensor).sum().item())}")
            parts.append(f"{name}.inf_count={int(torch.isinf(tensor).sum().item())}")
    return " ".join(parts)


def _build_debug_context(
    query,
    reference_points,
    input_flatten,
    input_spatial_shapes,
    input_level_start_index,
    input_padding_mask,
    value,
    sampling_offsets,
    attention_weights,
    sampling_locations,
):
    return " | ".join(
        [
            _tensor_summary("query", query),
            _tensor_summary("reference_points", reference_points),
            _tensor_summary("input_flatten", input_flatten),
            _tensor_summary("input_spatial_shapes", input_spatial_shapes),
            _tensor_summary("input_level_start_index", input_level_start_index),
            _tensor_summary("input_padding_mask", input_padding_mask),
            _tensor_summary("value", value),
            _tensor_summary("sampling_offsets", sampling_offsets),
            _tensor_summary("attention_weights", attention_weights),
            _tensor_summary("sampling_locations", sampling_locations),
        ]
    )


_MSDA_DUMP_COUNTER = 0


def _maybe_dump_msda_inputs(
    value,
    input_spatial_shapes,
    input_level_start_index,
    sampling_locations,
    attention_weights,
):
    global _MSDA_DUMP_COUNTER
    dump_dir = os.environ.get("DTLR_MSDA_DUMP_DIR", "").strip()
    if not dump_dir:
        return
    if _MSDA_DUMP_COUNTER > 0 and not _env_flag("DTLR_MSDA_DUMP_EVERY_CALL"):
        return

    path = Path(dump_dir)
    path.mkdir(parents=True, exist_ok=True)
    dump_path = path / f"msda_dump_{_MSDA_DUMP_COUNTER:04d}.pt"
    payload = {
        "value": value.detach().cpu(),
        "spatial_shapes": input_spatial_shapes.detach().cpu(),
        "level_start_index": input_level_start_index.detach().cpu(),
        "sampling_locations": sampling_locations.detach().cpu(),
        "attention_weights": attention_weights.detach().cpu(),
    }
    torch.save(payload, dump_path)
    _MSDA_DUMP_COUNTER += 1


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        debug_enabled = _env_flag("DTLR_MSDA_DIAG")
        force_pytorch = _env_flag("DTLR_MSDA_FORCE_PYTORCH")
        fallback_on_error = _env_flag("DTLR_MSDA_FALLBACK_ON_ERROR")

        debug_context = None
        if debug_enabled:
            debug_context = _build_debug_context(
                query=query,
                reference_points=reference_points,
                input_flatten=input_flatten,
                input_spatial_shapes=input_spatial_shapes,
                input_level_start_index=input_level_start_index,
                input_padding_mask=input_padding_mask,
                value=value,
                sampling_offsets=sampling_offsets,
                attention_weights=attention_weights,
                sampling_locations=sampling_locations,
            )

        def _run_pytorch_core(core_value, core_locations, output_dtype):
            core_output = ms_deform_attn_core_pytorch(
                core_value,
                input_spatial_shapes,
                core_locations,
                attention_weights.to(core_value.dtype),
            )
            if output_dtype is not None:
                core_output = core_output.to(output_dtype)
            return self.output_proj(core_output.to(query.dtype))

        kernel_value = value
        kernel_locations = sampling_locations
        kernel_attention_weights = attention_weights
        output_cast_dtype = None

        if value.dtype == torch.float16:
            kernel_value = value.to(torch.float32)
            kernel_locations = sampling_locations.to(torch.float32)
            output_cast_dtype = torch.float16

        _maybe_dump_msda_inputs(
            kernel_value,
            input_spatial_shapes,
            input_level_start_index,
            kernel_locations,
            kernel_attention_weights,
        )

        if force_pytorch:
            return _run_pytorch_core(kernel_value, kernel_locations, output_cast_dtype)

        try:
            output = MSDeformAttnFunction.apply(
                kernel_value,
                input_spatial_shapes,
                input_level_start_index,
                kernel_locations,
                kernel_attention_weights,
                self.im2col_step,
            )
            if output_cast_dtype is not None:
                output = output.to(output_cast_dtype)
            output = self.output_proj(output)
            return output
        except Exception as exc:
            if fallback_on_error:
                warnings.warn(
                    "MSDeformAttn CUDA path failed; falling back to PyTorch core for debugging."
                )
                return _run_pytorch_core(kernel_value, kernel_locations, output_cast_dtype)
            if debug_context is not None:
                raise RuntimeError(f"{exc}\n[MSDeformAttn debug] {debug_context}") from exc
            raise
