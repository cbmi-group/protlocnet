# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging

from . import protnet as prots
from . import vision_transformer as vits


logger = logging.getLogger("dinov2")


def build_model(
        configs,
        only_teacher=False,
        img_size=224,
        reconstruction_mode=False,
        generation_mode=False
):
    configs.arch = configs.arch.removesuffix("_memeff")
    if configs.arch.startswith('prot'):
        kwargs = dict(
            img_size=img_size,
            patch_size=configs.patch_size,
            init_values=configs.layerscale,
            ffn_layer=configs.ffn_layer,
            block_chunks=configs.block_chunks,
            qkv_bias=configs.qkv_bias,
            proj_bias=configs.proj_bias,
            ffn_bias=configs.ffn_bias,
            num_register_tokens=configs.num_register_tokens,
            interpolate_offset=configs.interpolate_offset,
            interpolate_antialias=configs.interpolate_antialias,
            branch_type=configs.get("branch_type", "full"),
            reconstruction_mode=reconstruction_mode,
            generation_mode=generation_mode,
        )
        teacher = prots.__dict__[configs.arch](**kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = prots.__dict__[configs.arch](
            **kwargs,
            drop_path_rate=configs.drop_path_rate,
            drop_path_uniform=configs.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    elif "vit" in configs.arch:
        vit_kwargs = dict(
            img_size=img_size,
            patch_size=configs.patch_size,
            init_values=configs.layerscale,
            ffn_layer=configs.ffn_layer,
            block_chunks=configs.block_chunks,
            qkv_bias=configs.qkv_bias,
            proj_bias=configs.proj_bias,
            ffn_bias=configs.ffn_bias,
            num_register_tokens=configs.num_register_tokens,
            interpolate_offset=configs.interpolate_offset,
            interpolate_antialias=configs.interpolate_antialias,
            in_chans=configs.in_chans,
            channel_adaptive=configs.channel_adaptive,
        )
        teacher = vits.__dict__[configs.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[configs.arch](
            **vit_kwargs,
            drop_path_rate=configs.drop_path_rate,
            drop_path_uniform=configs.drop_path_uniform,
        )
        embed_dim = student.embed_dim
    return student, teacher, embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=cfg.crops.global_crops_size,
        reconstruction_mode=cfg.train.get("reconstruction", False),
        generation_mode=cfg.train.get("generation", False),
    )
