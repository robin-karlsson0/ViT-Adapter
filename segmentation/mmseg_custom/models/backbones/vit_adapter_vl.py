# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
from mmcv.cnn import build_norm_layer

from .base.vit import VisionTransformerVL
from .adapter_modules import SpatialPriorModule, InteractionBlock, deform_inputs

_logger = logging.getLogger(__name__)


@BACKBONES.register_module()
class ViTAdapterVL(VisionTransformerVL):
    '''
    Changes
        self.blocks --> self.layers
    '''

    def __init__(
            self,
            # ViT parameters
            img_size=(512, 512),
            patch_size=16,
            patch_bias=False,
            in_channels=3,
            embed_dims=1024,
            num_layers=24,
            num_heads=16,
            mlp_ratio=4,
            out_indices=-1,
            qkv_bias=True,
            drop_rate=0.,
            drop_path_rate=0.4,
            with_cls_token=True,
            output_cls_token=False,
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            patch_norm=False,
            pre_norm=True,
            final_norm=True,
            return_qkv=True,
            interpolate_mode='bicubic',
            num_fcs=2,
            norm_eval=False,
            pretrained=None,
            # ViT-Adapter paramters
            pretrain_size=336,
            conv_inplane=64,
            n_points=4,
            deform_num_heads=16,
            cffn_ratio=0.25,
            deform_ratio=0.5,
            with_cp=True,  # set with_cp=True to save memory
            interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
            add_vit_feature=True,
            init_values=0.,
            with_cffn=True,
            use_extra_extractor=True,
            *args,
            **kwargs):

        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            patch_bias=patch_bias,
            in_channels=in_channels,
            embed_dims=embed_dims,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_indices=out_indices,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            with_cls_token=with_cls_token,
            output_cls_token=output_cls_token,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            patch_norm=patch_norm,
            pre_norm=pre_norm,
            final_norm=final_norm,
            return_qkv=return_qkv,
            interpolate_mode=interpolate_mode,
            num_fcs=num_fcs,
            norm_eval=norm_eval,
            with_cp=with_cp,
            pretrained=pretrained,
            init_cfg=None,
            #*args,
            #**kwargs,
        )

        # self.num_classes = 80
        self.cls_token = None
        self.num_block = len(self.layers)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.embed_dim = embed_dims
        embed_dim = self.embed_dim
        self.drop_path_rate = drop_path_rate
        # _, norm_layer = build_norm_layer(norm_cfg, embed_dims, postfix=1)
        # self.norm_layer = norm_layer
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim,
                                      with_cp=False)
        self.interactions = nn.Sequential(*[
            InteractionBlock(dim=embed_dim,
                             num_heads=deform_num_heads,
                             n_points=n_points,
                             init_values=init_values,
                             drop_path=self.drop_path_rate,
                             norm_layer=self.norm_layer,
                             with_cffn=with_cffn,
                             cffn_ratio=cffn_ratio,
                             deform_ratio=deform_ratio,
                             extra_extractor=(
                                 (True if i == len(interaction_indexes) -
                                  1 else False) and use_extra_extractor),
                             with_cp=with_cp)
            for i in range(len(interaction_indexes))
        ])
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(1, self.pretrain_size[0] // 16,
                                      self.pretrain_size[1] // 16,
                                      -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.layers[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs
            x1 = F.interpolate(x1,
                               scale_factor=4,
                               mode='bilinear',
                               align_corners=False)
            x2 = F.interpolate(x2,
                               scale_factor=2,
                               mode='bilinear',
                               align_corners=False)
            x4 = F.interpolate(x4,
                               scale_factor=0.5,
                               mode='bilinear',
                               align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
