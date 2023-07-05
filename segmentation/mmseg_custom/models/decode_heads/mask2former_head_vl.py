import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList, force_fp32
from mmseg.models.builder import HEADS, build_loss
from mmseg_custom.models.decode_heads.mask2former_head import Mask2FormerHead

from ...core import build_sampler, multi_apply, reduce_mean
from ..builder import build_assigner
from ..utils import get_uncertain_point_coords_with_randomness


@HEADS.register_module()
class Mask2FormerHeadVL(Mask2FormerHead):
    """
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(Mask2FormerHeadVL, self).__init__(
            in_channels,
            feat_channels,
            out_channels,
            num_things_classes,
            num_stuff_classes,
            num_queries,
            num_transformer_feat_level,
            pixel_decoder,
            enforce_decoder_input_project,
            transformer_decoder,
            positional_encoding,
            loss_cls,
            loss_mask,
            loss_dice,
            train_cfg,
            test_cfg,
            init_cfg,
            **kwargs,
        )
