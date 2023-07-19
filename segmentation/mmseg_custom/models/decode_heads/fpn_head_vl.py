# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.ops import Upsample, resize
from mmseg.models.builder import HEADS
from mmseg_custom.models.decode_heads.decode_head_vl import BaseDecodeHeadVL
from tools.convert_datasets.txt2idx_star import load_register


@HEADS.register_module()
class FPNHeadVL(BaseDecodeHeadVL):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self,
                 feature_strides,
                 output_size: tuple,
                 add_feat_maps: bool = True,
                 normalize_output: bool = True,
                 normalize_target_embs: bool = True,
                 idx_star2emb_path: str = 'idx_star2emb.pkl',
                 ignore_emb_idx: int = np.iinfo(np.uint32).max,
                 **kwargs):
        '''
        Args:
            add_feat_maps: Add multi-scale feature maps if True. Return largest
                           feature map if False.
            normalize_output: Normalize final output embedding map if True.
        '''
        super(FPNHeadVL, self).__init__(input_transform='multiple_select',
                                        num_classes=1,
                                        **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.output_size = output_size
        self.add_feat_maps = add_feat_maps
        self.normalize_output = normalize_output
        self.normalize_target_embs = normalize_target_embs

        if self.add_feat_maps:
            self.scale_heads = nn.ModuleList()
            for i in range(len(feature_strides)):
                head_length = max(
                    1,
                    int(
                        np.log2(feature_strides[i]) -
                        np.log2(feature_strides[0])))
                scale_head = []
                for k in range(head_length):
                    scale_head.append(
                        ConvModule(
                            self.in_channels[i] if k == 0 else self.channels,
                            self.channels,
                            3,
                            padding=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg))
                    if feature_strides[i] != feature_strides[0]:
                        scale_head.append(
                            Upsample(scale_factor=2,
                                     mode='bilinear',
                                     align_corners=self.align_corners))
                self.scale_heads.append(nn.Sequential(*scale_head))

        self.idx_star2emb = load_register(idx_star2emb_path)
        if self.normalize_target_embs:
            for key in self.idx_star2emb.keys():
                emb = self.idx_star2emb[key]
                emb = F.normalize(emb)
                self.idx_star2emb[key] = emb
        self.ignore_emb_idx = ignore_emb_idx

        self.conv_seg = None  # Remove to avoid unused parameters

    def label_idx2emb(self, idx_maps: torch.tensor) -> torch.tensor:
        '''
        Generates target embedding maps from idx maps.

        Args:
            idx_maps: (B, 1, H, W)
        
        Returns:
            Target embedding maps (B, D, H, W).
        '''
        B, _, H, W = idx_maps.shape
        device = idx_maps.device
        emb_maps = torch.zeros((B, self.channels, H, W), device=device)

        for batch_idx in range(B):
            idx_map = idx_maps[batch_idx, 0]  # (H, W)
            idxs = torch.unique(idx_map).tolist()
            for idx in idxs:
                emb = self.idx_star2emb[idx][0]  # (D)
                emb = emb.to(device)
                emb = emb.unsqueeze(-1).unsqueeze(-1).expand(-1, H, W)
                mask = idx_map == idx
                mask = mask.expand(self.channels, -1, -1)

                emb_maps[batch_idx][mask] = emb[mask]

        return emb_maps

    def label_idx2mask(self, idx_maps: torch.tensor) -> list:
        '''
        Generates target semantic masks from idx maps.

        Args:
            idx_maps: (B, 1, H, W)
        
        Returns:
            List of target semantic masks (K, H, W).
        '''
        B, _, H, W = idx_maps.shape
        device = idx_maps.device

        batch_masks = []
        for batch_idx in range(B):
            # Generate mask for batch sample 'i'
            idx_map = idx_maps[batch_idx, 0]  # (H, W)
            idxs = torch.unique(idx_map).tolist()
            mask_shape = (len(idxs), H, W)
            masks = torch.zeros(mask_shape, dtype=torch.bool)
            for mask_idx, sem_idx in enumerate(idxs):
                mask = idx_map == sem_idx
                masks[mask_idx][mask] = True
            masks = masks.to(device)

            batch_masks.append(masks)

        return batch_masks

    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        if self.add_feat_maps:
            output = self.scale_heads[0](x[0])
            for i in range(1, len(self.feature_strides)):
                # non inplace
                output = output + resize(self.scale_heads[i](x[i]),
                                         size=output.shape[2:],
                                         mode='bilinear',
                                         align_corners=self.align_corners)
        else:
            output = x[0]

        # Normalize
        if self.normalize_output:
            output = F.normalize(output)

        # Upsample
        output = F.interpolate(output, self.output_size, mode='nearest')

        return output

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        pred_embs = self.forward(inputs)

        # Label 'idx' --> 'emb' maps (B,D,H,W)
        label_embs = self.label_idx2emb(gt_semantic_seg)
        # Label 'idx' --> mask maps (list of (K,H,W) bool tensors)
        label_masks = self.label_idx2mask(gt_semantic_seg)

        losses = {}
        loss_decode = self.loss_decode(pred_embs, label_embs, label_masks)
        losses[self.loss_decode.loss_name] = loss_decode

        return losses