# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.models.builder import HEADS
from mmseg.ops import Upsample, resize

from mmseg_custom.models.decode_heads.decode_head_vl import BaseDecodeHeadVL
from tools.convert_datasets.txt2idx_star import load_register


@HEADS.register_module()
class SimpleHeadVL(BaseDecodeHeadVL):
    """Simple head for vision-language embedding output

    Takes the largest ViT-Adapter output feature map (B, D, H, W) and
    transforms it into an embedding map (B, D_emb, H, W) using a single 1x1
    convolution and resize operation.
    
    """

    def __init__(self,
                 output_size: tuple,
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
        super(SimpleHeadVL, self).__init__(input_transform='multiple_select',
                                           num_classes=1,
                                           in_index=[0, 1, 2, 3],
                                           **kwargs)
        self.output_size = output_size
        self.normalize_output = normalize_output
        self.normalize_target_embs = normalize_target_embs

        self.idx_star2emb = load_register(idx_star2emb_path)
        if self.normalize_target_embs:
            for key in self.idx_star2emb.keys():
                emb = self.idx_star2emb[key]
                emb = F.normalize(emb)
                self.idx_star2emb[key] = emb
        self.ignore_emb_idx = ignore_emb_idx

        self.conv_seg = None  # Remove to avoid unused parameters

        self.conv = ConvModule(self.in_channels[0],
                               self.channels,
                               kernel_size=1)

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
        '''
        Original implementation
            out = self.conv(inputs[0])
            out = F.normalize(out)
            out = F.interpolate(out, self.output_size, mode='nearest')
        '''
        # Upsample
        out = F.interpolate(inputs[0], self.output_size, mode='bilinear')

        # Reduce dim by 1x1 conv
        out = self.conv(out)

        # Normalize
        if self.normalize_output:
            out = F.normalize(out)

        return out

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