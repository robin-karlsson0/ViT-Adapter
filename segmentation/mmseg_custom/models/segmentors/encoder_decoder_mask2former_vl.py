import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.core import add_prefix
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.ops import resize
from mmseg_custom.models.segmentors.encoder_decoder_mask2former import EncoderDecoderMask2Former
from tools.convert_datasets.txt2idx_star import load_register


@SEGMENTORS.register_module()
class EncoderDecoderMask2FormerVL(EncoderDecoderMask2Former):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 idx_star2emb_path=None):
        super(EncoderDecoderMask2FormerVL,
              self).__init__(backbone, decode_head, neck, auxiliary_head,
                             train_cfg, test_cfg, pretrained, init_cfg)

        self.idx_star2emb = load_register(idx_star2emb_path)

        self.valid_idxs = set(self.idx_star2emb.keys())
        emb = list(self.idx_star2emb.values())[0]
        self.emb_dim = emb.shape[1]

        # TMP
        self.backbone.eval()

    def embed_seg_masks(self, gt_semantic_seg):
        """
        Args:
            gt_semantic_seg: Map (B,1,H,W) with indices specifying embeddings .
        """
        device = gt_semantic_seg.device

        b, _, h, w = gt_semantic_seg.shape

        emb_maps = []
        for batch_idx in range(b):

            idx_map = gt_semantic_seg[batch_idx][0]
            emb_map = torch.zeros((h, w, self.emb_dim), device=device)

            idxs = set(list(torch.unique(idx_map).cpu().numpy()))
            idxs = idxs.intersection(self.valid_idxs)

            for idx in idxs:
                emb = self.idx_star2emb[idx]
                emb = emb.to(device)
                mask = idx_map == torch.tensor(idx, device=device)
                emb_map[mask] = emb

            emb_map = torch.permute(emb_map, (2, 0, 1))
            emb_maps.append(emb_map)

        # (B,D,H,W)
        emb_maps = torch.stack(emb_maps)

        return emb_maps

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        #gt_labels = kwargs['gt_labels']  # [tensor([ 1,  2,  4,  5,  6,  8,  9, 10, 14, 17, 18, 19, 20, 23, 29],]
        #gt_masks = kwargs['gt_masks']  # [(15,h,w)]

        # emb_map = self.embed_seg_masks(gt_semantic_seg)

        # with torch.no_grad():
        #     print("Skip backbone grads!")
        x = self.extract_feat(img)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg,
                                                      **kwargs)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses