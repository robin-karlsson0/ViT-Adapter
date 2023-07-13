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
        # self.backbone.eval()

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

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """

        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = torch.zeros((batch_size, num_classes, h_img, w_img),
                            dtype=torch.float32,
                            device='cpu')
        count_mat = torch.zeros((batch_size, 1, h_img, w_img),
                                dtype=torch.float32,
                                device='cpu')
        # preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        # count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                crop_seg_logit = crop_seg_logit.cpu()
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(preds,
                           size=img_meta[0]['ori_shape'][:2],
                           mode='bilinear',
                           align_corners=self.align_corners,
                           warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        raise NotImplementedError

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: Normalized embeding map (B, D, H, W).
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        # output = F.softmax(seg_logit, dim=1)
        output = F.normalize(seg_logit)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        emb_map = self.inference(img, img_meta, rescale)

        # seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            emb_map = emb_map.unsqueeze(0)
            return emb_map
        emb_map = emb_map.cpu().numpy()
        # unravel batch dim
        emb_map = list(emb_map)
        return emb_map