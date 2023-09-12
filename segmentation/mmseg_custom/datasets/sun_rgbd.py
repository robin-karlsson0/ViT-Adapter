import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

from mmseg_custom.datasets.builder import DATASETS
from mmseg_custom.datasets.custom import CustomDataset
from tools.convert_datasets.txt2idx_star import load_register


@DATASETS.register_module()
class SUNRGBDVLDataset(CustomDataset):
    """
    """

    CLASSES = ('wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
               'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds',
               'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
               'floor_mat', 'clothes', 'ceiling', 'books', 'fridge', 'tv',
               'paper', 'towel', 'shower_curtain', 'box', 'whiteboard',
               'person', 'night_stand', 'toilet', 'sink', 'lamp', 'bathtub',
               'bag')

    PALETTE = (
        [0, 192, 64],
        [0, 192, 64],
        [0, 64, 96],
        [128, 192, 192],
        [0, 64, 64],
        [0, 192, 224],
        [0, 192, 192],
        [128, 192, 64],
        [0, 192, 96],
        [128, 192, 64],
        [128, 32, 192],
        [0, 0, 224],
        [0, 0, 64],
        [0, 160, 192],
        [128, 0, 96],
        [128, 0, 192],
        [0, 32, 192],
        [128, 128, 224],
        [0, 0, 192],
        [128, 160, 192],
        [128, 128, 0],
        [128, 0, 32],
        [128, 32, 0],
        [128, 0, 128],
        [64, 128, 32],
        [0, 160, 0],
        [0, 0, 0],
        [192, 128, 160],
        [0, 32, 0],
        [0, 128, 128],
        [64, 128, 160],
        [128, 160, 0],
        [0, 128, 0],
        [192, 128, 32],
        [128, 96, 128],
        [0, 0, 128],
        [64, 0, 32],
    )

    def __init__(self,
                 txt2idx_star_path='./txt2idx_star.pkl',
                 idx_star2emb_path='./idx_star2emb.pkl',
                 **kwargs):
        super(SUNRGBDVLDataset,
              self).__init__(img_suffix='.jpg',
                             seg_map_suffix='_vl_emb_idxs.npz',
                             reduce_zero_label=False,
                             **kwargs)
        self.txt2idx_star = load_register(txt2idx_star_path)

        self.idx_star2emb = load_register(idx_star2emb_path)
        # Normalize embedding vectors
        self.idx_star2emb = {
            key: val / np.linalg.norm(val)
            for key, val in self.idx_star2emb.items()
        }
        self.valid_idxs = set(self.idx_star2emb.keys())

        self.cls_embs = []
        for cls_txt in self.CLASSES:
            idx = self.txt2idx_star[cls_txt]
            emb = self.idx_star2emb[idx]
            self.cls_embs.append(emb)
        self.cls_embs = torch.cat(self.cls_embs)  # (19, D)

        # Dict for converting labels from 'idx*' maps --> 'class idx' maps
        self.idx_star2cls_idx = {}
        for cls_idx, cls_txt in enumerate(self.CLASSES):
            idx_star = self.txt2idx_star[cls_txt]
            self.idx_star2cls_idx[idx_star] = cls_idx

    @staticmethod
    def compute_iou(pred: np.array, label: np.array):
        '''
        Args:
            pred: Boolean mask (B,1,H,W).
            label: 

        Returns:
            Array of IoU values (B).
        '''
        intersection = np.logical_and(pred, label).sum((1, 2, 3))
        union = np.logical_or(pred, label).sum((1, 2, 3))
        iou = intersection / (union + np.finfo(float).eps)

        return iou

    def intersect_and_union(self,
                            pred_embs,
                            label,
                            num_classes,
                            label_map=dict()):
        """Calculate intersection and Union.

        Args:
            pred_embs (ndarray | str): Predicted embedding map (D, H, W).
            label (ndarray | str): Ground truth segmentation idx map (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.
            label_map (dict): Mapping old labels to new labels. The parameter will
                work only when label is str. Default: dict().

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """
        if label_map is not None:
            for old_id, new_id in label_map.items():
                label[label == old_id] = new_id

        pred_h, pred_w = pred_embs.shape[1:]
        label_h, label_w = label.shape
        scale_h = pred_h / label_h
        scale_w = pred_w / label_w
        if scale_h != 1. and scale_w != 1.:
            label = zoom(label, (scale_h, scale_w), order=0)

        # Transform semantics --> label probability --> seg map (H,W)
        pred_embs = torch.tensor(pred_embs).unsqueeze(0)
        pred_logits = F.conv2d(pred_embs, self.cls_embs[:, :, None, None])
        pred_probs = F.softmax(pred_logits, dim=1)
        pred_seg = pred_probs.argmax(dim=1)
        pred_seg = pred_seg[0].numpy()  # (H,W)

        # Convert label 'idx*' map --> 'class idx' map
        idx_stars = list(np.unique(label))

        # Create a new label map with 'cls' idxs including 'ignore' cls (255)
        label_cls = np.ones(label.shape, dtype=int)
        label_cls *= self.ignore_index
        for idx_star in idx_stars:
            if idx_star not in self.idx_star2cls_idx.keys():
                continue
            mask = label == idx_star
            label_cls[mask] = self.idx_star2cls_idx[idx_star]

        # pred_seg: torch.tensor int (H, W)
        # label_cls: torch.tensor int (H, W)
        # NOTE Need to remove 'ignore' idx* from mask
        valid_mask = (label != np.iinfo(np.uint32).max)
        pred_seg = pred_seg[valid_mask]
        label_cls = label_cls[valid_mask]

        pred_seg = torch.tensor(pred_seg)
        label_cls = torch.tensor(label_cls)

        # Extracts matching elements with class idx
        intersect = pred_seg[pred_seg == label_cls]
        # Sums up elements by class idx
        area_intersect = torch.histc(intersect.float(),
                                     bins=(num_classes),
                                     min=0,
                                     max=num_classes - 1)
        area_pred_label = torch.histc(pred_seg.float(),
                                      bins=(num_classes),
                                      min=0,
                                      max=num_classes - 1)
        area_label = torch.histc(label_cls.float(),
                                 bins=(num_classes),
                                 min=0,
                                 max=num_classes - 1)
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    def pre_eval(self, preds, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            seg_map = self.get_gt_seg_map_by_idx(index)
            pre_eval_results.append(
                self.intersect_and_union(pred, seg_map, len(self.CLASSES),
                                         self.label_map))

        return pre_eval_results