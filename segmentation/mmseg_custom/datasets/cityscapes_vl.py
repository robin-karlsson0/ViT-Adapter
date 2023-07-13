from mmseg_custom.datasets.builder import DATASETS
from mmseg_custom.datasets.custom import CustomDataset
import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger
import torch
import numpy as np

from tools.convert_datasets.txt2idx_star import load_register


@DATASETS.register_module()
class CityscapesVLDataset(CustomDataset):
    """Cityscapes vision-language dataset.
    """
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self,
                 txt2idx_star_path='./txt2idx_star.pkl',
                 idx_star2emb_path='./idx_star2emb.pkl',
                 **kwargs):
        super(CityscapesVLDataset,
              self).__init__(img_suffix='_leftImg8bit.png',
                             seg_map_suffix='_gtFine_vl_emb_idxs.npz',
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

        self.cls2emb = {}
        for cls_txt in self.CLASSES:
            idx = self.txt2idx_star[cls_txt]
            emb = self.idx_star2emb[idx]
            self.cls2emb[cls_txt] = emb

        # emb = list(self.idx_star2emb.values())[0]
        # self.emb_dim = emb.shape[1]

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
                            pred_label,
                            label,
                            num_classes,
                            label_map=dict(),
                            pred_tresh=0.9):
        """Calculate intersection and Union.

        Args:
            pred_label (ndarray | str): Prediction segmentation map
                or predict result filename.
            label (ndarray | str): Ground truth segmentation map
                or label filename.
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.
            label_map (dict): Mapping old labels to new labels. The parameter will
                work only when label is str. Default: dict().
            reduce_zero_label (bool): Whether ignore zero label. The parameter will
                work only when label is str. Default: False.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        if isinstance(pred_label, str):
            pred_label = np.load(pred_label)

        if isinstance(label, str):
            label = mmcv.imread(label, flag='unchanged', backend='pillow')
        else:
            label = label.astype(np.float32)

        if label_map is not None:
            for old_id, new_id in label_map.items():
                label[label == old_id] = new_id

        # Convert label idx --> emb and threshold by similarity

        # Valid label indices
        label_idxs = set(list(np.unique(label)))
        label_idxs = label_idxs.intersection(self.valid_idxs)

        # Finish the evaluation code
        raise NotImplementedError

        # miou = []
        for cls_txt in self.CLASSES:
            idx = self.txt2idx_star[cls_txt]
            if idx not in label_idxs:
                continue
            emb = self.cls2emb[cls_txt]

            sim = np.einsum('dhw,d->hw', pred_label, emb)
            pred_mask = sim > pred_tresh

            intersection = np.logical_and(pred_mask, label).sum((1, 2, 3))

            # union = np.logical_or(pred_mask, label).sum((1, 2, 3))

            # iou = self.compute_iou(pred_mask, label)
            # miou.append(iou)

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(intersect.float(),
                                     bins=(num_classes),
                                     min=0,
                                     max=num_classes - 1)
        area_pred_label = torch.histc(pred_label.float(),
                                      bins=(num_classes),
                                      min=0,
                                      max=num_classes - 1)
        area_label = torch.histc(label.float(),
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
