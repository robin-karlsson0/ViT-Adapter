import os.path as osp

from mmseg_custom.datasets.builder import DATASETS
from mmseg_custom.datasets.custom import CustomDataset
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom

from tools.convert_datasets.txt2idx_star import load_register


@DATASETS.register_module()
class MapillaryVistasV1_2VLDataset(CustomDataset):
    """"""

    CLASSES = ('Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail',
               'Barrier', 'Wall', 'Bike Lane', 'Crosswalk - Plain', 'Curb Cut',
               'Parking', 'Pedestrian Area', 'Rail Track', 'Road',
               'Service Lane', 'Sidewalk', 'Bridge', 'Building', 'Tunnel',
               'Person', 'Bicyclist', 'Motorcyclist', 'Other Rider',
               'Lane Marking - Crosswalk', 'Lane Marking - General',
               'Mountain', 'Sand', 'Sky', 'Snow', 'Terrain', 'Vegetation',
               'Water', 'Banner', 'Bench', 'Bike Rack', 'Billboard',
               'Catch Basin', 'CCTV Camera', 'Fire Hydrant', 'Junction Box',
               'Mailbox', 'Manhole', 'Phone Booth', 'Pothole', 'Street Light',
               'Pole', 'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
               'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can',
               'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle',
               'On Rails', 'Other Vehicle', 'Trailer', 'Truck', 'Wheeled Slow',
               'Car Mount', 'Ego Vehicle', 'Unlabeled')

    PALETTE = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
               [180, 165, 180], [90, 120, 150], [102, 102,
                                                 156], [128, 64, 255],
               [140, 140, 200], [170, 170, 170], [250, 170, 160], [96, 96, 96],
               [230, 150, 140], [128, 64, 128], [110, 110,
                                                 110], [244, 35, 232],
               [150, 100, 100], [70, 70, 70], [150, 120, 90], [220, 20, 60],
               [255, 0, 0], [255, 0, 100], [255, 0, 200], [200, 128, 128],
               [255, 255, 255], [64, 170, 64], [230, 160, 50], [70, 130, 180],
               [190, 255, 255], [152, 251, 152], [107, 142, 35], [0, 170, 30],
               [255, 255, 128], [250, 0, 30], [100, 140, 180], [220, 220, 220],
               [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
               [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
               [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
               [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
               [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142],
               [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64,
                                                       64], [0, 0, 110],
               [0, 0, 70], [0, 0, 192], [32, 32, 32], [120, 10, 10], [0, 0, 0]]

    def __init__(self,
                 txt2idx_star_path='./txt2idx_star.pkl',
                 idx_star2emb_path='./idx_star2emb.pkl',
                 **kwargs):
        super(MapillaryVistasV1_2VLDataset,
              self).__init__(img_suffix='.jpg',
                             seg_map_suffix='_v1_2_vl_emb_idxs.npz',
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
        self.cls_embs = torch.concat(self.cls_embs)  # (19, D)

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
