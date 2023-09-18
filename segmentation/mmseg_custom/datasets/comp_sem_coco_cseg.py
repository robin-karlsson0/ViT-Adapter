import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from sklearn.linear_model import LogisticRegression

from mmseg_custom.datasets.builder import DATASETS
from mmseg_custom.datasets.custom import CustomDataset
from tools.convert_datasets.txt2idx_star import load_register


@DATASETS.register_module()
class CompSemCOCOCsegDataset(CustomDataset):
    """
    """

    CLASSES = []

    # Original
    CLASSES += [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat',
        'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window',
        'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
        'teddy bear', 'hair drier', 'toothbrush', 'hair brush', 'banner',
        'blanket', 'branch', 'bridge', 'building', 'bush', 'cabinet', 'cage',
        'cardboard', 'carpet', 'ceiling', 'tile ceiling', 'cloth', 'clothes',
        'clouds', 'counter', 'cupboard', 'curtain', 'dirt', 'fence',
        'marble floor', 'floor', 'stone floor', 'tile floor', 'wood floor',
        'flower', 'fog', 'food', 'fruit', 'furniture', 'grass', 'gravel',
        'ground', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'moss',
        'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow',
        'plant', 'plastic', 'platform', 'playingfield', 'railing', 'railroad',
        'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand', 'sea',
        'shelf', 'sky', 'skyscraper', 'snow', 'solid', 'stairs', 'stone',
        'straw', 'structural', 'table', 'tent', 'textile', 'towel', 'tree',
        'vegetable', 'brick wall', 'concrete wall', 'wall', 'panel wall',
        'stone wall', 'tile wall', 'wood wall', 'water', 'waterdrops',
        'blind window', 'wood', 'vehicle', 'outdoor', 'animal', 'accessory',
        'sports', 'kitchen', 'electronic', 'appliance', 'indoor',
        'raw material'
    ]

    PALETTE = []
    for cls_idx in range(len(CLASSES)):
        rgb = np.random.randint(0, 256, size=3)
        rgb = list(rgb)
        PALETTE.append(rgb)
    PALETTE = tuple(PALETTE)

    def __init__(self,
                 txt2idx_star_path='./txt2idx_star_cseg_coco.pkl',
                 idx_star2emb_path='./idx_star2emb_cseg_coco.pkl',
                 **kwargs):
        super(CompSemCOCOCsegDataset, self).__init__(img_suffix='.jpg',
                                                     seg_map_suffix='.npz',
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

        # cls_groups[class txt] --> [List of class txts]
        self.cls_groups = {
            'person': ['person', 'outdoor'],
            'bicycle': ['bicycle', 'vehicle', 'outdoor'],
            'car': ['car', 'vehicle', 'outdoor'],
            'motorcycle': ['motorcycle', 'vehicle', 'outdoor'],
            'airplane': ['airplane', 'vehicle', 'outdoor'],
            'bus': ['bus', 'vehicle', 'outdoor'],
            'train': ['train', 'vehicle', 'outdoor'],
            'truck': ['truck', 'vehicle', 'outdoor'],
            'boat': ['boat', 'vehicle', 'outdoor'],
            'traffic light': ['traffic light', 'outdoor'],
            'fire hydrant': ['fire hydrant', 'outdoor'],
            'street sign': ['street sign', 'outdoor'],
            'stop sign': ['stop sign', 'outdoor'],
            'parking meter': ['parking meter', 'outdoor'],
            'bench': ['bench', 'outdoor'],
            'bird': ['bird', 'animal', 'outdoor'],
            'cat': ['cat', 'animal', 'outdoor'],
            'dog': ['dog', 'animal', 'outdoor'],
            'horse': ['horse', 'animal', 'outdoor'],
            'sheep': ['sheep', 'animal', 'outdoor'],
            'cow': ['cow', 'animal', 'outdoor'],
            'elephant': ['elephant', 'animal', 'outdoor'],
            'bear': ['bear', 'animal', 'outdoor'],
            'zebra': ['zebra', 'animal', 'outdoor'],
            'giraffe': ['giraffe', 'animal', 'outdoor'],
            'hat': ['hat', 'accessory', 'outdoor'],
            'backpack': ['backpack', 'accessory', 'outdoor'],
            'umbrella': ['umbrella', 'accessory', 'outdoor'],
            'shoe': ['shoe', 'accessory', 'outdoor'],
            'eye glasses': ['eye glasses', 'accessory', 'outdoor'],
            'handbag': ['handbag', 'accessory', 'outdoor'],
            'tie': ['tie', 'accessory', 'outdoor'],
            'suitcase': ['suitcase', 'accessory', 'outdoor'],
            'frisbee': ['frisbee', 'sports', 'outdoor'],
            'skis': ['skis', 'sports', 'outdoor'],
            'snowboard': ['snowboard', 'sports', 'outdoor'],
            'sports ball': ['sports ball', 'sports', 'outdoor'],
            'kite': ['kite', 'sports', 'outdoor'],
            'baseball bat': ['baseball bat', 'sports', 'outdoor'],
            'baseball glove': ['baseball glove', 'sports', 'outdoor'],
            'skateboard': ['skateboard', 'sports', 'outdoor'],
            'surfboard': ['surfboard', 'sports', 'outdoor'],
            'tennis racket': ['tennis racket', 'sports', 'outdoor'],
            'bottle': ['bottle', 'kitchen', 'indoor'],
            'plate': ['plate', 'kitchen', 'indoor'],
            'wine glass': ['wine glass', 'kitchen', 'indoor'],
            'cup': ['cup', 'kitchen', 'indoor'],
            'fork': ['fork', 'kitchen', 'indoor'],
            'knife': ['knife', 'kitchen', 'indoor'],
            'spoon': ['spoon', 'kitchen', 'indoor'],
            'bowl': ['bowl', 'kitchen', 'indoor'],
            'banana': ['banana', 'food', 'indoor'],
            'apple': ['apple', 'food', 'indoor'],
            'sandwich': ['sandwich', 'food', 'indoor'],
            'orange': ['orange', 'food', 'indoor'],
            'broccoli': ['broccoli', 'food', 'indoor'],
            'carrot': ['carrot', 'food', 'indoor'],
            'hot dog': ['hot dog', 'food', 'indoor'],
            'pizza': ['pizza', 'food', 'indoor'],
            'donut': ['donut', 'food', 'indoor'],
            'cake': ['cake', 'food', 'indoor'],
            'chair': ['chair', 'furniture', 'indoor'],
            'couch': ['couch', 'furniture', 'indoor'],
            'potted plant': ['potted plant', 'furniture', 'indoor'],
            'bed': ['bed', 'furniture', 'indoor'],
            'mirror': ['mirror', 'furniture', 'indoor'],
            'dining table': ['dining table', 'furniture', 'indoor'],
            'window': ['window', 'furniture', 'indoor'],
            'desk': ['desk', 'furniture', 'indoor'],
            'toilet': ['toilet', 'furniture', 'indoor'],
            'door': ['door', 'furniture', 'indoor'],
            'tv': ['tv', 'electronic', 'indoor'],
            'laptop': ['laptop', 'electronic', 'indoor'],
            'mouse': ['mouse', 'electronic', 'indoor'],
            'remote': ['remote', 'electronic', 'indoor'],
            'keyboard': ['keyboard', 'electronic', 'indoor'],
            'cell phone': ['cell phone', 'electronic', 'indoor'],
            'microwave': ['microwave', 'appliance', 'indoor'],
            'oven': ['oven', 'appliance', 'indoor'],
            'toaster': ['toaster', 'appliance', 'indoor'],
            'sink': ['sink', 'appliance', 'indoor'],
            'refrigerator': ['refrigerator', 'appliance', 'indoor'],
            'blender': ['blender', 'appliance', 'indoor'],
            'book': ['book', 'indoor'],
            'clock': ['clock', 'indoor'],
            'vase': ['vase', 'indoor'],
            'scissors': ['scissors', 'indoor'],
            'teddy bear': ['teddy bear', 'indoor'],
            'hair drier': ['hair drier', 'indoor'],
            'toothbrush': ['toothbrush', 'indoor'],
            'hair brush': ['hair brush', 'indoor'],
            'banner': ['banner', 'textile', 'indoor'],
            'blanket': ['blanket', 'textile', 'indoor'],
            'branch': ['branch', 'plant', 'outdoor'],
            'bridge': ['bridge', 'building', 'outdoor'],
            'building': ['building', 'outdoor'],
            'bush': ['bush', 'plant', 'outdoor'],
            'cabinet': ['cabinet', 'furniture', 'indoor'],
            'cage': ['cage', 'structural', 'outdoor'],
            'cardboard': ['cardboard', 'raw material', 'indoor'],
            'carpet': ['carpet', 'textile', 'indoor'],
            'ceiling': ['ceiling', 'building', 'outdoor'],
            'tile ceiling': ['tile ceiling', 'ceiling', 'indoor'],
            'cloth': ['cloth', 'textile', 'indoor'],
            'clothes': ['clothes', 'textile', 'indoor'],
            'clouds': ['clouds', 'sky', 'outdoor'],
            'counter': ['counter', 'furniture', 'indoor'],
            'cupboard': ['cupboard', 'furniture', 'indoor'],
            'curtain': ['curtain', 'textile', 'indoor'],
            'dirt': ['dirt', 'ground', 'outdoor'],
            'fence': ['fence', 'structural', 'outdoor'],
            'marble floor': ['marble floor', 'floor', 'indoor'],
            'floor': ['floor', 'indoor'],
            'stone floor': ['stone floor', 'floor', 'indoor'],
            'tile floor': ['tile floor', 'floor', 'indoor'],
            'wood floor': ['wood floor', 'floor', 'indoor'],
            'flower': ['flower', 'plant', 'outdoor'],
            'fog': ['fog', 'water', 'outdoor'],
            'food': ['food', 'indoor'],
            'fruit': ['fruit', 'food', 'indoor'],
            'furniture': ['furniture', 'indoor'],
            'grass': ['grass', 'plant', 'outdoor'],
            'gravel': ['gravel', 'ground', 'outdoor'],
            'ground': ['ground', 'outdoor'],
            'hill': ['hill', 'solid'],
            'house': ['house', 'building', 'outdoor'],
            'leaves': ['leaves', 'plant', 'outdoor'],
            'light': ['light', 'furniture', 'indoor'],
            'mat': ['mat', 'textile', 'indoor'],
            'metal': ['metal', 'raw material', 'indoor'],
            'moss': ['moss', 'plant', 'outdoor'],
            'mountain': ['mountain', 'solid', 'outdoor'],
            'mud': ['mud', 'ground', 'outdoor'],
            'napkin': ['napkin', 'textile', 'indoor'],
            'net': ['net', 'textile', 'indoor'],
            'paper': ['paper', 'raw material', 'indoor'],
            'pavement': ['pavement', 'ground', 'outdoor'],
            'pillow': ['pillow', 'textile', 'indoor'],
            'plant': ['plant', 'outdoor'],
            'plastic': ['plastic', 'raw material'],
            'platform': ['platform', 'ground', 'outdoor'],
            'playingfield': ['playingfield', 'ground', 'outdoor'],
            'railing': ['railing', 'ground', 'outdoor'],
            'railroad': ['railroad', 'ground', 'outdoor'],
            'river': ['river', 'water', 'outdoor'],
            'road': ['road', 'ground', 'outdoor'],
            'rock': ['rock', 'solid', 'outdoor'],
            'roof': ['roof', 'building', 'outdoor'],
            'rug': ['rug', 'textile', 'indoor'],
            'salad': ['salad', 'food', 'indoor'],
            'sand': ['sand', 'ground', 'outdoor'],
            'sea': ['sea', 'water', 'outdoor'],
            'shelf': ['shelf', 'furniture', 'indoor'],
            'sky': ['sky', 'outdoor'],
            'skyscraper': ['skyscraper', 'building', 'outdoor'],
            'snow': ['snow', 'ground', 'outdoor'],
            'solid': ['solid', 'outdoor'],
            'stairs': ['stairs', 'furniture', 'outdoor'],
            'stone': ['stone', 'solid', 'outdoor'],
            'straw': ['straw', 'plant', 'outdoor'],
            'structural': ['structural', 'outdoor'],
            'table': ['table', 'furniture', 'indoor'],
            'tent': ['tent', 'building', 'outdoor'],
            'textile': ['textile', 'indoor'],
            'towel': ['towel', 'textile', 'indoor'],
            'tree': ['tree', 'plant', 'outdoor'],
            'vegetable': ['vegetable', 'food', 'indoor'],
            'brick wall': ['brick wall', 'wall', 'indoor'],
            'concrete wall': ['concrete wall', 'wall', 'indoor'],
            'wall': ['wall', 'wall', 'indoor'],
            'panel wall': ['panel wall', 'wall', 'indoor'],
            'stone wall': ['stone wall', 'wall', 'indoor'],
            'tile wall': ['tile wall', 'wall', 'indoor'],
            'wood wall': ['wood wall', 'wall', 'indoor'],
            'water': ['water', 'outdoor'],
            'waterdrops': ['waterdrops', 'water', 'outdoor'],
            'blind window': ['blind window', 'window', 'indoor'],
            'wood': ['wood', 'raw material', 'indoor']
        }

        cls_txt2cls_idx = {}
        for cls_idx, cls_txt in enumerate(self.CLASSES):
            cls_txt2cls_idx[cls_txt] = cls_idx
        self.cls_txt2cls_idx = cls_txt2cls_idx

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

    def intersect_and_union_tresh(self,
                                  pred_embs,
                                  label,
                                  sim_treshs: np.array,
                                  num_classes,
                                  label_map=dict()):
        """Calculate IoU by sufficient similarity thresholding.

        Args:
            pred_embs (ndarray | str): Predicted embedding map (D, H, W).
            label (ndarray | str): Ground truth segmentation idx map (H, W).
            sim_treshs: Threshold values for sufficient similarity (K).
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

        # TMP Perfect pred for debug
        # pred_embs_test = np.ones_like(pred_embs)
        # idx_stars = list(np.unique(label))
        # for idx_star in idx_stars:
        #     if idx_star not in self.idx_star2cls_idx.keys():
        #         continue
        #     cls_idx = self.idx_star2cls_idx[idx_star]
        #     emb = self.idx_star2emb[idx_star]
        #     mask = label == idx_star
        #     pred_embs_test[:, mask] = emb.reshape(-1, 1)
        #     sim_treshs[cls_idx] = 0.9
        # pred_embs = pred_embs_test

        # Transform semantics --> label probability --> seg map (H,W)
        pred_embs = torch.tensor(pred_embs).unsqueeze(0)
        pred_sims = F.conv2d(pred_embs, self.cls_embs[:, :, None, None])
        pred_sims = pred_sims[0].numpy()  # (K,H,W)

        # Convert label 'idx*' map --> 'class idx' map
        idx_stars = list(np.unique(label))

        # Evaluate IoU per class without partitioning by sufficient similarity
        # thresholding.
        #
        # Algorithm:
        # for each category:
        #     1) Get boolean annotation mask
        #     2) Get boolean prediction mask
        #        - Sufficiently similar elements = True
        #     3) Extract valid elements from ann and pred masks
        #     4) Compute intersection and union over masks as #elements
        #     5) Store results in category-specific array elements
        area_intersect_sum = np.zeros(num_classes)
        area_union_sum = np.zeros(num_classes)
        area_pred_label_sum = np.zeros(num_classes)
        area_label_sum = np.zeros(num_classes)

        ##############################################
        #  Generate high-level masks and class list
        ##############################################
        label_h, label_w = label.shape
        label_clss = np.zeros((len(self.CLASSES), label_h, label_w))

        # Set of class idx for all annotated and hierarchical semantics
        cls_idxs = set()

        for idx_star in idx_stars:
            # Only process valid categories
            if idx_star not in self.idx_star2cls_idx.keys():
                continue

            # This mask is true for all members of the group
            mask = label == idx_star

            # Fill mask region for all semantic levels
            cls_idx = self.idx_star2cls_idx[idx_star]
            cls_txt = self.CLASSES[cls_idx]
            cls_txts = self.cls_groups[cls_txt]
            for cls_txt in cls_txts:

                # Boolean annotation mask (H, W) for current category
                cls_idx = self.cls_txt2cls_idx[cls_txt]
                label_clss[cls_idx][mask] = True

                cls_idxs.add(cls_idx)

        #####################################
        #  Evaluate high-level predictions
        #####################################
        for cls_idx in cls_idxs:

            # Skip evaluating semantics without a threshold value
            sim_thresh = sim_treshs[cls_idx]
            if sim_thresh is None:
                continue

            # Boolean prediction mask (H, W) by sufficient similarity
            pred_seg = np.zeros_like(label, dtype=bool)
            mask = pred_sims[cls_idx] > sim_thresh
            pred_seg[mask] = True

            # NOTE Need to remove 'ignore' idx from mask
            valid_mask = (label != np.iinfo(np.uint32).max)
            pred_seg = pred_seg[valid_mask]
            label_cls = label_clss[cls_idx][valid_mask]

            # Compute intersection and union by #elements
            area_intersect = np.logical_and(pred_seg, label_cls)
            area_union = np.logical_or(pred_seg, label_cls)

            area_intersect = np.sum(area_intersect)
            area_union = np.sum(area_union)
            area_pred_label = np.sum(pred_seg)
            area_label = np.sum(label_cls)

            # Add result to category-specific array elements
            area_intersect_sum[cls_idx] = area_intersect
            area_union_sum[cls_idx] = area_union
            area_pred_label_sum[cls_idx] = area_pred_label
            area_label_sum[cls_idx] = area_label

        area_intersect_sum = torch.tensor(area_intersect_sum)
        area_union_sum = torch.tensor(area_union_sum)
        area_pred_label_sum = torch.tensor(area_pred_label_sum)
        area_label_sum = torch.tensor(area_label_sum)

        return area_intersect_sum, area_union_sum, area_pred_label_sum, area_label_sum

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

    def pre_eval_thresh(self, preds, sim_treshs: torch.tensor, indices):
        """Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            sim_treshs: Array (K) of sufficient similarity threshold values.
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
                self.intersect_and_union_tresh(pred, seg_map, sim_treshs,
                                               len(self.CLASSES),
                                               self.label_map))

        return pre_eval_results

    def comp_logreg_decision_point(self,
                                   sim_pos: np.array,
                                   sim_neg: np.array,
                                   class_weight: str = 'balanced') -> float:
        """
        NOTE: To optiomize IoU the importance of points need to be balanced
              with relative frequency.

        Args:
            sim_pos: Array (N) of similarity values for positive elements.
            neg_pos:
        
        Returns:
            Optimal "sufficient similarity threshold" scalar value maximizing
            data likelihood.
        """
        # Set up data matrix and label vector
        X = np.concatenate((sim_neg, sim_pos))
        X = np.expand_dims(X, 1)  # (N, 1)
        y = np.concatenate((np.zeros(len(sim_neg)), np.ones(len(sim_pos))))

        # Initialize and train a logistic regression model
        model = LogisticRegression(solver='liblinear',
                                   class_weight=class_weight)
        model.fit(X, y)

        # Decision boundary: w * x + b = 0.5
        coef = model.coef_[0][0]
        intercept = model.intercept_[0]
        x_boundary = (0.5 - intercept) / coef

        return x_boundary

    def comp_sim(self, pred_embs, indices, max_count=int(1e5)):
        """Computes  a list of similarity values for each dataset semantic.

        Args:
            pred_embs (list[torch.Tensor] | torch.Tensor): the segmentation
                logit after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.
        
        Returns:
            sim_pos: List of lists of similarity values for each semantic.
                     Ex: [[0.72, 0.81, ...]_{k=1}, ... ]
            sim_neg: 
        """
        # For compatibility with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(pred_embs, list):
            pred_embs = [pred_embs]

        K = len(self.CLASSES)
        sim_poss = [[] for _ in range(K)]
        sim_negs = [[] for _ in range(K)]

        for pred_emb, index in zip(pred_embs, indices):
            # Resize annotation to output prediction size
            seg_map = self.get_gt_seg_map_by_idx(index)
            pred_h, pred_w = pred_emb.shape[1:]
            seg_map_h, seg_map_w = seg_map.shape
            scale_h = pred_h / seg_map_h
            scale_w = pred_w / seg_map_w
            if scale_h != 1. and scale_w != 1.:
                seg_map = zoom(seg_map, (scale_h, scale_w), order=0)

            pred_emb = torch.tensor(pred_emb).unsqueeze(0)
            pred_sims = F.conv2d(pred_emb, self.cls_embs[:, :, None, None])

            # Convert label 'idx*' map --> 'class idx' map
            idx_stars = list(np.unique(seg_map))

            for idx_star in idx_stars:
                if idx_star not in self.idx_star2cls_idx.keys():
                    continue
                cls_idx = self.idx_star2cls_idx[idx_star]

                mask = seg_map == idx_star

                sim_pos = pred_sims[0, cls_idx][mask]
                sim_neg = pred_sims[0, cls_idx][~mask]

                if len(sim_pos) > max_count:
                    random_idxs = random.sample(range(len(sim_pos)), max_count)
                    sim_pos = np.take(sim_pos, random_idxs)

                if len(sim_neg) > max_count:
                    random_idxs = random.sample(range(len(sim_neg)), max_count)
                    sim_neg = np.take(sim_neg, random_idxs)

                sim_poss[cls_idx].extend(sim_pos.tolist())
                sim_negs[cls_idx].extend(sim_neg.tolist())

        return sim_poss, sim_negs