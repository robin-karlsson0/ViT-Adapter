import datetime
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from sklearn.linear_model import LogisticRegression

# Special hierarchical taxonomy of COCO-Stuff
CLS_GROUPS = {
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


def intersect_and_union_tresh(pred_embs,
                              label,
                              cls_embs,
                              cls_txts,
                              idx_star2cls_idx,
                              cls_txt2cls_idx,
                              sim_treshs: np.array,
                              num_classes,
                              hierarchical: bool = False,
                              label_map=None):
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
    pred_sims = F.conv2d(pred_embs, cls_embs[:, :, None, None])
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
    label_clss = np.zeros((len(cls_txts), label_h, label_w))

    # Set of class idx for all annotated and hierarchical semantics
    cls_idxs = set()

    for idx_star in idx_stars:
        # Only process valid categories
        if idx_star not in idx_star2cls_idx.keys():
            continue

        # This mask is true for all members of the group
        mask = label == idx_star

        # Fill mask region for all semantic levels
        cls_idx = idx_star2cls_idx[idx_star]

        # Add higher-level semantics to label
        if hierarchical:
            cls_txt = cls_txts[cls_idx]
            cls_group_txts = CLS_GROUPS[cls_txt]
            for cls_txt in cls_group_txts:

                # Boolean annotation mask (H, W) for current category
                cls_idx = cls_txt2cls_idx[cls_txt]
                label_clss[cls_idx][mask] = True

                cls_idxs.add(cls_idx)

        # Lower-level semantics only
        else:
            label_clss[cls_idx][mask] = True

    #####################################
    #  Evaluate high-level predictions
    #####################################
    # for cls_idx in cls_idxs:
    for cls_idx in range(len(cls_txts)):

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


def comp_sim(pred_emb: np.array,
             seg_map: np.array,
             cls_embs: np.array,
             idx_star2cls_idx,
             max_count=int(1e5)) -> tuple:
    '''
    Args:
        emb_map: (D, H, W)
    '''
    # Sample-wise counts
    K = cls_embs.shape[0]
    sim_poss = [[] for _ in range(K)]
    sim_negs = [[] for _ in range(K)]

    # Resize annotation to output prediction size
    pred_h, pred_w = pred_emb.shape[1:]
    seg_map_h, seg_map_w = seg_map.shape
    scale_h = pred_h / seg_map_h
    scale_w = pred_w / seg_map_w
    if scale_h != 1. and scale_w != 1.:
        seg_map = zoom(seg_map, (scale_h, scale_w), order=0)

    pred_emb = torch.tensor(pred_emb).unsqueeze(0)
    pred_sims = F.conv2d(pred_emb, cls_embs[:, :, None, None])

    # Convert label 'idx*' map --> 'class idx' map
    idx_stars = list(np.unique(seg_map))

    for idx_star in idx_stars:
        if idx_star not in idx_star2cls_idx.keys():
            continue
        cls_idx = idx_star2cls_idx[idx_star]

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


def comp_logreg_decision_point(sim_pos: np.array,
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
    model = LogisticRegression(solver='liblinear', class_weight=class_weight)
    model.fit(X, y)

    # Decision boundary: w * x + b = 0.5
    coef = model.coef_[0][0]
    intercept = model.intercept_[0]
    x_boundary = (0.5 - intercept) / coef

    return x_boundary


def print_sim_treshs(sim_threshs: list, txts: list, sim_poss: list,
                     sim_negs: list):
    """
    Args:
        sim_threshs: List of similarity thresholds for each category.
        txts: List of category descriptions.
        sim_poss: List of similarity values for true elements.
        sim_negs: List of similarity values for false elements.
    """
    print('\nSimilarity thresholds (idx, txt, thresh, correct ratio pos|neg,'
          'num pos|neg)')
    entries = []
    for idx, (txt, sim) in enumerate(zip(txts, sim_threshs)):
        if len(sim_poss[idx]) == 0 and len(sim_negs[idx]) == 0:
            continue

        sim_pos = np.array(sim_poss[idx])
        sim_neg = np.array(sim_negs[idx])
        num_pos = len(sim_pos)
        num_neg = len(sim_neg)
        ratio_true = np.sum(sim_pos > sim) / num_pos if num_pos > 0 else None
        ratio_false = np.sum(sim_neg < sim) / num_neg if num_neg > 0 else None

        entry = {
            'txt': [txt],
            'sim': [sim],
            'ratio_true': [ratio_true],
            'ratio_false': [ratio_false],
            'num_pos': [num_pos],
            'num_false': [num_neg]
        }
        entries.append(entry)

    # Merge entries into data dictionary
    data = {}
    for d in entries:
        for k, v in d.items():
            if k in data:
                data[k] += v
            else:
                data[k] = v

    df = pd.DataFrame(data)
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):
        print(df)


def save_thresh_dict(sim_threshs: list, txts: list):
    sim_thresh_dict = {}
    for sim_thresh, txt in zip(sim_threshs, txts):
        sim_thresh_dict[txt] = sim_thresh

    now = datetime.datetime.now()
    ts = now.strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f'sim_threshs_{ts}.pkl'
    with open(file_name, "wb") as f:
        pickle.dump(sim_thresh_dict, f)

    print(f'sim_threshs to file: {file_name}')
