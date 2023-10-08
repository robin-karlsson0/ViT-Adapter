# Ref: https://github.com/concept-fusion/concept-fusion/blob/main/examples/extract_conceptfusion_features.py

import argparse
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from mmseg.core.evaluation.metrics import total_area_to_metrics
from prettytable import PrettyTable
from scipy.ndimage import zoom
from tqdm import tqdm

from tools.baseline_eval.datasets.load_dataset import load_dataset
from tools.baseline_eval.models.concept_fusion_model import ConceptFusionModel
from tools.baseline_eval.models.lseg_model import LSegModel
from tools.baseline_eval.models.rp_clip_model import RegionProposalCLIPModel
from tools.baseline_eval.suff_sim_thresh import (
    CLS_GROUPS, comp_logreg_decision_point, comp_sim,
    intersect_and_union_tresh, print_sim_treshs, save_thresh_dict)
from tools.convert_datasets.txt2idx_star import load_register

IGNORE_IDX = np.iinfo(np.uint32).max


def print_miou_results(total_area_intersect, total_area_union,
                       total_area_pred_label, total_area_label):
    '''
    Print an mIoU table
    '''
    ret_metrics = total_area_to_metrics(total_area_intersect, total_area_union,
                                        total_area_pred_label,
                                        total_area_label)

    # summary table
    ret_metrics_summary = OrderedDict({
        ret_metric:
        np.round(np.nanmean(ret_metric_value) * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })

    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric:
        np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': cls_txts})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    print('\n' + class_table_data.get_string(), )
    print('Summary:')
    print('\n' + summary_table_data.get_string())


def viz_predictions(pred_embs: torch.tensor, cls_txts: list,
                    cls_txt2cls_idx: dict, cls_embs: torch.tensor):
    '''
        '''
    import matplotlib.pyplot as plt
    import PIL.Image as Image
    num_preds = len(cls_txts)

    img = Image.open(
        '/home/robin/datasets/concat_coco_cseg_weighted/imgs/viz/0001081.jpg')
    h, w = pred_embs.shape[1:]
    img = img.resize((w, h))

    # Predict partition
    cls_embs_subset = []
    for cls_txt in cls_txts:
        if cls_txt == 'other':
            cls_emb = model.conv_txt2emb(cls_txt)
            cls_emb /= torch.norm(cls_emb)
            cls_emb = cls_emb[0]
        else:
            cls_idx = cls_txt2cls_idx[cls_txt]
            cls_emb = cls_embs[cls_idx]
        cls_embs_subset.append(cls_emb)
    cls_embs_subset = torch.stack(cls_embs_subset)

    pred_embs = torch.tensor(pred_embs).unsqueeze(0)
    pred_logits = F.conv2d(pred_embs, cls_embs_subset[:, :, None, None])
    pred_probs = F.softmax(pred_logits, dim=1)
    pred_seg = pred_probs.argmax(dim=1)
    pred_seg = pred_seg[0].numpy()  # (H,W)

    for idx, cls_txt in enumerate(cls_txts):
        mask = pred_seg == idx

        plt.subplot(1, num_preds, idx + 1)
        plt.imshow(img)
        plt.imshow(mask, alpha=0.5)
        plt.title(cls_txt)

    plt.show()


def intersect_and_union(pred_embs: np.array,
                        label: np.array,
                        num_classes: int,
                        ignore_index: int,
                        cls_embs: dict,
                        idx_star2cls_idx: dict,
                        hierarchical: bool = False) -> tuple:
    """Calculate intersection and Union.

    Args:
        pred_embs: Predicted embedding map (D, H, W).
        label: Ground truth segmentation idx map (H, W).
        num_classes: Number of categories.
        ignore_index: Index that will be ignored in evaluation.

    Returns:
        torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes (K).
        torch.Tensor: The union of prediction and ground truth histogram on
            all classes (K).
        torch.Tensor: The prediction histogram on all classes (K).
        torch.Tensor: The ground truth histogram on all classes (K).
    """
    pred_h, pred_w = pred_embs.shape[1:]
    label_h, label_w = label.shape
    scale_h = pred_h / label_h
    scale_w = pred_w / label_w
    if scale_h != 1. and scale_w != 1.:
        label = zoom(label, (scale_h, scale_w), order=0)

    # ['couch', 'furniture', 'other']
    # ['couch', 'other']
    # ['furniture', 'other']
    # viz_predictions(pred_embs, ['furniture', 'other'],
    #                 cls_txt2cls_idx, cls_embs)
    # exit()

    # Transform semantics --> label probability --> seg map (H,W)
    pred_embs = torch.tensor(pred_embs).unsqueeze(0)
    pred_logits = F.conv2d(pred_embs, cls_embs[:, :, None, None])
    pred_probs = F.softmax(pred_logits, dim=1)
    pred_seg = pred_probs.argmax(dim=1)
    pred_seg = pred_seg[0].numpy()  # (H,W)

    # Convert label 'idx*' map --> 'class idx' map
    idx_stars = list(np.unique(label))

    area_intersect_sum = np.zeros(num_classes)
    area_union_sum = np.zeros(num_classes)
    area_pred_label_sum = np.zeros(num_classes)
    area_label_sum = np.zeros(num_classes)

    # Create a new label map with 'cls' idxs filled according to label
    label_h, label_w = label.shape
    label_clss = np.zeros((num_classes, label_h, label_w))
    for idx_star in idx_stars:
        if idx_star not in idx_star2cls_idx.keys():
            continue
        mask = label == idx_star
        cls_idx = idx_star2cls_idx[idx_star]

        # Add higher-level semantics to label
        if hierarchical:
            cls_txt = cls_txts[cls_idx]
            cls_group_txts = CLS_GROUPS[cls_txt]
            for cls_txt in cls_group_txts:

                # Boolean annotation mask (H, W) for current category
                cls_idx = cls_txt2cls_idx[cls_txt]
                label_clss[cls_idx][mask] = True

        # Lower-level semantics only
        else:
            label_clss[cls_idx][mask] = True

    for cls_idx in range(num_classes):

        pred_seg_cls = pred_seg == cls_idx

        # NOTE Need to remove 'ignore' idx from mask
        valid_mask = (label != np.iinfo(np.uint32).max)
        pred_seg_cls = pred_seg_cls[valid_mask]
        label_cls = label_clss[cls_idx][valid_mask]

        # Compute intersection and union by #elements
        area_intersect = np.logical_and(pred_seg_cls, label_cls)
        area_union = np.logical_or(pred_seg_cls, label_cls)

        area_intersect = np.sum(area_intersect)
        area_union = np.sum(area_union)
        area_pred_label = np.sum(pred_seg_cls)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate the ConceptFusion OV semseg model')
    parser.add_argument('dataset_type', type=str, help='\{mapillary|TODO\}')
    parser.add_argument('dataset_path',
                        type=str,
                        help='Path to dataset root dir.')
    parser.add_argument('txt2idx_star_path',
                        type=str,
                        help='Path to txt --> idx_star dict pickle file.')
    parser.add_argument('idx_star2emb_path',
                        type=str,
                        help='Path to idx_star --> emb dict pickle file.')
    parser.add_argument('dataset_split', type=str, help='train, val, etc.')
    parser.add_argument('model_type',
                        type=str,
                        help='\{rp_clip | concept_fusion | lseg\}')
    parser.add_argument('ckpt_path',
                        type=str,
                        default=None,
                        help='Path to trained model checkpoint.')
    parser.add_argument('--rp_model_type',
                        type=str,
                        default=None,
                        help='\{vit_h|TODO\}')
    parser.add_argument('--img_target_size', type=int, default=1024)
    parser.add_argument('--eval_type',
                        type=str,
                        default='most_sim',
                        help='\{most_sim | suff_sim\}')
    parser.add_argument('--sim_thresh_dict_path', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    How to use

    (1) Most similar evaluation
        --eval_type most_sim
    
    (2) Sufficient similarity evaluation
        --eval_type suff_sim
        Optionally
        --sim_thresh_dict_path <path to precomputed sim_thresh dict>

    '''
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.autograd.set_grad_enabled(False)

    #############
    #  Dataset
    #############
    out = load_dataset(args.dataset_type, args.dataset_path,
                       args.dataset_split, args.img_target_size)
    dataset, cls_txts, rgbs = out

    num_clss = len(cls_txts)

    # For evaluating overlapping high-level semantics
    if args.dataset_type in ['coco_cseg']:
        hierarchical = True
    else:
        hierarchical = False

    #########################################
    #  Annotation --> embedding conversion
    #########################################
    #  Converts class text descriptions into an embedding tensor with row
    #  vectors corresponding to class order.
    #
    #  Result:
    #    txt2idx_star[cls] --> idx
    #    idx_star_emb[idx] --> emb
    #    idx_star2cls_idx[idx] --> cls

    txt2idx_star = load_register(args.txt2idx_star_path)
    idx_star2emb = load_register(args.idx_star2emb_path)

    cls_txt2cls_idx = {}
    for cls_idx, cls_txt in enumerate(cls_txts):
        cls_txt2cls_idx[cls_txt] = cls_idx

    # Normalize embedding vectors
    idx_star2emb = {key: F.normalize(val) for key, val in idx_star2emb.items()}
    valid_idxs = set(idx_star2emb.keys())

    # Crate tensor evaluation class VL embeddings
    cls_embs = []
    for cls_txt in cls_txts:
        idx = txt2idx_star[cls_txt]
        emb = idx_star2emb[idx]
        cls_embs.append(emb)
    cls_embs = torch.cat(cls_embs)  # (19, D)

    # Dict for converting labels from 'idx*' maps --> 'class idx' maps
    idx_star2cls_idx = {}
    for cls_idx, cls_txt in enumerate(cls_txts):
        idx_star = txt2idx_star[cls_txt]
        idx_star2cls_idx[idx_star] = cls_idx

    ################
    #  Load model
    ################
    if args.model_type == 'rp_clip':
        model = RegionProposalCLIPModel(args.rp_model_type, args.ckpt_path,
                                        device)
    elif args.model_type == 'concept_fusion':
        model = ConceptFusionModel(args.rp_model_type, args.ckpt_path, device)
    elif args.model_type == 'lseg':
        model = LSegModel(args.ckpt_path, device)

        # Compute txt embs with CLIP used for LSeg
        cls_embs = []
        for cls_txt in cls_txts:
            emb = model.conv_txt2emb(cls_txt)
            emb /= torch.norm(emb)
            cls_embs.append(emb)
        cls_embs = torch.cat(cls_embs)  # (19, D)

    else:
        raise IOError(f'Model type not implemented ({args.model_type})')

    ################
    #  Evaluation
    ################
    total_area_intersect = torch.zeros((num_clss))
    total_area_union = torch.zeros((num_clss))
    total_area_pred_label = torch.zeros((num_clss))
    total_area_label = torch.zeros((num_clss))

    ##############################################
    #  Compute sufficient similarity thresholds
    ##############################################
    if args.eval_type == 'suff_sim' and args.sim_thresh_dict_path is None:

        K = len(cls_txts)
        sim_poss = [[] for _ in range(K)]
        sim_negs = [[] for _ in range(K)]

        for sample_idx in tqdm(range(len(dataset))):

            img, label = dataset[sample_idx]
            with torch.no_grad():
                emb_map = model.forward(img)
            emb_map = emb_map.cpu().numpy()

            sim_pos, sim_neg = comp_sim(emb_map, label, cls_embs,
                                        idx_star2cls_idx)

            for k in range(K):
                sim_poss[k].extend(sim_pos[k])
                sim_negs[k].extend(sim_neg[k])

        # Compute thresholds as optimal decision boundary points
        sim_threshs = [None] * K
        for k in range(K):
            sim_pos = sim_poss[k]
            sim_neg = sim_negs[k]
            if len(sim_pos) > 0 and len(sim_neg) > 0:
                dec_b = comp_logreg_decision_point(sim_pos, sim_neg)
                sim_threshs[k] = dec_b
        # Clip similarity thresholds
        sim_threshs = [
            min(1, max(-1, s)) if s is not None else s for s in sim_threshs
        ]
        print_sim_treshs(sim_threshs, cls_txts, sim_poss, sim_negs)
        save_thresh_dict(sim_threshs, cls_txts)

    # Load precomputed similarity threshold values from a .pkl file
    if args.eval_type == 'suff_sim' and args.sim_thresh_dict_path:
        with open(args.sim_thresh_dict_path, 'rb') as f:
            sim_thresh_dict = pickle.load(f)
        sim_threshs = []
        for cls_txt in cls_txts:
            sim_thresh = sim_thresh_dict[cls_txt]
            sim_threshs.append(sim_thresh)

    for sample_idx in tqdm(range(len(dataset))):
        img, label = dataset[sample_idx]

        emb_map = model.forward(img)
        emb_map = emb_map.cpu().numpy()

        if args.eval_type == "most_sim":
            out = intersect_and_union(emb_map, label, num_clss, IGNORE_IDX,
                                      cls_embs, idx_star2cls_idx, hierarchical)
            area_intersect, area_union, area_pred_label, area_label = out
        elif args.eval_type == 'suff_sim':
            K = len(cls_txts)
            out = intersect_and_union_tresh(emb_map, label, cls_embs, cls_txts,
                                            idx_star2cls_idx, cls_txt2cls_idx,
                                            sim_threshs, K, hierarchical)
            area_intersect, area_union, area_pred_label, area_label = out

        # Add sample results to accumulated counts
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label

        if (sample_idx + 1) % 100 == 0:
            print(f'\nIntermediate results ({sample_idx} / {len(dataset)})')
            print_miou_results(total_area_intersect, total_area_union,
                               total_area_pred_label, total_area_label)

    print(f'\Evaluation results')
    print_miou_results(total_area_intersect, total_area_union,
                       total_area_pred_label, total_area_label)
