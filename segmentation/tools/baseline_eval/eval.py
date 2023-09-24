# Ref: https://github.com/concept-fusion/concept-fusion/blob/main/examples/extract_conceptfusion_features.py

import argparse
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


def intersect_and_union(pred_embs: np.array, label: np.array, num_classes: int,
                        ignore_index: int, cls_embs: dict,
                        idx_star2cls_idx: dict) -> tuple:
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

    # Transform semantics --> label probability --> seg map (H,W)
    pred_embs = torch.tensor(pred_embs).unsqueeze(0)
    pred_logits = F.conv2d(pred_embs, cls_embs[:, :, None, None])
    pred_probs = F.softmax(pred_logits, dim=1)
    pred_seg = pred_probs.argmax(dim=1)
    pred_seg = pred_seg[0].numpy()  # (H,W)

    # Convert label 'idx*' map --> 'class idx' map
    idx_stars = list(np.unique(label))

    # Create a new label map with 'cls' idxs including 'ignore' cls (255)
    label_cls = np.ones(label.shape, dtype=int)
    label_cls *= ignore_index
    for idx_star in idx_stars:
        if idx_star not in idx_star2cls_idx.keys():
            continue
        mask = label == idx_star
        label_cls[mask] = idx_star2cls_idx[idx_star]

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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
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

    for sample_idx in tqdm(range(len(dataset))):
        img, label = dataset[sample_idx]

        emb_map = model.forward(img)
        emb_map = emb_map.cpu().numpy()

        out = intersect_and_union(emb_map, label, num_clss, IGNORE_IDX,
                                  cls_embs, idx_star2cls_idx)
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
