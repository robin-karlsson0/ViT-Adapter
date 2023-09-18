import argparse
import os
import os.path as osp
from glob import glob

import numpy as np
import pandas as pd

from tools.convert_datasets.txt2idx_star import load_register

NP_TYPE = np.uint32
IDX_STR_LEN = 12


def get_sample_idx_stars(ann_path: str,
                         ignore_idx: int = np.iinfo(NP_TYPE).max):
    ann = np.load(ann_path)
    ann = ann.f.arr_0
    idx_stars = np.unique(ann)
    idx_stars = idx_stars.tolist()
    if ignore_idx in idx_stars:
        idx_stars.remove(ignore_idx)

    return idx_stars


def get_ann_idx(ann_path: str):
    '''
    Presumed path format:
        coco/stuffthingmaps_trainval2017/train2017/000000000139_vl_emb_idxs.npz
    '''
    ann_filename = ann_path.split('/')[-1]
    ann_idx = ann_filename.split('_')[0]
    ann_idx = int(ann_idx)
    return ann_idx


def print_counts(cls_counts: dict):
    data = {key: len(val) for key, val in cls_counts.items()}
    df = pd.DataFrame.from_dict(data, orient='index')
    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None):
        print(df)


def get_rarest_category(cls_counts: dict):
    '''
    Finds the rarest category, samples one sample_idx from it, and removes the
    sample and category if empty.

    NOTE Ensure input dictionary does not contain empty categories.
    '''
    min_cls_counts = np.inf
    min_cls_count_txt = None
    for key, val in cls_counts.items():
        cls_count = len(val)
        if cls_count < min_cls_counts:
            min_cls_counts = cls_count
            min_cls_count_txt = key

    sample_idxs = cls_counts[min_cls_count_txt]
    sample_idx = np.random.choice(sample_idxs)

    # Remove sample from list
    cls_counts[min_cls_count_txt].remove(sample_idx)
    if len(cls_counts[min_cls_count_txt]) == 0:
        cls_counts.pop(min_cls_count_txt)

    return sample_idx, cls_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        'Script for generating a dataset for finding optimal similarity threshold values'
    )  # noqa
    parser.add_argument('coco_dir', type=str, help='Path to \'coco/\'')
    parser.add_argument('out_dir', type=str, help='Path to output directory')
    parser.add_argument('num_samples', type=int, help='Number of subsamples')
    parser.add_argument('--register_path',
                        type=str,
                        default='txt2idx_star_cseg_coco.pkl',
                        help='Path to txt2idx_star dict')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    coco_dir = args.coco_dir
    out_dir = args.out_dir

    ann_paths = glob(
        osp.join(coco_dir, 'stuffthingmaps_trainval2017/train2017', '*.npz'))
    print(f'Found {len(ann_paths)} .npz annotation files')
    if len(ann_paths) == 0:
        raise IOError('No files found')

    txt2idx_star = load_register(args.register_path)

    idx_star2txt = {}
    for key, val in txt2idx_star.items():
        idx_star2txt[val] = key

    # For listing samples with annotations
    cls_counts = {}
    for key, _ in txt2idx_star.items():
        cls_counts[key] = []

    for ann_path in ann_paths:

        idx_stars = get_sample_idx_stars(ann_path)

        for idx_star in idx_stars:
            cls_txt = idx_star2txt[idx_star]
            ann_idx = get_ann_idx(ann_path)
            cls_counts[cls_txt].append(ann_idx)

    print_counts(cls_counts)

    ##########################################################################
    #  Create a balanced list of annotation categories by sampling from the
    #  fewest categorires
    ##########################################################################
    # Remove empty categories
    empty_cls_txts = []
    for key, val in cls_counts.items():
        if len(val) == 0:
            empty_cls_txts.append(key)
    for cls_txt in empty_cls_txts:
        cls_counts.pop(cls_txt)

    subsample_idxs = []
    # for sample_idx in range(args.num_samples):
    while len(subsample_idxs) < args.num_samples:
        sample_idx, cls_counts = get_rarest_category(cls_counts)
        if sample_idx not in subsample_idxs:
            subsample_idxs.append(sample_idx)

    ##################################
    #  Crate a new sample directory
    ##################################
    new_img_dir = osp.join(args.coco_dir, 'images', 'train2017_sem_thresh')
    new_ann_dir = osp.join(args.coco_dir,
                           'stuffthingmaps_trainval2017/train2017_sem_thresh')
    for new_dir in [new_img_dir, new_ann_dir]:
        if not osp.isdir(new_dir):
            os.mkdir(new_dir)

    for sample_idx in subsample_idxs:

        idx_str = str(sample_idx).zfill(IDX_STR_LEN)

        img_filename = idx_str + '.jpg'
        img_path = osp.join(args.coco_dir, 'images/train2017', img_filename)

        ann_filename = idx_str + '_vl_emb_idxs.npz'
        ann_path = osp.join(args.coco_dir,
                            'stuffthingmaps_trainval2017/train2017',
                            ann_filename)

        new_img_path = osp.join(new_img_dir, img_filename)
        new_ann_path = osp.join(new_ann_dir, ann_filename)

        os.symlink(img_path, new_img_path)
        os.symlink(ann_path, new_ann_path)

    print(f'Subsampled {len(subsample_idxs)} samples for sim thresholding')

    # Printing subsampled category distribution
    ann_paths = glob(
        osp.join(coco_dir, 'stuffthingmaps_trainval2017/train2017_sem_thresh',
                 '*.npz'))

    # For listing samples with annotations
    cls_counts = {}
    for key, _ in txt2idx_star.items():
        cls_counts[key] = []

    for ann_path in ann_paths:

        idx_stars = get_sample_idx_stars(ann_path)

        for idx_star in idx_stars:
            cls_txt = idx_star2txt[idx_star]
            ann_idx = get_ann_idx(ann_path)
            cls_counts[cls_txt].append(ann_idx)

    print_counts(cls_counts)