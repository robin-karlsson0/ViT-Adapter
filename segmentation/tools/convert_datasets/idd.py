# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import os.path as osp
import pickle

import mmcv
import numpy as np
# from cityscapesscripts.preparation.json2labelImg import json2labelImg
from mmengine.utils import (mkdir_or_exist, scandir, track_parallel_progress,
                            track_progress)
from PIL import Image

from tools.convert_datasets.txt2idx_star import (add_entry, load_register,
                                                 save_register)

NP_TYPE = np.uint32

LABEL_SUFFIX = "_vl_emb_idxs.npz"

IDX2TXT_IDD = {
    0: 'road',
    1: 'sidewalk',
    2: 'building',
    3: 'wall',
    4: 'fence',
    5: 'pole',
    6: 'traffic light',
    7: 'traffic sign',
    8: 'vegetation',
    9: 'terrain',
    10: 'sky',
    11: 'person',
    12: 'rider',
    13: 'car',
    14: 'truck',
    15: 'bus',
    16: 'train',
    17: 'motorcycle',
    18: 'bicycle',
}


def modify_label_filename(label_filepath):
    """Returns a mmsegmentation-combatible label filename."""
    # Ensure that label filenames are modified only once
    if LABEL_SUFFIX in label_filepath:
        return label_filepath

    # label_filepath = label_filepath.replace('_label_', '_camera_')
    label_filepath = label_filepath.replace('_labelcsTrainIds.png',
                                            LABEL_SUFFIX)
    return label_filepath


def convert_label_to_idx_star(task: tuple,
                              ignore_id: int = np.iinfo(NP_TYPE).max):
    """Saves a new vision-language text index matrix with unique indices.

    The new image is saved into the same directory as the original image having
    an additional suffix.

    Args:
        label_filepath: Path to the original semantic label.
        ignore_id: Default value for unlabeled elements.
    """
    label_filepath, txt2idx_star = task
    # Read label file as Numpy array (H, W, 3) --> (H, W)
    orig_label = mmcv.imread(label_filepath)
    orig_label = orig_label[:, :, 0]
    orig_label = orig_label.astype(NP_TYPE)

    # Empty array with all elements set as 'ignore id' label
    H, W = orig_label.shape
    new_label = ignore_id * np.ones((H, W), dtype=NP_TYPE)

    # Find valid indices to mask
    orig_idxs = list(np.unique(orig_label))
    all_idxs = list(IDX2TXT_IDD.keys())
    idxs = list(set(orig_idxs).intersection(all_idxs))
    for idx in idxs:
        mask = (orig_label == idx)

        # Get uniqe idx representing semantic txt
        idx_star = txt2idx_star[IDX2TXT_IDD[idx]]
        new_label[mask] = idx_star

    # Save new vision-language text idx label as
    label_filepath = modify_label_filename(label_filepath)
    np.savez_compressed(label_filepath, new_label)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert IDD annotations to VL embedding maps')
    parser.add_argument('dataset_path', help='Path to IDD_Segmentation/ dir.')
    parser.add_argument('-o', '--out-dir', help='output path')
    parser.add_argument('--nproc',
                        default=1,
                        type=int,
                        help='number of process')
    parser.add_argument('--register_path',
                        default='txt2idx_star.pkl',
                        help='Path to txt2idx_star dict')
    args = parser.parse_args()
    return args


def main():
    """Preprocesses the IDD dataset annotations into a vision-language
    embedding index labels with text semantics specified in 'txt2idx_star.pkl'.

    VL embedding index files:
        005506_gtFine_labelcsTrainIds.png
        --> 005506_gtFine_vl_emb_idxs.npz (np.uint32)
    
    New text semantic entries are added to the 'txt2idx_star' dict.
    """
    args = parse_args()
    dataset_path = args.dataset_path
    out_dir = args.out_dir if args.out_dir else dataset_path
    mkdir_or_exist(out_dir)

    #######################################
    #  Add text descriptions to register
    #######################################
    txt2idx_star = load_register(args.register_path)
    for txt in IDX2TXT_IDD.values():
        txt2idx_star = add_entry(txt2idx_star, txt)
    save_register(args.register_path, txt2idx_star)

    ####################
    #  Convert labels
    ####################
    search_str = osp.join(dataset_path,
                          'gtFine/*/*/*_gtFine_labelcsTrainIds.png')
    label_filepaths = glob.glob(search_str)

    tasks = [(path, txt2idx_star) for path in label_filepaths]

    if args.nproc > 1:
        track_parallel_progress(convert_label_to_idx_star, tasks, args.nproc)
    else:
        track_progress(convert_label_to_idx_star, tasks)


if __name__ == '__main__':
    main()
