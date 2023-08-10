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

IDX2TXT_BDD100K = {
    0: 'unlabeled',
    1: 'dynamic',
    2: 'ego vehicle',
    3: 'ground',
    4: 'static',
    5: 'parking',
    6: 'rail track',
    7: 'road',
    8: 'sidewalk',
    9: 'bridge',
    10: 'building',
    11: 'fence',
    12: 'garage',
    13: 'guard rail',
    14: 'tunnel',
    15: 'wall',
    16: 'banner',
    17: 'billboard',
    18: 'lane divider',
    19: 'parking sign',
    20: 'pole',
    21: 'polegroup',
    22: 'street light',
    23: 'traffic cone',
    24: 'traffic device',
    25: 'traffic light',
    26: 'traffic sign',
    27: 'traffic sign frame',
    28: 'terrain',
    29: 'vegetation',
    30: 'sky',
    31: 'person',
    32: 'rider',
    33: 'bicycle',
    34: 'bus',
    35: 'car',
    36: 'caravan',
    37: 'motorcycle',
    38: 'trailer',
    39: 'train',
    40: 'truck',
}


def modify_label_filename(label_filepath):
    """Returns a mmsegmentation-combatible label filename."""
    # Ensure that label filenames are modified only once
    if LABEL_SUFFIX in label_filepath:
        return label_filepath

    # label_filepath = label_filepath.replace('_label_', '_camera_')
    label_filepath = label_filepath.replace('.png', LABEL_SUFFIX)
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
    orig_label = Image.open(label_filepath)  # RGBA image
    orig_label = np.array(orig_label)
    orig_label = orig_label[:, :, 0]  # Get R channel
    orig_label = orig_label.astype(NP_TYPE)

    # Empty array with all elements set as 'ignore id' label
    H, W = orig_label.shape
    new_label = ignore_id * np.ones((H, W), dtype=NP_TYPE)

    # Find valid indices to mask
    orig_idxs = list(np.unique(orig_label))
    all_idxs = list(IDX2TXT_BDD100K.keys())
    idxs = list(set(orig_idxs).intersection(all_idxs))
    for idx in idxs:
        mask = (orig_label == idx)

        # Get uniqe idx representing semantic txt
        idx_star = txt2idx_star[IDX2TXT_BDD100K[idx]]
        new_label[mask] = idx_star

    # Save new vision-language text idx label as
    label_filepath = modify_label_filename(label_filepath)
    np.savez_compressed(label_filepath, new_label)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Cityscapes annotations to TrainIds')
    parser.add_argument('dataset_path', help='cityscapes data path')
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
    """Preprocesses the BD100K panoptic dataset into a vision-language
    embedding index labels with text semantics specified in 'txt2idx_star.pkl'.

    VL embedding index files:
        0a0a0b1a-7c39d841_train_id.png
        --> 0a0a0b1a-7c39d841_vl_emb_idxs.npz (np.uint32)
    
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
    for txt in IDX2TXT_BDD100K.values():
        txt2idx_star = add_entry(txt2idx_star, txt)
    save_register(args.register_path, txt2idx_star)

    ####################
    #  Convert labels
    ####################
    search_str = osp.join(dataset_path, 'labels/pan_seg/bitmasks/*/*.png')
    label_filepaths = glob.glob(search_str)

    tasks = [(path, txt2idx_star) for path in label_filepaths]

    if args.nproc > 1:
        track_parallel_progress(convert_label_to_idx_star, tasks, args.nproc)
    else:
        track_progress(convert_label_to_idx_star, tasks)


if __name__ == '__main__':
    main()
