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

IDX2TXT_SUNRGBD = {
    # 0: IGNORE
    1: 'wall',
    2: 'floor',
    3: 'cabinet',
    4: 'bed',
    5: 'chair',
    6: 'sofa',
    7: 'table',
    8: 'door',
    9: 'window',
    10: 'bookshelf',
    11: 'picture',
    12: 'counter',
    13: 'blinds',
    14: 'desk',
    15: 'shelves',
    16: 'curtain',
    17: 'dresser',
    18: 'pillow',
    19: 'mirror',
    20: 'floor_mat',
    21: 'clothes',
    22: 'ceiling',
    23: 'books',
    24: 'fridge',
    25: 'tv',
    26: 'paper',
    27: 'towel',
    28: 'shower_curtain',
    29: 'box',
    30: 'whiteboard',
    31: 'person',
    32: 'night_stand',
    33: 'toilet',
    34: 'sink',
    35: 'lamp',
    36: 'bathtub',
    37: 'bag',
}


def modify_label_filename(label_filepath):
    """Returns a mmsegmentation-combatible label filename."""
    # Ensure that label filenames are modified only once
    if LABEL_SUFFIX in label_filepath:
        return label_filepath

    # Match image format
    # img-003354.jpg
    #   00003554.png
    filename = label_filepath.split('/')[-1]
    dir = label_filepath[:-len(filename)]
    filename = "img-" + filename[2:]
    filename = filename.replace('.png', LABEL_SUFFIX)

    label_filepath = osp.join(dir, filename)
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
    all_idxs = list(IDX2TXT_SUNRGBD.keys())
    idxs = list(set(orig_idxs).intersection(all_idxs))
    for idx in idxs:
        mask = (orig_label == idx)

        # Get uniqe idx representing semantic txt
        idx_star = txt2idx_star[IDX2TXT_SUNRGBD[idx]]
        new_label[mask] = idx_star

    # Save new vision-language text idx label as
    label_filepath = modify_label_filename(label_filepath)
    np.savez_compressed(label_filepath, new_label)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert SUN RGB-D annotations to VL embedding maps')
    parser.add_argument('dataset_path', help='Path to SUNRGBD/ dir.')
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
        00000001.png --> 00000001_vl_emb_idxs.npz (np.uint32)
    
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
    for txt in IDX2TXT_SUNRGBD.values():
        txt2idx_star = add_entry(txt2idx_star, txt)
    save_register(args.register_path, txt2idx_star)

    ####################
    #  Convert labels
    ####################
    label_filepaths = glob.glob(osp.join(dataset_path, 'label37/*/*.png'))

    tasks = [(path, txt2idx_star) for path in label_filepaths]

    if args.nproc > 1:
        track_parallel_progress(convert_label_to_idx_star, tasks, args.nproc)
    else:
        track_progress(convert_label_to_idx_star, tasks)


if __name__ == '__main__':
    main()
