#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import glob
import json
import os
import os.path as osp
import random
from os import symlink
from shutil import copyfile

import mmcv
import numpy as np

from tools.convert_datasets.txt2idx_star import (add_entry, load_register,
                                                 save_register)

random.seed(14)

NP_TYPE = np.uint32


def gen_idx2cls_dict(objectinfo150_path: str) -> dict:
    '''
    Generate a dict for converting 'class_mask' idx --> 'objectnames' txt.
        Ex: idx2cls[2] --> 'abacus'
    
    NOTE The class index map (i.e. 'class_mask') add a 0 ignore idx
         ==> Add +1 to actual object index.
        
    Args:
        objectinfo150_path: Path to the 'objectinfo150.txt' dataset file.
    
    Returns:
        Dict mapping idx --> cls txt.
    '''
    if not os.path.isfile(objectinfo150_path):
        raise IOError(f'Objectinfo file does not exist ({objectinfo150_path})')
    with open(objectinfo150_path, 'r') as f:
        lines = f.readlines()

    idx2cls = {}
    # Skip first heading row
    for line in lines[1:]:
        idx = line.split('\t')[0]
        idx = int(idx)
        # Read first entry and remove possible newline
        cls_txt = line.split('\t')[-1].split(', ')[0]
        cls_txt = cls_txt.replace('\n', '')
        idx2cls[idx] = cls_txt

    return idx2cls


def gen_idx2idx_star_dict(txt2idx_star: dict, idx2cls: dict):
    '''
    Generate a dict for converting ADE20K class idx --> idx*.
        Ex: idx2idx_star[2] --> 235
    '''
    idx2idx_star = {}
    for idx, cls in idx2cls.items():
        idx_star = txt2idx_star[cls]
        idx2idx_star[idx] = idx_star

    return idx2idx_star


def convert_label_to_idx_star(task: tuple,
                              ignore_id: int = np.iinfo(NP_TYPE).max,
                              bg_idx: int = 0):
    """Saves a new vision-language text index matrix with unique indices.

    The new image is saved into the same directory as the original image having
    an additional suffix.

    Args:
        label_filepath: Path to the original semantic label.
        ignore_id: Default value for unlabeled elements.
        bg_idx: Ignored 'background' idx in original labels.
    """
    label_filepath, txt2idx_star = task
    # Read label file as Numpy array (H, W, 3) --> (H, W)
    orig_label = mmcv.imread(label_filepath)
    orig_label = orig_label[:, :, 0]
    orig_label = orig_label.astype(NP_TYPE)

    # Remove ignored 'background' idx
    orig_idxs = list(np.unique(orig_label))
    orig_idxs = set(orig_idxs) - {bg_idx}
    orig_idxs = list(orig_idxs)

    # Empty array with all elements set as 'ignore id' label
    H, W = orig_label.shape
    new_label = ignore_id * np.ones((H, W), dtype=NP_TYPE)

    for orig_idx in orig_idxs:
        mask = (orig_label == orig_idx)

        # Get uniqe idx representing semantic txt
        idx_star = txt2idx_star[orig_idx]
        new_label[mask] = idx_star

    # Save new vision-language text idx label as
    label_filepath = label_filepath.replace('.png', '.npz')
    np.savez_compressed(label_filepath, new_label)


def restructure_adechallenge_directory(adechallenge_path,
                                       train_on_val_and_test=False,
                                       use_symlinks=True):
    """Creates a new directory structure and link existing files into it.
    Required to make the ADE Challenge 2016 dataset conform to the
    mmsegmentation frameworks expected dataset structure.

    └── img_dir
    │   ├── train
    │   │   ├── xxx{img_suffix}
    |   |   ...
    │   ├── val
    │   │   ├── yyy{img_suffix}
    │   │   ...
    │   ...
    └── ann_dir
        ├── train
        │   ├── xxx{seg_map_suffix}
        |   ...
        ├── val
        |   ├── yyy{seg_map_suffix}
        |   ...
        ...
    Args:
        adechallenge_path: Absolute path to the ADE Challenge 2016 dataset root
                           directory.
        train_on_val_and_test: Use validation and test samples as training
                               samples if True.
        use_symlinks: Symbolically link existing files in the original GTA 5
                      dataset directory. If false, files will be copied.
    """
    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(adechallenge_path, 'img_dir', 'training'))
    mmcv.mkdir_or_exist(osp.join(adechallenge_path, 'img_dir', 'validation'))
    mmcv.mkdir_or_exist(osp.join(adechallenge_path, 'ann_dir', 'training'))
    mmcv.mkdir_or_exist(osp.join(adechallenge_path, 'ann_dir', 'validation'))

    for split in ['training', 'validation']:
        # Ex: ADEChallengeData2016/images/training/*.jpg
        img_search_str = f'images/{split}/*.jpg'
        img_filepaths = glob.glob(osp.join(adechallenge_path, img_search_str))

        assert len(img_filepaths) > 0

        for img_filepath in img_filepaths:

            ann_filepath = img_filepath.replace('images/', 'annotations/')
            ann_filepath = ann_filepath.replace('.jpg', '.npz')

            img_filename = img_filepath.split('/')[-1]
            ann_filename = ann_filepath.split('/')[-1]

            img_linkpath = f'{adechallenge_path}/img_dir/{split}/{img_filename}'
            ann_linkpath = f'{adechallenge_path}/ann_dir/{split}/{ann_filename}'

            if use_symlinks:
                # NOTE: Can only create new symlinks if no prior ones exists
                try:
                    symlink(img_filepath, img_linkpath)
                except FileExistsError:
                    pass
                try:
                    symlink(ann_filepath, ann_linkpath)
                except FileExistsError:
                    pass

            else:
                copyfile(img_filepath, img_linkpath)
                copyfile(ann_filepath, ann_linkpath)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ADE Challenge 2016 annotations to VL index masks')
    parser.add_argument('adechallenge_path',
                        help='adechallenge segmentation data absolute path\
                           (NOT the symbolically linked one!)')
    parser.add_argument('-o', '--out-dir', help='Output path')
    parser.add_argument('--no-convert',
                        dest='convert',
                        action='store_false',
                        help='Skips converting label images')
    parser.set_defaults(convert=True)
    parser.add_argument('--no-restruct',
                        dest='restruct',
                        action='store_false',
                        help='Skips restructuring directory structure')
    parser.set_defaults(restruct=True)
    parser.add_argument('--register_path',
                        default='txt2idx_star.pkl',
                        help='Path to txt2idx_star dict')
    parser.add_argument(
        '--train-on-val-and-test',
        dest='train_on_val_and_test',
        action='store_true',
        help='Use validation and test samples as training samples')
    parser.add_argument('--objectinfo_file',
                        default='objectInfo150.txt',
                        type=str)
    parser.set_defaults(train_on_val_and_test=False)
    parser.add_argument('--nproc',
                        default=1,
                        type=int,
                        help='Number of process')
    parser.add_argument('--no-symlink',
                        dest='symlink',
                        action='store_false',
                        help='Use hard links instead of symbolic links')
    parser.set_defaults(symlink=True)
    args = parser.parse_args()
    return args


def main():
    """A script for making the ADE Challenge 2016 dataset compatible with
    vision-langauge mmsegmentation.

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e. data/ADEChallengeData2016)

    Segmentation label conversion:

        The function 'convert_TYPE_trainids()' converts all class index
        segmentation to their corresponding universal class index and saves
        them as new label image files.

    Dataset split:

        Arranges samples into 'train', 'val', and 'test' splits according to
        predetermined directory structure

    NOTE: Add the optional argument `--train-on-val-and-test` to train on the
    entire dataset, as is usefull in the synthetic-to-real domain adaptation
    experiment setting.

    Add `--nproc N` for multiprocessing using N threads.

    Example usage:
        python tools/convert_datasets/adechallenge.py abs_path/to/ADEChallengeData2016
    """
    args = parse_args()
    adechallenge_path = args.adechallenge_path
    out_dir = args.out_dir if args.out_dir else adechallenge_path
    mmcv.mkdir_or_exist(out_dir)

    objectinfo_path = os.path.join(adechallenge_path, args.objectinfo_file)
    idx2cls = gen_idx2cls_dict(objectinfo_path)

    #######################################
    #  Add text descriptions to register
    #######################################
    txt2idx_star = load_register(args.register_path)
    for txt in idx2cls.values():
        txt2idx_star = add_entry(txt2idx_star, txt)
    save_register(args.register_path, txt2idx_star)

    ####################
    #  Convert labels
    ####################
    if args.convert:

        # Create 'class_mask' idx --> idx_star dict
        idx2idx_star = gen_idx2idx_star_dict(txt2idx_star, idx2cls)

        search_str = osp.join(adechallenge_path, f'annotations/*/*.png')
        ann_paths = glob.glob(search_str)

        tasks = [(ann_path, idx2idx_star) for ann_path in ann_paths]

        if args.nproc > 1:
            mmcv.track_parallel_progress(convert_label_to_idx_star, tasks,
                                         args.nproc)
        else:
            mmcv.track_progress(convert_label_to_idx_star, tasks)

    # Restructure directory structure into 'img_dir' and 'ann_dir'
    # if args.restruct:
    #     restructure_adechallenge_directory(out_dir, args.train_on_val_and_test,
    #                                        args.symlink)


if __name__ == '__main__':
    main()
