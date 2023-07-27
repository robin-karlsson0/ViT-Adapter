#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import glob
import json
import os
import os.path as osp
import pickle
import random
from os import symlink
from shutil import copyfile

import cv2
# import ipdb
import matplotlib._color_data as mcd
import mmcv
import numpy as np
from PIL import Image

from tools.convert_datasets.txt2idx_star import (add_entry, load_register,
                                                 save_register)

random.seed(14)

NP_TYPE = np.uint32

##########################
#  Ref: utils_ade20k.py
##########################

_NUMERALS = '0123456789abcdefABCDEF'
_HEXDEC = {
    v: int(v, 16)
    for v in (x + y for x in _NUMERALS for y in _NUMERALS)
}
LOWERCASE, UPPERCASE = 'x', 'X'


def rgb(triplet):
    return _HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]], _HEXDEC[triplet[4:6]]


def loadAde20K(file):
    fileseg = file.replace('.jpg', '_seg.png')
    with Image.open(fileseg) as io:
        seg = np.array(io)

    # Obtain the segmentation mask, bult from the RGB channels of the _seg file
    R = seg[:, :, 0]
    G = seg[:, :, 1]
    B = seg[:, :, 2]
    ObjectClassMasks = (R / 10).astype(np.int32) * 256 + (G.astype(np.int32))

    # Obtain the instance mask from the blue channel of the _seg file
    Minstances_hat = np.unique(B, return_inverse=True)[1]
    Minstances_hat = np.reshape(Minstances_hat, B.shape)
    ObjectInstanceMasks = Minstances_hat

    level = 0
    PartsClassMasks = []
    PartsInstanceMasks = []
    while True:
        level = level + 1
        file_parts = file.replace('.jpg', '_parts_{}.png'.format(level))
        if osp.isfile(file_parts):
            with Image.open(file_parts) as io:
                partsseg = np.array(io)
            R = partsseg[:, :, 0]
            G = partsseg[:, :, 1]
            B = partsseg[:, :, 2]
            PartsClassMasks.append((np.int32(R) / 10) * 256 + np.int32(G))
            PartsInstanceMasks = PartsClassMasks
            # TODO:  correct partinstancemasks

        else:
            break

    objects = {}
    parts = {}

    attr_file_name = file.replace('.jpg', '.json')
    if osp.isfile(attr_file_name):
        with open(attr_file_name, 'r') as f:
            input_info = json.load(f)

        contents = input_info['annotation']['object']
        instance = np.array([int(x['id']) for x in contents])
        names = [x['raw_name'] for x in contents]
        corrected_raw_name = [x['name'] for x in contents]
        partlevel = np.array([int(x['parts']['part_level']) for x in contents])
        ispart = np.array([p > 0 for p in partlevel])
        iscrop = np.array([int(x['crop']) for x in contents])
        listattributes = [x['attributes'] for x in contents]
        polygon = [x['polygon'] for x in contents]
        for p in polygon:
            p['x'] = np.array(p['x'])
            p['y'] = np.array(p['y'])

        objects['instancendx'] = instance[ispart == 0]
        objects['class'] = [names[x] for x in list(np.where(ispart == 0)[0])]
        objects['corrected_raw_name'] = [
            corrected_raw_name[x] for x in list(np.where(ispart == 0)[0])
        ]
        objects['iscrop'] = iscrop[ispart == 0]
        objects['listattributes'] = [
            listattributes[x] for x in list(np.where(ispart == 0)[0])
        ]
        objects['polygon'] = [
            polygon[x] for x in list(np.where(ispart == 0)[0])
        ]

        parts['instancendx'] = instance[ispart == 1]
        parts['class'] = [names[x] for x in list(np.where(ispart == 1)[0])]
        parts['corrected_raw_name'] = [
            corrected_raw_name[x] for x in list(np.where(ispart == 1)[0])
        ]
        parts['iscrop'] = iscrop[ispart == 1]
        parts['listattributes'] = [
            listattributes[x] for x in list(np.where(ispart == 1)[0])
        ]
        parts['polygon'] = [polygon[x] for x in list(np.where(ispart == 1)[0])]

    return {
        'img_name': file,
        'segm_name': fileseg,
        'class_mask': ObjectClassMasks,
        'instance_mask': ObjectInstanceMasks,
        'partclass_mask': PartsClassMasks,
        'part_instance_mask': PartsInstanceMasks,
        'objects': objects,
        'parts': parts
    }


def load_sample(sample_idx: int, index_ade20k: dict) -> tuple:
    '''
    Returns an ADE20K sample dict by sample index.

    Args:
        sample_idx: Sample index (i.e. 0 --> N).
        index_ade20k: Dataset index dict object un-pickled from a DL file.

    Returns:
        img: Sample RGB image as (H,W,3) np.uint8.
        seg: Sample class idx segmentation map as (H, W) np.uint32.
    '''
    if sample_idx >= len(index_ade20k['folder']):
        raise IOError('Sample idx greater than number of samples')

    sample_dir = index_ade20k['folder'][sample_idx]
    sample_filename = index_ade20k['filename'][sample_idx]
    sample_path = osp.join('data', sample_dir, sample_filename)

    if not osp.isfile(sample_path):
        raise IOError(f'Could not read sample file ({sample_path})')
    sample = loadAde20K(sample_path)

    # img = cv2.imread(sample['img_name'])[:, :, ::-1]
    seg = sample['class_mask']

    return seg, sample_path


###########################
# End of utils_ade20k.py
###########################


def gen_idx2cls_dict(index_ade20k: dict) -> dict:
    '''
    Generate a dict for converting 'class_mask' idx --> 'objectnames' txt.
        Ex: idx2cls[2] --> 'abacus'
    
    NOTE The class index map (i.e. 'class_mask') add a 0 ignore idx
         ==> Add +1 to actual object index.

    '''
    idx2cls = {}
    for obj_idx in range(len(index_ade20k['objectnames'])):
        cls_idx = obj_idx + 1
        idx2cls[cls_idx] = index_ade20k['objectnames'][obj_idx]

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
                              ignore_id: int = np.iinfo(NP_TYPE).max):
    """Saves a new semantic label following the v2.0 'trainids' format.

    The new image is saved into the same directory as the original image having
    an additional suffix.
    Args:
        label_filepath: Path to the original semantic label.
        ignore_id: Default value for unlabeled elements.
    """
    sample_idx, idx2idx_star, index_ade20k = task

    seg, sample_path = load_sample(sample_idx, index_ade20k)

    seg_idxs = list(np.unique(seg))
    # Remove background elements labeled by idx = 0
    seg_idxs = set(seg_idxs) - {0}
    seg_idxs = list(seg_idxs)

    H, W = seg.shape
    new_label = ignore_id * np.ones((H, W), dtype=NP_TYPE)

    for idx in seg_idxs:
        mask = (seg == idx)
        idx_star = idx2idx_star[idx]
        np.place(new_label, mask, [idx_star])

    ann_filepath = sample_path.replace('.jpg', '.npz')
    np.savez_compressed(ann_filepath, new_label)


def restructure_ade20k_directory(ade20k_path,
                                 train_on_val_and_test=False,
                                 use_symlinks=True):
    """Creates a new directory structure and link existing files into it.
    Required to make the ADE20K dataset conform to the mmsegmentation
    frameworks expected dataset structure.

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
        ade20k_path: Absolute path to the ADE20K dataset root directory.
        train_on_val_and_test: Use validation and test samples as training
                               samples if True.
        use_symlinks: Symbolically link existing files in the original GTA 5
                      dataset directory. If false, files will be copied.
    """
    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(ade20k_path, 'img_dir', 'training'))
    mmcv.mkdir_or_exist(osp.join(ade20k_path, 'img_dir', 'validation'))
    mmcv.mkdir_or_exist(osp.join(ade20k_path, 'ann_dir', 'training'))
    mmcv.mkdir_or_exist(osp.join(ade20k_path, 'ann_dir', 'validation'))

    for split in ['training', 'validation']:
        # Ex: ADE20K_2021_17_01/images/ADE/training/cultural/apse__indoor/*.jpg
        img_search_str = f'images/ADE/{split}/*/*/*.jpg'
        img_filepaths = glob.glob(osp.join(ade20k_path, img_search_str))

        assert len(img_filepaths) > 0

        for img_filepath in img_filepaths:

            ann_filepath = img_filepath.replace('.jpg', '.npz')

            img_filename = img_filepath.split('/')[-1]
            ann_filename = ann_filepath.split('/')[-1]

            img_linkpath = f'{ade20k_path}/img_dir/{split}/{img_filename}'
            ann_linkpath = f'{ade20k_path}/ann_dir/{split}/{ann_filename}'

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
        description='Convert ADE20K annotations to VL index masks')
    parser.add_argument('ade20k_path',
                        help='ADE20K segmentation data absolute path\
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
    """A script for making the ADE20K dataset compatible with vision-langauge
    mmsegmentation.

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e. data/ADE20K)

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
        python tools/convert_datasets/ade20k.py abs_path/to/ade20k
    """
    args = parse_args()
    ade20k_path = args.ade20k_path
    out_dir = args.out_dir if args.out_dir else ade20k_path
    mmcv.mkdir_or_exist(out_dir)

    with open(osp.join(ade20k_path, 'index_ade20k.pkl'), 'rb') as f:
        index_ade20k = pickle.load(f)

    idx2cls = gen_idx2cls_dict(index_ade20k)

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

        num_samples = len(index_ade20k['folder'])
        tasks = [(idx, idx2idx_star, index_ade20k)
                 for idx in range(num_samples)]

        if args.nproc > 1:
            mmcv.track_parallel_progress(convert_label_to_idx_star, tasks,
                                         args.nproc)
        else:
            mmcv.track_progress(convert_label_to_idx_star, tasks)

    # Restructure directory structure into 'img_dir' and 'ann_dir'
    if args.restruct:
        restructure_ade20k_directory(out_dir, args.train_on_val_and_test,
                                     args.symlink)


if __name__ == '__main__':
    main()
