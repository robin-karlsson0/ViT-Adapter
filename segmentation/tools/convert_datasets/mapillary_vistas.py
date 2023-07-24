#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import json
import glob
import os.path as osp
import random
import os
from os import symlink
from shutil import copyfile

import mmcv
import numpy as np
from tools.convert_datasets.txt2idx_star import (load_register, add_entry,
                                                 save_register)

random.seed(14)

NP_TYPE = np.uint32


def load_cls2rgb_dict(config_path: str, label_depth: int) -> dict:
    """Returns a dict containing class names and RGB values as keys and values.

    NOTE Actual depth is bounded by actual sample class label depth.

    Args:
        config_path: Path to 'config_vX.json' file.
        depth: Negative integer selecting label granularity
               Ex: construction--barrier--guard-rail
                        [0]        [1]       [2]

    Returns:
        Dict [class name] -> RGB tuple.
    """
    if not os.path.isfile(config_path):
        raise IOError(f'Config path does not exist: {config_path}')
    with open(config_path, 'r') as f:
        config = json.load(f)

    cls2rgb = {}
    for cls_dict in config['labels']:
        # clss = cls_dict['name']
        cls = cls_dict['readable']
        rgb = cls_dict['color']

        # Get class label at specified depth
        #clss = clss.split('--')
        #if label_depth >= len(clss):
        #    depth_sample = len(clss) - 1
        #else:
        #    depth_sample = label_depth
        #cls = clss[depth_sample]
        #cls = cls.replace('-', ' ')

        cls2rgb[cls] = rgb

    return cls2rgb


def modify_label_filename(label_filepath, ver):
    """Returns a mmsegmentation-combatible label filename."""
    ver = ver.replace('.', '_')  # Ex: v1.2 --> v1_2
    label_suffix = f'_{ver}_vl_emb_idxs.npz'
    # Ensure that label filenames are modified only once
    if label_suffix in label_filepath:
        return label_filepath
    orig_suffix = label_filepath[-4:]
    if orig_suffix == '.png':
        label_filepath = label_filepath.replace('.png', label_suffix)
    elif orig_suffix == '.jpg':
        label_filepath = label_filepath.replace('.jpg', label_suffix)
    return label_filepath


def convert_label_to_idx_star(task: tuple,
                              ignore_id: int = np.iinfo(NP_TYPE).max):
    """Saves a new semantic label following the v2.0 'trainids' format.

    The new image is saved into the same directory as the original image having
    an additional suffix.
    Args:
        label_filepath: Path to the original semantic label.
        ignore_id: Default value for unlabeled elements.
    """
    label_filepath, rgb2idx_star, ver = task
    # Read label file as Numpy array (H, W, 3) --> (H, W)
    orig_label = mmcv.imread(label_filepath, channel_order='rgb')
    orig_label = orig_label.astype(NP_TYPE)
    seg_colors = np.unique(orig_label.reshape(-1, orig_label.shape[2]), axis=0)
    seg_colors = list(seg_colors)

    # Empty array with all elements set as 'ignore id' label
    H, W, _ = orig_label.shape
    new_label = ignore_id * np.ones((H, W), dtype=NP_TYPE)
    # new_label = 0 * np.ones((H, W), dtype=NP_TYPE)

    for seg_color in seg_colors:

        # The operation produce (H,W,3) array of (i,j,k)-wise truth values
        mask = (orig_label == seg_color)
        # collapse RGB channel dimension (H,W,3) --> (H,W)
        #   Ex: [True, False, False] --> [False]
        mask = np.prod(mask, axis=-1)
        # Get uniqe idx representing semantic txt
        idx_star = rgb2idx_star[tuple(seg_color)]

        np.place(new_label, mask, [idx_star])

    # Save new vision-language text idx label as
    label_filepath = modify_label_filename(label_filepath, ver)
    np.savez_compressed(label_filepath, new_label, ver)


def restructure_vistas_directory(vistas_path,
                                 ver,
                                 train_on_val_and_test=False,
                                 use_symlinks=True):
    """Creates a new directory structure and link existing files into it.
    Required to make the Mapillary Vistas dataset conform to the mmsegmentation
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
        vistas_path: Absolute path to the Mapillary Vistas 'vistas/' directory.
        ver: Version string (e.g. v1.2)
        train_on_val_and_test: Use validation and test samples as training
                               samples if True.
        label_suffix: Label filename ending string.
        use_symlinks: Symbolically link existing files in the original GTA 5
                      dataset directory. If false, files will be copied.
    """
    # Create new directory structure (if not already exist)
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'img_dir'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'img_dir', 'training'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'img_dir', 'validation'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'img_dir', 'testing'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'ann_dir'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'ann_dir', 'training'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'ann_dir', 'validation'))
    mmcv.mkdir_or_exist(osp.join(vistas_path, 'ann_dir', 'testing'))

    for split in ['training', 'validation', 'testing']:
        img_filepaths = glob.glob(f'{vistas_path}/{split}/images/*.jpg')

        assert len(img_filepaths) > 0

        for img_filepath in img_filepaths:

            img_filename = img_filepath.split('/')[-1]

            ann_filename = modify_label_filename(img_filename, ver)
            ann_filepath = f'{vistas_path}/{split}/{ver}/labels/{ann_filename}'

            img_linkpath = f'{vistas_path}/img_dir/{split}/{img_filename}'
            if split == 'testing':
                ann_linkpath = None
            else:
                ann_linkpath = f'{vistas_path}/ann_dir/{split}/{ann_filename}'

            if use_symlinks:
                # NOTE: Can only create new symlinks if no priors ones exists
                try:
                    symlink(img_filepath, img_linkpath)
                except FileExistsError:
                    pass
                try:
                    if split != 'testing':
                        symlink(ann_filepath, ann_linkpath)
                except FileExistsError:
                    pass

            else:
                copyfile(img_filepath, img_linkpath)
                if split != 'testing':
                    copyfile(ann_filepath, ann_linkpath)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Mapillary Vistas annotations to trainIds')
    parser.add_argument('vistas_path',
                        help='Mapillary vistas segmentation data absolute path\
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
    parser.add_argument('--version',
                        default='v2.0',
                        help='Semantic label version: \'v2.0\' (124 classes)')
    parser.add_argument(
        '--label_depth',
        type=int,
        default='2',
        help='Class label depth (e.g. construction--barrier--guard-rail)')
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
    """A script for making the Mapillary Vistas dataset compatible with
    mmsegmentation.

    NOTE: The input argument path must be the ABSOLUTE PATH to the dataset
          - NOT the symbolically linked one (i.e. data/mapillary_vistas)

    Segmentation label conversion:

        The function 'convert_TYPE_trainids()' converts all RGB segmentation to
        their corresponding categorical segmentation and saves them as new
        label image files.

        Label choice 'cityscapes' (default) results in labels with 18 classes
        with the filename suffix '_trainIds.png'.

    Dataset split:

        Arranges samples into 'train', 'val', and 'test' splits according to
        predetermined directory structure

    NOTE: Add the optional argument `--train-on-val-and-test` to train on the
    entire dataset, as is usefull in the synthetic-to-real domain adaptation
    experiment setting.

    Add `--nproc N` for multiprocessing using N threads.

    Example usage:
        python tools/convert_datasets/mapillary_vistas.py
            abs_path/to/mapillary_vistas
    """
    args = parse_args()
    vistas_path = args.vistas_path
    out_dir = args.out_dir if args.out_dir else vistas_path
    mmcv.mkdir_or_exist(out_dir)
    ver = args.version

    ##############################
    #  Generate seg RGB --> cls
    ##############################

    #######################################
    #  Add text descriptions to register
    #######################################
    config_path = os.path.join(args.vistas_path, f'config_{ver}.json')
    cls2rgb = load_cls2rgb_dict(config_path, args.label_depth)
    txt2idx_star = load_register(args.register_path)
    for txt in cls2rgb.keys():
        txt2idx_star = add_entry(txt2idx_star, txt)
    save_register(args.register_path, txt2idx_star)

    ####################
    #  Convert labels
    ####################
    if args.convert:

        # Create seg RGB --> idx_star dict
        rgb2idx_star = {}
        for cls, rgb in cls2rgb.items():
            idx_star = txt2idx_star[cls]
            rgb2idx_star[tuple(rgb)] = idx_star

        # Create a list of filepaths to all original labels
        ignore_suffix_wo_png = '_trainIds'
        label_filepaths = glob.glob(
            osp.join(vistas_path,
                     f'*/{ver}/labels/*[!{ignore_suffix_wo_png}].png'))

        tasks = [(path, rgb2idx_star, ver) for path in label_filepaths]

        if args.nproc > 1:
            mmcv.track_parallel_progress(convert_label_to_idx_star, tasks,
                                         args.nproc)
        else:
            mmcv.track_progress(convert_label_to_idx_star, tasks)

    # Restructure directory structure into 'img_dir' and 'ann_dir'
    if args.restruct:
        restructure_vistas_directory(out_dir, ver, args.train_on_val_and_test,
                                     args.symlink)


if __name__ == '__main__':
    main()
