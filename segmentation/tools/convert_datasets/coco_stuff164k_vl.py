# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from glob import glob

import mmcv
import numpy as np

from tools.convert_datasets.txt2idx_star import (add_entry, load_register,
                                                 save_register)

COCO_LEN = 123287

NP_TYPE = np.uint32

LABEL_SUFFIX = "_vl_emb_idxs.npz"

# NOTE: .PNG label indices are SHIFTED -1 from the provided labels
# Ref: https://github.com/nightrome/cocostuff/issues/7
# Ref: https://github.com/nightrome/cocostuff/blob/master/labels.txt
IDX2TXT_COCOSTUFF = {
    #: 'unlabeled'
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'street sign',
    12: 'stop sign',
    13: 'parking meter',
    14: 'bench',
    15: 'bird',
    16: 'cat',
    17: 'dog',
    18: 'horse',
    19: 'sheep',
    20: 'cow',
    21: 'elephant',
    22: 'bear',
    23: 'zebra',
    24: 'giraffe',
    25: 'hat',
    26: 'backpack',
    27: 'umbrella',
    28: 'shoe',
    29: 'eye glasses',
    30: 'handbag',
    31: 'tie',
    32: 'suitcase',
    33: 'frisbee',
    34: 'skis',
    35: 'snowboard',
    36: 'sports ball',
    37: 'kite',
    38: 'baseball bat',
    39: 'baseball glove',
    40: 'skateboard',
    41: 'surfboard',
    42: 'tennis racket',
    43: 'bottle',
    44: 'plate',
    45: 'wine glass',
    46: 'cup',
    47: 'fork',
    48: 'knife',
    49: 'spoon',
    50: 'bowl',
    51: 'banana',
    52: 'apple',
    53: 'sandwich',
    54: 'orange',
    55: 'broccoli',
    56: 'carrot',
    57: 'hot dog',
    58: 'pizza',
    59: 'donut',
    60: 'cake',
    61: 'chair',
    62: 'couch',
    63: 'potted plant',
    64: 'bed',
    65: 'mirror',
    66: 'dining table',
    67: 'window',
    68: 'desk',
    69: 'toilet',
    70: 'door',
    71: 'tv',
    72: 'laptop',
    73: 'mouse',
    74: 'remote',
    75: 'keyboard',
    76: 'cell phone',
    77: 'microwave',
    78: 'oven',
    79: 'toaster',
    80: 'sink',
    81: 'refrigerator',
    82: 'blender',
    83: 'book',
    84: 'clock',
    85: 'vase',
    86: 'scissors',
    87: 'teddy bear',
    88: 'hair drier',
    89: 'toothbrush',
    90: 'hair brush',
    91: 'banner',
    92: 'blanket',
    93: 'branch',
    94: 'bridge',
    95: 'building',
    96: 'bush',
    97: 'cabinet',
    98: 'cage',
    99: 'cardboard',
    100: 'carpet',
    101: 'ceiling',
    102: 'tile ceiling',
    103: 'cloth',
    104: 'clothes',
    105: 'clouds',
    106: 'counter',
    107: 'cupboard',
    108: 'curtain',
    109: 'desk',
    110: 'dirt',
    111: 'door',
    112: 'fence',
    113: 'marble floor',
    114: 'floor',
    115: 'stone floor',
    116: 'tile floor',
    117: 'wood floor',
    118: 'flower',
    119: 'fog',
    120: 'food',
    121: 'fruit',
    122: 'furniture',
    123: 'grass',
    124: 'gravel',
    125: 'ground',
    126: 'hill',
    127: 'house',
    128: 'leaves',
    129: 'light',
    130: 'mat',
    131: 'metal',
    132: 'mirror',
    133: 'moss',
    134: 'mountain',
    135: 'mud',
    136: 'napkin',
    137: 'net',
    138: 'paper',
    139: 'pavement',
    140: 'pillow',
    141: 'plant',
    142: 'plastic',
    143: 'platform',
    144: 'playingfield',
    145: 'railing',
    146: 'railroad',
    147: 'river',
    148: 'road',
    149: 'rock',
    150: 'roof',
    151: 'rug',
    152: 'salad',
    153: 'sand',
    154: 'sea',
    155: 'shelf',
    156: 'sky',
    157: 'skyscraper',
    158: 'snow',
    159: 'solid',
    160: 'stairs',
    161: 'stone',
    162: 'straw',
    163: 'structural',
    164: 'table',
    165: 'tent',
    166: 'textile',
    167: 'towel',
    168: 'tree',
    169: 'vegetable',
    170: 'brick wall',
    171: 'concrete wall',
    172: 'wall',
    173: 'panel wall',
    174: 'stone wall',
    175: 'tile wall',
    176: 'wood wall',
    177: 'water',
    178: 'waterdrops',
    179: 'blind window',
    180: 'window',
    181: 'wood',
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
    orig_label = mmcv.imread(label_filepath)
    orig_label = orig_label[:, :, 0]
    orig_label = orig_label.astype(NP_TYPE)

    # Empty array with all elements set as 'ignore id' label
    H, W = orig_label.shape
    new_label = ignore_id * np.ones((H, W), dtype=NP_TYPE)

    # Find valid indices to mask
    orig_idxs = list(np.unique(orig_label))
    all_idxs = list(IDX2TXT_COCOSTUFF.keys())
    idxs = list(set(orig_idxs).intersection(all_idxs))
    for idx in idxs:
        mask = (orig_label == idx)

        # Get uniqe idx representing semantic txt
        idx_star = txt2idx_star[IDX2TXT_COCOSTUFF[idx]]
        new_label[mask] = idx_star

    # Save new vision-language text idx label as
    label_filepath = modify_label_filename(label_filepath)
    np.savez_compressed(label_filepath, new_label)


def parse_args():
    parser = argparse.ArgumentParser(
        description=\
        'Convert COCO Stuff 164k annotations to mmsegmentation format')  # noqa
    parser.add_argument('coco_path',
                        help='Path to \'stuffthingmaps_trainval2017\'')
    parser.add_argument('-o', '--out_dir', help='output path')
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
    args = parse_args()
    coco_path = args.coco_path
    nproc = args.nproc

    # out_dir = args.out_dir or coco_path
    # out_img_dir = osp.join(out_dir, 'images')
    # out_mask_dir = osp.join(out_dir, 'annotations')

    # mmcv.mkdir_or_exist(osp.join(out_dir, 'train2017'))
    # mmcv.mkdir_or_exist(osp.join(out_dir, 'val2017'))

    #######################################
    #  Add text descriptions to register
    #######################################
    txt2idx_star = load_register(args.register_path)
    for txt in IDX2TXT_COCOSTUFF.values():
        txt2idx_star = add_entry(txt2idx_star, txt)
    save_register(args.register_path, txt2idx_star)

    # if out_dir != coco_path:
    #     shutil.copytree(osp.join(coco_path, 'images'), out_img_dir)

    train_list = glob(osp.join(coco_path, 'train2017', '*.png'))
    test_list = glob(osp.join(coco_path, 'val2017', '*.png'))
    assert (len(train_list) +
            len(test_list)) == COCO_LEN, 'Wrong length of list {} & {}'.format(
                len(train_list), len(test_list))

    ####################
    #  Convert labels
    ####################
    train_tasks = [(path, txt2idx_star) for path in train_list]
    test_tasks = [(path, txt2idx_star) for path in test_list]

    if args.nproc > 1:
        mmcv.track_parallel_progress(convert_label_to_idx_star,
                                     train_tasks,
                                     nproc=nproc)
        mmcv.track_parallel_progress(convert_label_to_idx_star,
                                     test_tasks,
                                     nproc=nproc)
    else:
        mmcv.track_progress(convert_label_to_idx_star, train_tasks)
        mmcv.track_progress(convert_label_to_idx_star, test_tasks)

    print('Done!')


if __name__ == '__main__':
    main()