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

IDX2TXT_COCOSTUFF = {
    0: 'unlabeled',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    12: 'street sign',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    26: 'hat',
    27: 'backpack',
    28: 'umbrella',
    29: 'shoe',
    30: 'eye glasses',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    45: 'plate',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    66: 'mirror',
    67: 'dining table',
    68: 'window',
    69: 'desk',
    70: 'toilet',
    71: 'door',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    83: 'blender',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
    91: 'hair brush',
    92: 'banner',
    93: 'blanket',
    94: 'branch',
    95: 'bridge',
    96: 'building-other',
    97: 'bush',
    98: 'cabinet',
    99: 'cage',
    100: 'cardboard',
    101: 'carpet',
    102: 'ceiling-other',
    103: 'ceiling-tile',
    104: 'cloth',
    105: 'clothes',
    106: 'clouds',
    107: 'counter',
    108: 'cupboard',
    109: 'curtain',
    110: 'desk-stuff',
    111: 'dirt',
    112: 'door-stuff',
    113: 'fence',
    114: 'floor-marble',
    115: 'floor-other',
    116: 'floor-stone',
    117: 'floor-tile',
    118: 'floor-wood',
    119: 'flower',
    120: 'fog',
    121: 'food-other',
    122: 'fruit',
    123: 'furniture-other',
    124: 'grass',
    125: 'gravel',
    126: 'ground-other',
    127: 'hill',
    128: 'house',
    129: 'leaves',
    130: 'light',
    131: 'mat',
    132: 'metal',
    133: 'mirror-stuff',
    134: 'moss',
    135: 'mountain',
    136: 'mud',
    137: 'napkin',
    138: 'net',
    139: 'paper',
    140: 'pavement',
    141: 'pillow',
    142: 'plant-other',
    143: 'plastic',
    144: 'platform',
    145: 'playingfield',
    146: 'railing',
    147: 'railroad',
    148: 'river',
    149: 'road',
    150: 'rock',
    151: 'roof',
    152: 'rug',
    153: 'salad',
    154: 'sand',
    155: 'sea',
    156: 'shelf',
    157: 'sky-other',
    158: 'skyscraper',
    159: 'snow',
    160: 'solid-other',
    161: 'stairs',
    162: 'stone',
    163: 'straw',
    164: 'structural-other',
    165: 'table',
    166: 'tent',
    167: 'textile-other',
    168: 'towel',
    169: 'tree',
    170: 'vegetable',
    171: 'wall-brick',
    172: 'wall-concrete',
    173: 'wall-other',
    174: 'wall-panel',
    175: 'wall-stone',
    176: 'wall-tile',
    177: 'wall-wood',
    178: 'water-other',
    179: 'waterdrops',
    180: 'window-blind',
    181: 'window-other',
    182: 'wood',
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