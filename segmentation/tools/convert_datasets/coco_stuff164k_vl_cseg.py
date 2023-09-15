# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import random
from glob import glob

import mmcv
import numpy as np

from tools.convert_datasets.txt2idx_star import (add_entry, load_register,
                                                 save_register)

random.seed(14)
np.random.seed(14)

COCO_LEN = 123287

NP_TYPE = np.uint32

LABEL_SUFFIX = "_vl_emb_idxs.npz"

# NOTE: .PNG label indices are SHIFTED -1 from the provided labels
# Ref: https://github.com/nightrome/cocostuff/issues/7
# Ref: https://github.com/nightrome/cocostuff/blob/master/labels.txt
IDX2TXT_COCOSTUFF_LVL_1 = {
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

IDX2TXT_COCOSTUFF_LVL_2 = {
    #: 'unlabeled'
    0: 'person',  #'person', 
    1: 'vehicle',  #'bicycle',
    2: 'vehicle',  #'car',
    3: 'vehicle',  #'motorcycle',
    4: 'vehicle',  #'airplane',
    5: 'vehicle',  #'bus',
    6: 'vehicle',  #'train',
    7: 'vehicle',  #'truck',
    8: 'vehicle',  #'boat',
    9: 'outdoor',  #'traffic light',
    10: 'outdoor',  #'fire hydrant',
    11: 'outdoor',  #'street sign',
    12: 'outdoor',  #'stop sign',
    13: 'outdoor',  #'parking meter',
    14: 'outdoor',  #'bench',
    15: 'animal',  #'bird',
    16: 'animal',  #'cat',
    17: 'animal',  #'dog',
    18: 'animal',  #'horse',
    19: 'animal',  #'sheep',
    20: 'animal',  #'cow',
    21: 'animal',  #'elephant',
    22: 'animal',  #'bear',
    23: 'animal',  #'zebra',
    24: 'animal',  #'giraffe',
    25: 'accessory',  #'hat',
    26: 'accessory',  #'backpack',
    27: 'accessory',  #'umbrella',
    28: 'accessory',  #'shoe',
    29: 'accessory',  #'eye glasses',
    30: 'accessory',  #'handbag',
    31: 'accessory',  #'tie',
    32: 'accessory',  #'suitcase',
    33: 'sports',  #'frisbee',
    34: 'sports',  #'skis',
    35: 'sports',  #'snowboard',
    36: 'sports',  #'sports ball',
    37: 'sports',  #'kite',
    38: 'sports',  #'baseball bat',
    39: 'sports',  #'baseball glove',
    40: 'sports',  #'skateboard',
    41: 'sports',  #'surfboard',
    42: 'sports',  #'tennis racket',
    43: 'kitchen',  #'bottle',
    44: 'kitchen',  #'plate',
    45: 'kitchen',  #'wine glass',
    46: 'kitchen',  #'cup',
    47: 'kitchen',  #'fork',
    48: 'kitchen',  #'knife',
    49: 'kitchen',  #'spoon',
    50: 'kitchen',  #'bowl',
    51: 'food',  #'banana',
    52: 'food',  #'apple',
    53: 'food',  #'sandwich',
    54: 'food',  #'orange',
    55: 'food',  #'broccoli',
    56: 'food',  #'carrot',
    57: 'food',  #'hot dog',
    58: 'food',  #'pizza',
    59: 'food',  #'donut',
    60: 'food',  #'cake',
    61: 'furniture',  #'chair',
    62: 'furniture',  #'couch',
    63: 'furniture',  #'potted plant',
    64: 'furniture',  #'bed',
    65: 'furniture',  #'mirror',
    66: 'furniture',  #'dining table',
    67: 'furniture',  #'window',
    68: 'furniture',  #'desk',
    69: 'furniture',  #'toilet',
    70: 'furniture',  #'door',
    71: 'electronic',  #'tv',
    72: 'electronic',  #'laptop',
    73: 'electronic',  #'mouse',
    74: 'electronic',  #'remote',
    75: 'electronic',  #'keyboard',
    76: 'electronic',  #'cell phone',
    77: 'appliance',  #'microwave',
    78: 'appliance',  #'oven',
    79: 'appliance',  #'toaster',
    80: 'appliance',  #'sink',
    81: 'appliance',  #'refrigerator',
    82: 'appliance',  #'blender',
    83: 'indoor',  #'book',
    84: 'indoor',  #'clock',
    85: 'indoor',  #'vase',
    86: 'indoor',  #'scissors',
    87: 'indoor',  #'teddy bear',
    88: 'indoor',  #'hair drier',
    89: 'indoor',  #'toothbrush',
    90: 'indoor',  #'hair brush',
    91: 'textile',  #'banner',
    92: 'textile',  #'blanket',
    93: 'plant',  #'branch',
    94: 'building',  #'bridge',
    95: 'building',  #'building',
    96: 'plant',  #'bush',
    97: 'furniture',  #'cabinet',
    98: 'structural',  #'cage',
    99: 'raw material',  #'cardboard',
    100: 'textile',  #'carpet',
    101: 'ceiling',  #'ceiling',
    102: 'ceiling',  #'tile ceiling',
    103: 'textile',  #'cloth',
    104: 'textile',  #'clothes',
    105: 'sky',  #'clouds',
    106: 'furniture',  #'counter',
    107: 'furniture',  #'cupboard',
    108: 'textile',  #'curtain',
    109: 'furniture',  #'desk',
    110: 'ground',  #'dirt',
    111: 'furniture',  #'door',
    112: 'structural',  #'fence',
    113: 'floor',  #'marble floor',
    114: 'floor',  #'floor',
    115: 'floor',  #'stone floor',
    116: 'floor',  #'tile floor',
    117: 'floor',  #'wood floor',
    118: 'plant',  #'flower',
    119: 'water',  #'fog',
    120: 'food',  #'food',
    121: 'food',  #'fruit',
    122: 'furniture',  #'furniture',
    123: 'plant',  #'grass',
    124: 'ground',  #'gravel',
    125: 'ground',  #'ground',
    126: 'solid',  #'hill',
    127: 'building',  #'house',
    128: 'plant',  #'leaves',
    129: 'furniture',  #'light',
    130: 'textile',  #'mat',
    131: 'raw material',  #'metal',
    132: 'furniture',  #'mirror',
    133: 'plant',  #'moss',
    134: 'solid',  #'mountain',
    135: 'ground',  #'mud',
    136: 'textile',  #'napkin',
    137: 'structural',  #'net',
    138: 'raw material',  #'paper',
    139: 'ground',  #'pavement',
    140: 'textile',  #'pillow',
    141: 'plant',  #'plant',
    142: 'raw material',  #'plastic',
    143: 'ground',  #'platform',
    144: 'ground',  #'playingfield',
    145: 'structural',  #'railing',
    146: 'ground',  #'railroad',
    147: 'water',  #'river',
    148: 'ground',  #'road',
    149: 'solid',  #'rock',
    150: 'building',  #'roof',
    151: 'textile',  #'rug',
    152: 'food',  #'salad',
    153: 'ground',  #'sand',
    154: 'water',  #'sea',
    155: 'furniture',  #'shelf',
    156: 'sky',  #'sky',
    157: 'building',  #'skyscraper',
    158: 'ground',  #'snow',
    159: 'solid',  #'solid',
    160: 'furniture',  #'stairs',
    161: 'solid',  #'stone',
    162: 'plant',  #'straw',
    163: 'structural',  #'structural',
    164: 'furniture',  #'table',
    165: 'building',  #'tent',
    166: 'textile',  #'textile',
    167: 'textile',  #'towel',
    168: 'plant',  #'tree',
    169: 'plant',  #'vegetable',
    170: 'wall',  #'brick wall',
    171: 'wall',  #'concrete wall',
    172: 'wall',  #'wall',
    173: 'wall',  #'panel wall',
    174: 'wall',  #'stone wall',
    175: 'wall',  #'tile wall',
    176: 'wall',  #'wood wall',
    177: 'water',  #'water',
    178: 'water',  #'waterdrops',
    179: 'window',  #'blind window',
    180: 'window',  #'window',
    181: 'solid',  #'wood',
}

IDX2TXT_COCOSTUFF_LVL_3 = {
    #: 'unlabeled'
    0: 'outdoor',  #'person', 
    1: 'outdoor',  #'bicycle',
    2: 'outdoor',  #'car',
    3: 'outdoor',  #'motorcycle',
    4: 'outdoor',  #'airplane',
    5: 'outdoor',  #'bus',
    6: 'outdoor',  #'train',
    7: 'outdoor',  #'truck',
    8: 'outdoor',  #'boat',
    9: 'outdoor',  #'trafic light'
    10: 'outdoor',  #'fire hydrant',
    11: 'outdoor',  #'street sign',
    12: 'outdoor',  #'stop sign',
    13: 'outdoor',  #'parking meter',
    14: 'outdoor',  #'bench',
    15: 'outdoor',  #'bird',
    16: 'outdoor',  #'cat',
    17: 'outdoor',  #'dog',
    18: 'outdoor',  #'horse',
    19: 'outdoor',  #'sheep',
    20: 'outdoor',  #'cow',
    21: 'outdoor',  #'elephant',
    22: 'outdoor',  #'bear',
    23: 'outdoor',  #'zebra',
    24: 'outdoor',  #'giraffe',
    25: 'outdoor',  #'hat',
    26: 'outdoor',  #'backpack',
    27: 'outdoor',  #'umbrella',
    28: 'outdoor',  #'shoe',
    29: 'outdoor',  #'eye glasses',
    30: 'outdoor',  #'handbag',
    31: 'outdoor',  #'tie',
    32: 'outdoor',  #'suitcase',
    33: 'outdoor',  #'frisbee',
    34: 'outdoor',  #'skis',
    35: 'outdoor',  #'snowboard',
    36: 'outdoor',  #'sports ball',
    37: 'outdoor',  #'kite',
    38: 'outdoor',  #'baseball bat',
    39: 'outdoor',  #'baseball glove',
    40: 'outdoor',  #'skateboard',
    41: 'outdoor',  #'surfboard',
    42: 'outdoor',  #'tennis racket',
    43: 'indoor',  #'bottle',
    44: 'indoor',  #'plate',
    45: 'indoor',  #'wine glass',
    46: 'indoor',  #'cup',
    47: 'indoor',  #'fork',
    48: 'indoor',  #'knife',
    49: 'indoor',  #'spoon',
    50: 'indoor',  #'bowl',
    51: 'indoor',  #'banana',
    52: 'indoor',  #'apple',
    53: 'indoor',  #'sandwich',
    54: 'indoor',  #'orange',
    55: 'indoor',  #'broccoli',
    56: 'indoor',  #'carrot',
    57: 'indoor',  #'hot dog',
    58: 'indoor',  #'pizza',
    59: 'indoor',  #'donut',
    60: 'indoor',  #'cake',
    61: 'indoor',  #'chair',
    62: 'indoor',  #'couch',
    63: 'indoor',  #'potted outdoor',
    64: 'indoor',  #'bed',
    65: 'indoor',  #'mirror',
    66: 'indoor',  #'dining table',
    67: 'indoor',  #'indoor',
    68: 'indoor',  #'desk',
    69: 'indoor',  #'toilet',
    70: 'indoor',  #'door',
    71: 'indoor',  #'tv',
    72: 'indoor',  #'laptop',
    73: 'indoor',  #'mouse',
    74: 'indoor',  #'remote',
    75: 'indoor',  #'keyboard',
    76: 'indoor',  #'cell phone',
    77: 'indoor',  #'microwave',
    78: 'indoor',  #'oven',
    79: 'indoor',  #'toaster',
    80: 'indoor',  #'sink',
    81: 'indoor',  #'refrigerator',
    82: 'indoor',  #'blender',
    83: 'indoor',  #'book',
    84: 'indoor',  #'clock',
    85: 'indoor',  #'vase',
    86: 'indoor',  #'scissors',
    87: 'indoor',  #'teddy bear',
    88: 'indoor',  #'hair drier',
    89: 'indoor',  #'toothbrush',
    90: 'indoor',  #'hair brush',
    91: 'indoor',  #'banner',
    92: 'indoor',  #'blanket',
    93: 'outdoor',  #'branch',
    94: 'outdoor',  #'bridge',
    95: 'outdoor',  #'outdoor',
    96: 'outdoor',  #'bush',
    97: 'indoor',  #'cabinet',
    98: 'outdoor',  #'cage',
    99: 'indoor',  #'cardboard',
    100: 'indoor',  #'carpet',
    101: 'indoor',  #'indoor',
    102: 'indoor',  #'tile indoor',
    103: 'indoor',  #'cloth',
    104: 'indoor',  #'clothes',
    105: 'outdoor',  #'clouds',
    106: 'indoor',  #'counter',
    107: 'indoor',  #'cupboard',
    108: 'indoor',  #'curtain',
    109: 'indoor',  #'desk',
    110: 'outdoor',  #'dirt',
    111: 'indoor',  #'door',
    112: 'outdoor',  #'fence',
    113: 'indoor',  #'marble indoor',
    114: 'indoor',  #'indoor',
    115: 'indoor',  #'stone indoor',
    116: 'indoor',  #'tile indoor',
    117: 'indoor',  #'wood indoor',
    118: 'outdoor',  #'flower',
    119: 'outdoor',  #'fog',
    120: 'indoor',  #'indoor',
    121: 'indoor',  #'fruit',
    122: 'indoor',  #'indoor',
    123: 'outdoor',  #'grass',
    124: 'outdoor',  #'gravel',
    125: 'outdoor',  #'outdoor',
    126: 'outdoor',  #'hill',
    127: 'outdoor',  #'house',
    128: 'outdoor',  #'leaves',
    129: 'indoor',  #'light',
    130: 'indoor',  #'mat',
    131: 'indoor',  #'metal',
    132: 'indoor',  #'mirror',
    133: 'outdoor',  #'moss',
    134: 'outdoor',  #'mountain',
    135: 'outdoor',  #'mud',
    136: 'indoor',  #'napkin',
    137: 'outdoor',  #'net',
    138: 'indoor',  #'paper',
    139: 'outdoor',  #'pavement',
    140: 'indoor',  #'pillow',
    141: 'outdoor',  #'outdoor',
    142: 'indoor',  #'plastic',
    143: 'outdoor',  #'platform',
    144: 'outdoor',  #'playingfield',
    145: 'outdoor',  #'railing',
    146: 'outdoor',  #'railroad',
    147: 'outdoor',  #'river',
    148: 'outdoor',  #'road',
    149: 'outdoor',  #'rock',
    150: 'outdoor',  #'roof',
    151: 'indoor',  #'rug',
    152: 'indoor',  #'salad',
    153: 'outdoor',  #'sand',
    154: 'outdoor',  #'sea',
    155: 'indoor',  #'shelf',
    156: 'outdoor',  #'outdoor',
    157: 'outdoor',  #'outdoorscraper',
    158: 'outdoor',  #'snow',
    159: 'outdoor',  #'outdoor',
    160: 'indoor',  #'stairs',
    161: 'outdoor',  #'stone',
    162: 'outdoor',  #'straw',
    163: 'outdoor',  #'outdoor',
    164: 'indoor',  #'table',
    165: 'outdoor',  #'tent',
    166: 'indoor',  #'indoor',
    167: 'indoor',  #'towel',
    168: 'outdoor',  #'tree',
    169: 'outdoor',  #'vegetable',
    170: 'indoor',  #'brick indoor',
    171: 'indoor',  #'concrete indoor',
    172: 'indoor',  #'indoor',
    173: 'indoor',  #'panel indoor',
    174: 'indoor',  #'stone indoor',
    175: 'indoor',  #'tile indoor',
    176: 'indoor',  #'wood indoor',
    177: 'outdoor',  #'outdoor',
    178: 'outdoor',  #'outdoordrops',
    179: 'indoor',  #'blind indoor',
    180: 'indoor',  #'indoor',
    181: 'outdoor',  #'wood',
}


def concat_idx2txt_lists(txts_1, txts_2):
    concat_txts = []
    concat_txts += txts_1

    for txt in txts_2:
        if txt not in concat_txts:
            concat_txts.append(txt)

    return concat_txts


txts = []
txts = concat_idx2txt_lists(txts, IDX2TXT_COCOSTUFF_LVL_1.values())
txts = concat_idx2txt_lists(txts, IDX2TXT_COCOSTUFF_LVL_2.values())
txts = concat_idx2txt_lists(txts, IDX2TXT_COCOSTUFF_LVL_3.values())
IDX2TXT_COCOSTUFF_ONTOLOGY_TXTS = txts

IDX2TXT_COCOSTUFF_ONTOLOGY_SET = [
    IDX2TXT_COCOSTUFF_LVL_1, IDX2TXT_COCOSTUFF_LVL_2, IDX2TXT_COCOSTUFF_LVL_3
]

# LVL_SAMPLING_PROB = [0.87, 0.12, 0.01]
LVL_SAMPLING_PROB = [1 / 3, 1 / 3, 1 / 3]


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
    all_idxs = list(IDX2TXT_COCOSTUFF_LVL_1.keys())  # Same keys for all dicts
    idxs = list(set(orig_idxs).intersection(all_idxs))
    for idx in idxs:
        mask = (orig_label == idx)

        # Get uniqe idx representing semantic txt
        lvl_idx = np.random.choice([0, 1, 2], p=LVL_SAMPLING_PROB)

        idx_star = txt2idx_star[IDX2TXT_COCOSTUFF_ONTOLOGY_SET[lvl_idx][idx]]
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
                        default='txt2idx_star_cseg_coco.pkl',
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
    for txt in IDX2TXT_COCOSTUFF_ONTOLOGY_TXTS:
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