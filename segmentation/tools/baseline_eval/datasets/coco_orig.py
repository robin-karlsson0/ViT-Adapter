import numpy as np

from .coco import COCOStuff


class COCOStuffOrig(COCOStuff):
    '''
    Returns (img, ann) samples with idx_star indexing text labels.
    '''

    def __init__(self,
                 root_dir: str,
                 split: str,
                 target_size: int,
                 img_transform=None):
        super().__init__(root_dir, split, target_size, img_transform)

    @staticmethod
    def get_clss_and_rgbs():
        classes = (
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'street sign', 'stop sign', 'parking meter', 'bench', 'bird',
            'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
            'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
            'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush', 'hair brush', 'banner', 'blanket', 'branch',
            'bridge', 'building', 'bush', 'cabinet', 'cage', 'cardboard',
            'carpet', 'ceiling', 'tile ceiling', 'cloth', 'clothes', 'clouds',
            'counter', 'cupboard', 'curtain', 'desk', 'dirt', 'door', 'fence',
            'marble floor', 'floor', 'stone floor', 'tile floor', 'wood floor',
            'flower', 'fog', 'food', 'fruit', 'furniture', 'grass', 'gravel',
            'ground', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
            'mirror', 'moss', 'mountain', 'mud', 'napkin', 'net', 'paper',
            'pavement', 'pillow', 'plant', 'plastic', 'platform',
            'playingfield', 'railing', 'railroad', 'river', 'road', 'rock',
            'roof', 'rug', 'salad', 'sand', 'sea', 'shelf', 'sky',
            'skyscraper', 'snow', 'solid', 'stairs', 'stone', 'straw',
            'structural', 'table', 'tent', 'textile', 'towel', 'tree',
            'vegetable', 'brick wall', 'concrete wall', 'wall', 'panel wall',
            'stone wall', 'tile wall', 'wood wall', 'water', 'waterdrops',
            'blind window', 'window', 'wood')
        rgbs = []
        for _ in range(len(classes)):
            rgb = np.random.randint(0, 256, size=3)
            rgb = list(rgb)
            rgbs.append(rgb)
        return classes, rgbs
