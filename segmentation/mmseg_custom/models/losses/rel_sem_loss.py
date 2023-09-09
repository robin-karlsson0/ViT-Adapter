import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import LOSSES

from tools.convert_datasets.txt2idx_star import load_register


@LOSSES.register_module(force=True)
class RelativeSemanticLoss(nn.Module):

    def __init__(self,
                 dataset: str,
                 txt2idx_star_path: str = './txt2idx_star.pkl',
                 idx_star2emb_path: str = './idx_star2emb.pkl',
                 objectinfo150_path:
                 str = 'data/ADEChallengeData2016/objectInfo150.txt',
                 temp: float = 0.07,
                 loss_weight: float = 1.,
                 loss_name: str = 'loss_relsem',
                 **kwargs):
        """
        Args:
            margin: Threshold for ignoring dissimilarity distance.
        """
        super(RelativeSemanticLoss, self).__init__()
        self.loss_weight = loss_weight
        self._loss_name = loss_name

        if dataset == 'adechallenge':
            idx_star2cls_idx, cls_embs = self.load_ade_challenge_info(
                txt2idx_star_path, idx_star2emb_path, objectinfo150_path)
        elif dataset == 'coco':
            idx_star2cls_idx, cls_embs = self.load_coco_info(
                txt2idx_star_path, idx_star2emb_path)
        elif dataset == 'bdd100k':
            idx_star2cls_idx, cls_embs = self.load_bdd100k_info(
                txt2idx_star_path, idx_star2emb_path)
        elif dataset == 'idd':
            idx_star2cls_idx, cls_embs = self.load_idd_info(
                txt2idx_star_path, idx_star2emb_path)
        elif dataset == 'sun_rgbd':
            idx_star2cls_idx, cls_embs = self.load_sun_rgbd_info(
                txt2idx_star_path, idx_star2emb_path)
        else:
            raise IOError(f'Given dataset not implemented ({dataset})')

        self.idx_star2cls_idx = idx_star2cls_idx
        self.cls_embs = cls_embs  # (K, D)
        self.num_clss = self.cls_embs.shape[0]
        # Temperature parameter scaling the unnormalized logits distribution
        self.temp = temp

        self.ce = torch.nn.CrossEntropyLoss(ignore_index=255)

    @staticmethod
    def convert_map_idx_star2cls_idx(idx_star_map: torch.tensor,
                                     idx_star2cls_idx: dict) -> torch.tensor:
        '''Converts a idx_star semantic label into a class idx label.

        NOTE Ignored elements are set as 0 and indicated by the ignore mask.

        Args:
            idx_star_map: Semantics encoded by 'idx_star' indices (B, H, W).
            idx_star2cls_idx: Mapping between 'idx_star' --> class indices.

        Returns:
            cls_idx_map: Semantic label with class indices (B, H, W).
        '''
        cls_idx_map = 255 * torch.ones_like(idx_star_map)

        idx_stars = torch.unique(idx_star_map)
        idx_stars = idx_stars.tolist()
        valid_idxs = list(idx_star2cls_idx.keys())
        idx_stars = list(set(idx_stars).intersection(valid_idxs))
        for idx_star in idx_stars:
            # Mask for elements corresponding to a class
            mask = idx_star_map == idx_star
            # Corresponding class index
            cls_idx = idx_star2cls_idx[idx_star]

            cls_idx_map[mask] = cls_idx

        return cls_idx_map[:, 0]

    @staticmethod
    def convert_map_cls_idx2cls_masks(cls_idx_map: torch.tensor,
                                      num_clss: int) -> torch.tensor:
        '''Convert an integer tensor to a batched boolean mask tensor using
        one-hot encoding.

        Args:
            cls_idx_map: Semantic label with class indices (B, H, W).
            num_clss: The number of integers K constituting the set of classes.

        Returns:
            cls_mask_map: Boolean mask tensor (B, K, H, W).
        '''
        B, H, W = cls_idx_map.shape

        # Reshape input tensor to (B, H, W, 1) to be broadcasted with one-hot tensor
        cls_idx_map = cls_idx_map.view(B, H, W, 1)

        # Generate one-hot tensor using scatter method
        device = cls_idx_map.device
        one_hot = torch.zeros(B, H, W, num_clss, device=device)
        one_hot = one_hot.scatter(3, cls_idx_map, 1)
        # Reshape to (B, K, H, W)
        cls_masks = one_hot.permute(0, 3, 1, 2)
        return cls_masks

    def forward(self, pred_embs: torch.tensor,
                idx_star_map: torch.tensor) -> torch.tensor:
        """
        Args:
            pred_embs: (B, D, H, W)
            idx_star_map: Semantics encoded by 'idx_star' indices (B, H, W).

        Returns:
            Avg. loss over all mask avg. cosine embdedding distance losses.
        """

        # Convert label to a class index map (B, H, W)
        cls_idx_map = self.convert_map_idx_star2cls_idx(
            idx_star_map, self.idx_star2cls_idx)

        # Compute semantic similarity scores
        device = pred_embs.device
        sim = F.conv2d(pred_embs, self.cls_embs[:, :, None, None].to(device))
        # NOTE CrossEntropyLoss() requires unnormalized logits
        sim = sim / self.temp
        # sim = F.softmax(sim / self.temp, dim=1)  # (B, K, H, W)

        loss = self.ce(sim, cls_idx_map)

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

    @staticmethod
    def load_ade_challenge_info(
        txt2idx_star_path: str = './txt2idx_star.pkl',
        idx_star2emb_path: str = './idx_star2emb.pkl',
        objectinfo150_path: str = 'data/ADEChallengeData2016/objectInfo150.txt',
    ) -> tuple:
        """Returns mapping and class embeddings for the ADE Challenge dataset.
        """
        txt2idx_star = load_register(txt2idx_star_path)
        idx_star2emb = load_register(idx_star2emb_path)

        # Normalize embedding vectors
        idx_star2emb = {
            key: val / np.linalg.norm(val)
            for key, val in idx_star2emb.items()
        }

        # Parse class names from object infor file
        if not os.path.isfile(objectinfo150_path):
            raise IOError(
                f'Objectinfo file does not exist ({objectinfo150_path})')
        with open(objectinfo150_path, 'r') as f:
            lines = f.readlines()

        cls_txts = []
        for line in lines[1:]:
            # Read first entry and remove possible newline
            cls_txt = line.split('\t')[-1].split(', ')[0]
            cls_txt = cls_txt.replace('\n', '')
            cls_txts.append(cls_txt)

        # Generate class embedding row matrix (K, D)
        cls_embs = []
        for cls_txt in cls_txts:
            idx = txt2idx_star[cls_txt]
            emb = idx_star2emb[idx]
            cls_embs.append(emb)
        cls_embs = torch.cat(cls_embs)

        # Dict for converting labels from 'idx*' maps --> 'class idx' maps
        idx_star2cls_idx = {}
        for cls_idx, cls_txt in enumerate(cls_txts):
            idx_star = txt2idx_star[cls_txt]
            idx_star2cls_idx[idx_star] = cls_idx

        return idx_star2cls_idx, cls_embs

    @staticmethod
    def load_coco_info(txt2idx_star_path: str = './txt2idx_star.pkl',
                       idx_star2emb_path: str = './idx_star2emb.pkl') -> tuple:
        """Returns mapping and class embeddings for the ADE Challenge dataset.
        """
        txt2idx_star = load_register(txt2idx_star_path)
        idx_star2emb = load_register(idx_star2emb_path)

        # Normalize embedding vectors
        idx_star2emb = {
            key: val / np.linalg.norm(val)
            for key, val in idx_star2emb.items()
        }

        cls_txts = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
            'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
            'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
            'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
            'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
            'floor-other', 'floor-stone', 'floor-tile', 'floor-wood', 'flower',
            'fog', 'food-other', 'fruit', 'furniture-other', 'grass', 'gravel',
            'ground-other', 'hill', 'house', 'leaves', 'light', 'mat', 'metal',
            'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
            'paper', 'pavement', 'pillow', 'plant-other', 'plastic',
            'platform', 'playingfield', 'railing', 'railroad', 'river', 'road',
            'rock', 'roof', 'rug', 'salad', 'sand', 'sea', 'shelf',
            'sky-other', 'skyscraper', 'snow', 'solid-other', 'stairs',
            'stone', 'straw', 'structural-other', 'table', 'tent',
            'textile-other', 'towel', 'tree', 'vegetable', 'wall-brick',
            'wall-concrete', 'wall-other', 'wall-panel', 'wall-stone',
            'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
            'window-blind', 'window-other', 'wood'
        ]

        # Generate class embedding row matrix (K, D)
        cls_embs = []
        for cls_txt in cls_txts:
            idx = txt2idx_star[cls_txt]
            emb = idx_star2emb[idx]
            cls_embs.append(emb)
        cls_embs = torch.cat(cls_embs)

        # Dict for converting labels from 'idx*' maps --> 'class idx' maps
        idx_star2cls_idx = {}
        for cls_idx, cls_txt in enumerate(cls_txts):
            idx_star = txt2idx_star[cls_txt]
            idx_star2cls_idx[idx_star] = cls_idx

        return idx_star2cls_idx, cls_embs

    @staticmethod
    def load_bdd100k_info(
            txt2idx_star_path: str = './txt2idx_star.pkl',
            idx_star2emb_path: str = './idx_star2emb.pkl') -> tuple:
        """Returns mapping and class embeddings for the BDD100K dataset.
        """
        txt2idx_star = load_register(txt2idx_star_path)
        idx_star2emb = load_register(idx_star2emb_path)

        # Normalize embedding vectors
        idx_star2emb = {
            key: val / np.linalg.norm(val)
            for key, val in idx_star2emb.items()
        }

        cls_txts = [
            'unlabeled',
            'dynamic',
            'ego vehicle',
            'ground',
            'static',
            'parking',
            'rail track',
            'road',
            'sidewalk',
            'bridge',
            'building',
            'fence',
            'garage',
            'guard rail',
            'tunnel',
            'wall',
            'banner',
            'billboard',
            'lane divider',
            'parking sign',
            'pole',
            'polegroup',
            'street light',
            'traffic cone',
            'traffic device',
            'traffic light',
            'traffic sign',
            'traffic sign frame',
            'terrain',
            'vegetation',
            'sky',
            'person',
            'rider',
            'bicycle',
            'bus',
            'car',
            'caravan',
            'motorcycle',
            'trailer',
            'train',
            'truck',
        ]

        # Generate class embedding row matrix (K, D)
        cls_embs = []
        for cls_txt in cls_txts:
            idx = txt2idx_star[cls_txt]
            emb = idx_star2emb[idx]
            cls_embs.append(emb)
        cls_embs = torch.cat(cls_embs)

        # Dict for converting labels from 'idx*' maps --> 'class idx' maps
        idx_star2cls_idx = {}
        for cls_idx, cls_txt in enumerate(cls_txts):
            idx_star = txt2idx_star[cls_txt]
            idx_star2cls_idx[idx_star] = cls_idx

        return idx_star2cls_idx, cls_embs

    @staticmethod
    def load_idd_info(txt2idx_star_path: str = './txt2idx_star.pkl',
                      idx_star2emb_path: str = './idx_star2emb.pkl') -> tuple:
        """Returns mapping and class embeddings for the IDD dataset.
        """
        txt2idx_star = load_register(txt2idx_star_path)
        idx_star2emb = load_register(idx_star2emb_path)

        # Normalize embedding vectors
        idx_star2emb = {
            key: val / np.linalg.norm(val)
            for key, val in idx_star2emb.items()
        }

        cls_txts = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
            'bicycle'
        ]

        # Generate class embedding row matrix (K, D)
        cls_embs = []
        for cls_txt in cls_txts:
            idx = txt2idx_star[cls_txt]
            emb = idx_star2emb[idx]
            cls_embs.append(emb)
        cls_embs = torch.cat(cls_embs)

        # Dict for converting labels from 'idx*' maps --> 'class idx' maps
        idx_star2cls_idx = {}
        for cls_idx, cls_txt in enumerate(cls_txts):
            idx_star = txt2idx_star[cls_txt]
            idx_star2cls_idx[idx_star] = cls_idx

        return idx_star2cls_idx, cls_embs

    @staticmethod
    def load_sun_rgbd_info(
            txt2idx_star_path: str = './txt2idx_star.pkl',
            idx_star2emb_path: str = './idx_star2emb.pkl') -> tuple:
        """Returns mapping and class embeddings for the SUN RGB-D dataset.
        """
        txt2idx_star = load_register(txt2idx_star_path)
        idx_star2emb = load_register(idx_star2emb_path)

        # Normalize embedding vectors
        idx_star2emb = {
            key: val / np.linalg.norm(val)
            for key, val in idx_star2emb.items()
        }

        cls_txts = [
            'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
            'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds',
            'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror',
            'floor_mat', 'clothes', 'ceiling', 'books', 'fridge', 'tv',
            'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 'person',
            'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag'
        ]

        # Generate class embedding row matrix (K, D)
        cls_embs = []
        for cls_txt in cls_txts:
            idx = txt2idx_star[cls_txt]
            emb = idx_star2emb[idx]
            cls_embs.append(emb)
        cls_embs = torch.cat(cls_embs)

        # Dict for converting labels from 'idx*' maps --> 'class idx' maps
        idx_star2cls_idx = {}
        for cls_idx, cls_txt in enumerate(cls_txts):
            idx_star = txt2idx_star[cls_txt]
            idx_star2cls_idx[idx_star] = cls_idx

        return idx_star2cls_idx, cls_embs
