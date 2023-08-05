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