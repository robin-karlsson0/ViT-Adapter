import torch
import torch.nn as nn

from mmseg.models.builder import LOSSES


@LOSSES.register_module(force=True)
class CosineEmbLoss(nn.Module):

    def __init__(self,
                 margin=0.5,
                 reduction='mean',
                 loss_weight=1.,
                 loss_name='loss_ce'):
        """
        Args:
            margin: Threshold for ignoring dissimilarity distance.
        """
        super(CosineEmbLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name

        self.criterion = torch.nn.CosineEmbeddingLoss(margin=margin)

    def forward(self, x1, x2, y):
        """
        Args:
            x1: Embeddings (B, N, D).
            x2: Embeddings (B, N, D).
            y: Target labels (B, N) specifying if each embedding 'n' is suposed to
               be same (+1) or different (-1).
        """
        loss = []
        bs = x1.shape[0]
        for idx in range(bs):
            x1_b = x1[idx]
            x2_b = x2[idx]
            y_b = y[idx]

            # Remove non-labeled elements in x1, x2, y
            nonzero_idxs = torch.nonzero(torch.all(x2_b != 0, dim=1)).squeeze()
            x1_b = x1_b[nonzero_idxs]
            x2_b = x2_b[nonzero_idxs]
            y_b = y_b[nonzero_idxs]

            loss_b = self.loss_weight * self.criterion(x1_b, x2_b, y_b)
            loss.append(loss_b)
        loss = torch.stack(loss)
        loss = torch.mean(loss)
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
