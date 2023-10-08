import os

import clip
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from tools.baseline_eval.models.lseg.lseg_net import LSegNet


class LSegModel():
    '''
    Language-driven Semantic Segmentation model for converting images into
    general vision-language embedding maps.

    How to use:
        model := LSegModel(ckpt_path, device)
        emb_map := model.forward(img_path)

    Checkpoint: lseg_minimal_e200.ckpt

    Ref: https://github.com/krrish94/lseg-minimal
    '''

    def __init__(self, lseg_ckpt_path: str, device: str):

        self.model = LSegNet(backbone="clip_vitl16_384",
                             features=256,
                             crop_size=480,
                             arch_option=0,
                             block_depth=0,
                             activation='lrelu')

        self.device = device

        if not os.path.isfile(lseg_ckpt_path):
            raise IOError('Specified LSeg model checkpoint does not exist')

        self.model.load_state_dict(torch.load(lseg_ckpt_path))
        self.model.eval()
        self.model.to(self.device)

        self.clip_text_encoder = self.model.clip_pretrained.encode_text

    def preproc_img(self, img: np.array) -> torch.tensor:
        '''Preprocesses an image into a tensor input sample.
        
        Args:
            img: uint8 RGB image (e.g. cv2.cvtColor(cv2.imread(), BGR2RGB)).
        '''
        img = cv2.resize(img, (640, 480))
        img = torch.from_numpy(img).float() / 255.0
        img = img[..., :3]  # drop alpha channel, if present
        img = img.to(self.device)
        img = img.permute(2, 0, 1)  # C, H, W
        img = img.unsqueeze(0)  # 1, C, H, W

        return img

    def conv_txt2emb(self, txt: str) -> torch.tensor:
        '''
        Args:
            txt: Text string to evaluate semantic similarity.
        '''
        token = clip.tokenize(txt)
        token = token.to(self.device)
        emb = self.clip_text_encoder(token)
        emb = emb.float().cpu()
        return emb

    def forward(self, img: np.array) -> torch.tensor:
        '''Returns an embedding map with CLIP embeddings from region proposals.

        img --> DPT() --> low-res VL emb map --> NN interp --> VL emb map

        Dense Prediction Transformer (DPT)
          Backbone:
            x --> ViT() --> MS token matrices --> Reassemble() --> MS feat maps
          Neck:
            MS feat maps --> Fusion() --> Feat map (B, 512, 240, 320)
          Decoder head:
            Feat map (256) --> Proj() --> Low-res VL emb map (512)

        Args:
            img: uint8 RGB image (e.g. cv2.cvtColor(cv2.imread(), BGR2RGB)).
 
        Returns:
            VL embedding map (D, H, W) as float tensor.
        '''
        H, W, _ = img.shape
        img = self.preproc_img(img)  # Resizes any image to (B, 3, 640, 480)

        emb_map = self.model.forward(img)
        emb_map = F.normalize(emb_map, dim=1)
        emb_map = F.interpolate(emb_map, size=(H, W), mode='nearest')

        return emb_map[0]  # (512, H, W)
