import clip
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


class RegionProposalCLIPModel():
    '''
    Region Proposal + CLIP model for converting images into general
    vision-language embedding maps.

    How to use:
        model := RegionProposalCLIPModel(rp_model_type, rp_ckpt_path, device)
        emb_map := model.forward(img_path)

    Ref: https://github.com/concept-fusion/concept-fusion/blob/main/examples/extract_conceptfusion_features.py 
    '''

    def __init__(self, rp_model_type: str, ckpt_path: str, device: str):
        '''
        Args:
            rp_model_type: String specifying region proposal model.
            ckpt_path: Path to region proposal model checkpoint file.
            device:
        '''

        # Region proposal model
        sam = sam_model_registry[rp_model_type](checkpoint=ckpt_path)
        sam.to(device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=8,
            pred_iou_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

        # Vision-langauge embedding model
        out = clip.load('ViT-L/14@336px', device)
        self.clip_model = out[0]
        self.clip_preprocess = out[1]

    def region_props(self, img: np.array) -> list:
        '''
        Args:
            img: uint8 RGB image (e.g. cv2.cvtColor(cv2.imread(), BGR2RGB)).
        
        Returns:
            Mask proposals as list of dicts.
        '''
        masks = self.mask_generator.generate(img)
        return masks

    def clip_emb(self, img: Image) -> torch.tensor:
        '''
        Args:
            img: Image opened as PIL.Image.open(img_path)

        Returns:
            CLIP embedding (1, D).
        '''
        img = self.clip_preprocess(img)
        img = img.unsqueeze(0).cuda()
        emb = self.clip_model.encode_image(img)
        return emb

    def mask_clip_emb(self, img, mask: dict) -> tuple:
        '''
        Args:
            img: uint8 RGB image (e.g. cv2.cvtColor(cv2.imread(), BGR2RGB)).
            mask: Region proposal generated mask object.

        Returns:
            roi_emb: CLIP embedding for bounding box spanning the mask.
            nonzero_inds: (N, 2) matrix w. masked pixels (i,j) as row vectors.
        '''
        _x, _y, _w, _h = tuple(mask["bbox"])  # xywh bounding box
        # NOTE Sometimes indices can be floats
        _x = int(_x)
        _y = int(_y)
        _w = int(_w)
        _h = int(_h)
        # seg = masks[maskidx]["segmentation"]
        nonzero_inds = torch.argwhere(torch.from_numpy(mask["segmentation"]))
        # NOTE Image is (H, W, 3). In SAM output, y coords are along height, x along width
        img_roi = img[_y:_y + _h, _x:_x + _w, :]
        img_roi = Image.fromarray(img_roi)
        roi_emb = self.clip_emb(img_roi)

        return roi_emb, nonzero_inds

    def forward(self, img: np.array) -> torch.tensor:
        '''Returns an embedding map with CLIP embeddings from region proposals.
        
        Summary:
            1) Generate region proposals
                {mask} := region_proposal()

            2) Generate CLIP embeddings for image crops
                {mask} --> {img crops}
                {emb} := CLIP({img crops})

            3) Add embeddings to masked regions
                emb_map := add({mask}, {emb})

        Args:
            img: uint8 RGB image (e.g. cv2.cvtColor(cv2.imread(), BGR2RGB)).
 
        Returns:
            VL embedding map (D, H, W) as float tensor.
        '''
        img_h, img_w = img.shape[0], img.shape[1]

        # Region proposals: list of dicts
        masks = self.region_props(img)

        # Local embeddings
        feat_per_roi = []
        roi_nonzero_inds = []
        for mask_idx in range(len(masks)):
            mask = masks[mask_idx]

            out = self.mask_clip_emb(img, mask)
            roi_emb = out[0]  # (1, D)
            nonzero_inds = out[1]  # (N, 2)
            roi_emb = F.normalize(roi_emb)

            feat_per_roi.append(roi_emb)
            roi_nonzero_inds.append(nonzero_inds)

        # Generate embedding map
        feat_dim = feat_per_roi[0].shape[-1]
        emb_map = torch.zeros(img_h, img_w, feat_dim, dtype=torch.half)

        for mask_idx in range(len(masks)):
            mask_px_is = roi_nonzero_inds[mask_idx][:, 0]  # (N)
            mask_px_js = roi_nonzero_inds[mask_idx][:, 1]  # (N)
            mask_emb = feat_per_roi[mask_idx][0].detach().cpu().half()  # (D)
            emb_map[mask_px_is, mask_px_js] += mask_emb

            # Normalize masked elements iteratively to reduce memory usage
            mask_emb_norm = emb_map[mask_px_is, mask_px_js]
            mask_emb_norm = F.normalize(mask_emb_norm.float(), dim=-1)
            emb_map[mask_px_is, mask_px_js] = mask_emb_norm.half()

        emb_map = torch.permute(emb_map, (2, 0, 1)).float()
        return emb_map  # (D, H, W)
