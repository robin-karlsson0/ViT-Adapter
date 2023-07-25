import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from tools.baseline_eval.models.rp_clip_model import RegionProposalCLIPModel


class ConceptFusionModel(RegionProposalCLIPModel):
    '''
    Region Proposal + CLIP model + ConceptFusion model for converting images
    into general vision-language embedding maps.

    How to use:
        model := ConceptFusionModel(rp_model_type, rp_ckpt_path, device)
        emb_map := model.forward(img_path)

    Ref: https://github.com/concept-fusion/concept-fusion/blob/main/examples/extract_conceptfusion_features.py 
    '''

    def __init__(self, model_type: str, ckpt_path: str, device):
        '''
        Args:
            rp_model_type: String specifying region proposal model.
            ckpt_path: Path to region proposal model checkpoint file.
            device:
        '''
        super().__init__(model_type, ckpt_path, device)

        self.cos_sim = torch.nn.CosineSimilarity()

    def forward(self, img_path: str) -> torch.tensor:
        '''Returns an embedding map with fused CLIP embeddings from region
        proposals.
        
        Summary:
            1) Generate region proposals
                {mask} := region_proposal()

            2) Generate CLIP embeddings for image crops
                {mask} --> {img crops}
                {emb} := CLIP({img crops})

            3) Add embeddings to masked regions
                emb_map := add({mask}, {emb})

        Args:
            img_path: Path to image file
 
        Returns:
            VL embedding map (D, H, W).
        '''
        img_cv2 = cv2.imread(img_path)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_cv2.shape[0], img_cv2.shape[1]

        # Region proposals: list of dicts
        masks = self.region_props(img_cv2)

        # Global embedding
        img_pil = Image.open(img_path)
        with torch.cuda.amp.autocast():
            global_emb = self.clip_emb(img_pil)  # (1, D)
        global_emb = global_emb.half().cuda()
        global_emb = torch.nn.functional.normalize(global_emb)

        # Local embeddings
        feat_per_roi = []
        roi_nonzero_inds = []
        similarity_scores = []
        for mask_idx in range(len(masks)):
            mask = masks[mask_idx]

            roi_emb, nonzero_inds = self.mask_clip_emb(img_cv2, mask)
            roi_emb = torch.nn.functional.normalize(roi_emb)

            feat_per_roi.append(roi_emb)
            roi_nonzero_inds.append(nonzero_inds)
            sim = self.cos_sim(global_emb, roi_emb)
            similarity_scores.append(sim)

        # Embedding fusion
        similarity_scores = torch.cat(similarity_scores)
        softmax_scores = torch.nn.functional.softmax(similarity_scores, dim=0)

        # Generate embedding map
        feat_dim = global_emb.shape[-1]
        emb_map = torch.zeros(img_h, img_w, feat_dim, dtype=torch.half)

        for mask_idx in range(len(masks)):

            # Embedding fusion
            w = softmax_scores[mask_idx]
            w_emb = w * global_emb + (1 - w) * feat_per_roi[mask_idx]
            w_emb = torch.nn.functional.normalize(w_emb, dim=-1)

            mask_px_is = roi_nonzero_inds[mask_idx][:, 0]  # (N)
            mask_px_js = roi_nonzero_inds[mask_idx][:, 1]  # (N)
            mask_emb = w_emb[0].detach().cpu().half()  # (D)
            emb_map[mask_px_is, mask_px_js] += mask_emb

            # Normalize masked elements iteratively to reduce memory usage
            mask_emb_norm = emb_map[mask_px_is, mask_px_js]
            mask_emb_norm = F.normalize(mask_emb_norm.float(), dim=-1)
            emb_map[mask_px_is, mask_px_js] = mask_emb_norm.half()

        emb_map = torch.permute(emb_map, (2, 0, 1))
        return emb_map  # (D, H, W)
