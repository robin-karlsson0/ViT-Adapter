import argparse
import os

import clip
import numpy as np
import open_clip
import torch
from sentence_transformers import SentenceTransformer

from tools.convert_datasets.txt2idx_star import load_register, save_register

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Preprocess semantic indices to semantic embeddings')
    parser.add_argument('txt2idx_star_dict_path',
                        type=str,
                        help='Path to the pregenerated \'txt2idx_star\' dict.')
    parser.add_argument('output_path',
                        type=str,
                        help='Path to the generated \'idx_star2emb\' dict.')
    parser.add_argument('--output_filename',
                        type=str,
                        default='idx_star2emb.pkl')
    parser.add_argument('--encoder',
                        type=str,
                        default='clip',
                        help='Implemented options: \{clip\}.')
    parser.add_argument('--add_ignore_emb',
                        type=bool,
                        action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    encoder_type = args.encoder
    if encoder_type not in ['clip', 'openclip', 'sbert']:
        print(f'Specified encoder not supported ({encoder_type})')
        exit()

    # Read pregenerated (idx_star, txt) pairs
    txt2idx_star_path = args.txt2idx_star_dict_path
    try:
        txt2idx_star = load_register(txt2idx_star_path)
    except IOError:
        print(f'Cannot read \'txt2idx_star\' path ({txt2idx_star_path})')
        exit()

    # Load text encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if encoder_type == 'clip':
        model, _ = clip.load('ViT-L/14@336px', device)
        encoder = model.encode_text
        tokenize = clip.tokenize
    elif encoder_type == 'openclip':
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-bigG-14', pretrained='laion2b_s39b_b160k')
        model.to(device)
        model.eval()
        encoder = model
        tokenize = open_clip.tokenize
    elif encoder_type == 'sbert':
        model = SentenceTransformer('all-mpnet-base-v2', device=device)
        encoder = model.encode
    else:
        print(f'Invalid encoder type ({encoder_type})')
        exit()

    # Pregenerate embeddings
    idx_star2emb = {}
    for txt, idx in txt2idx_star.items():

        if encoder_type in ['clip', 'openclip']:
            tok = tokenize(txt).to(device)
            with torch.no_grad():
                emb = model.encode_text(tok).float()
                emb /= emb.norm(dim=-1, keepdim=True)
                emb = emb.cpu().float()

        elif encoder_type == 'sbert':
            with torch.no_grad():
                emb = encoder(txt)  # Normalized 768 dim np.float32
                emb = torch.tensor(emb).unsqueeze(0)  # (1, D)

        else:
            print(f'Invalid encoder type ({encoder_type})')
            exit()

        idx_star2emb[idx] = emb

    if args.add_ignore_emb:
        idx = np.iinfo(np.uint32).max
        txt = 'ignore'
        if encoder_type == 'clip':
            txt = torch.tensor(clip.tokenize(txt)).to(device)
            with torch.no_grad():
                emb = encoder(txt)
                emb = emb.cpu().float()
        elif encoder_type == 'sbert':
            with torch.no_grad():
                emb = encoder(txt)  # Normalized 768 dim np.float32
                emb = torch.tensor(emb).unsqueeze(0)  # (1, D)
        else:
            raise Exception

        idx_star2emb[idx] = emb

    idx_star2emb_path = os.path.join(args.output_path, args.output_filename)
    save_register(idx_star2emb_path, idx_star2emb)
