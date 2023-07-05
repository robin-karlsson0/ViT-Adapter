import argparse
import pickle

import clip
import torch
import os

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
    parser.add_argument('--encoder',
                        type=str,
                        default='clip',
                        help='Implemented options: \{clip\}.')
    args = parser.parse_args()

    encoder_type = args.encoder
    if encoder_type not in ['clip']:
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
    if encoder_type == 'clip':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load('ViT-L/14@336px', device)
        encoder = model.encode_text
    else:
        print(f'Invalid encoder type ({encoder_type})')
        exit()

    # Pregenerate embeddings
    idx_star2emb = {}
    for txt, idx in txt2idx_star.items():
        txt = torch.tensor(clip.tokenize(txt)).to(device)
        with torch.no_grad():
            emb = encoder(txt)
            emb = emb.cpu()
        idx_star2emb[idx] = emb.float()

    idx_star2emb_path = os.path.join(args.output_path, 'idx_star2emb.pkl')
    save_register(idx_star2emb_path, idx_star2emb)
