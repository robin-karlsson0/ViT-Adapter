import os

import cv2
import numpy as np
from scipy.ndimage import zoom
from torch.utils.data import Dataset


class COCOStuff(Dataset):
    '''
    Returns (img, ann) samples with idx_star indexing text labels.
    '''

    def __init__(self,
                 root_dir: str,
                 split: str,
                 target_size: int,
                 img_transform=None):
        self.root_dir = root_dir
        self.split = split
        self.img_transform = img_transform
        self.img_dir = os.path.join(root_dir, 'imgs', split)
        self.label_dir = os.path.join(root_dir, 'anns', split)
        self.img_list = os.listdir(self.img_dir)
        self.img_list.sort()

        self.target_size = target_size

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        '''
        Returns:
            image_path: String as image file path.
            label: np.array uint32 (H, W) with idx_star indexing text labels.
        '''
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_name = img_name.replace('.jpg', '.npz')
        label_path = os.path.join(self.label_dir, label_name)
        label = np.load(label_path)
        label = label.f.arr_0

        # Resize smallest dimension to target dimension.
        img_h, img_w = img.shape[:2]
        smallest_dim = min(img_h, img_w)
        scale_factor = self.target_size / smallest_dim

        img = cv2.resize(img,
                         None,
                         fx=scale_factor,
                         fy=scale_factor,
                         interpolation=cv2.INTER_AREA)
        label = zoom(label, (scale_factor, scale_factor), order=0)

        return img, label