import os

import cv2
import numpy as np
from scipy.ndimage import zoom
from torch.utils.data import Dataset


class MapillaryVistasV1_2Dataset(Dataset):
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

    @staticmethod
    def get_clss_and_rgbs():
        classes = ('Bird', 'Ground Animal', 'Curb', 'Fence', 'Guard Rail',
                   'Barrier', 'Wall', 'Bike Lane', 'Crosswalk - Plain',
                   'Curb Cut', 'Parking', 'Pedestrian Area', 'Rail Track',
                   'Road', 'Service Lane', 'Sidewalk', 'Bridge', 'Building',
                   'Tunnel', 'Person', 'Bicyclist', 'Motorcyclist',
                   'Other Rider', 'Lane Marking - Crosswalk',
                   'Lane Marking - General', 'Mountain', 'Sand', 'Sky', 'Snow',
                   'Terrain', 'Vegetation', 'Water', 'Banner', 'Bench',
                   'Bike Rack', 'Billboard', 'Catch Basin', 'CCTV Camera',
                   'Fire Hydrant', 'Junction Box', 'Mailbox', 'Manhole',
                   'Phone Booth', 'Pothole', 'Street Light', 'Pole',
                   'Traffic Sign Frame', 'Utility Pole', 'Traffic Light',
                   'Traffic Sign (Back)', 'Traffic Sign (Front)', 'Trash Can',
                   'Bicycle', 'Boat', 'Bus', 'Car', 'Caravan', 'Motorcycle',
                   'On Rails', 'Other Vehicle', 'Trailer', 'Truck',
                   'Wheeled Slow', 'Car Mount', 'Ego Vehicle', 'Unlabeled')
        rgbs = [[165, 42, 42], [0, 192, 0], [196, 196, 196], [190, 153, 153],
                [180, 165, 180], [90, 120, 150], [102, 102, 156],
                [128, 64, 255], [140, 140, 200], [170, 170, 170],
                [250, 170, 160], [96, 96, 96], [230, 150, 140], [128, 64, 128],
                [110, 110, 110], [244, 35, 232], [150, 100, 100], [70, 70, 70],
                [150, 120, 90], [220, 20, 60], [255, 0, 0], [255, 0, 100],
                [255, 0, 200], [200, 128, 128], [255, 255, 255], [64, 170, 64],
                [230, 160, 50], [70, 130, 180], [190, 255,
                                                 255], [152, 251, 152],
                [107, 142, 35], [0, 170, 30], [255, 255, 128], [250, 0, 30],
                [100, 140, 180], [220, 220, 220],
                [220, 128, 128], [222, 40, 40], [100, 170, 30], [40, 40, 40],
                [33, 33, 33], [100, 128, 160], [142, 0, 0], [70, 100, 150],
                [210, 170, 100], [153, 153, 153], [128, 128, 128], [0, 0, 80],
                [250, 170, 30], [192, 192, 192], [220, 220, 0], [140, 140, 20],
                [119, 11, 32], [150, 0, 255], [0, 60, 100], [0, 0, 142],
                [0, 0, 90], [0, 0, 230], [0, 80, 100], [128, 64, 64],
                [0, 0, 110], [0, 0, 70], [0, 0, 192], [32, 32, 32],
                [120, 10, 10], [0, 0, 0]]
        return classes, rgbs
