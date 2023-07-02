from mmseg_custom.datasets.builder import DATASETS
from mmseg_custom.datasets.custom import CustomDataset
import mmcv
from mmcv.utils import print_log
from mmseg.utils import get_root_logger


@DATASETS.register_module()
class CityscapesVLDataset(CustomDataset):
    """Cityscapes vision-language dataset.
    """
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, **kwargs):
        super(CityscapesVLDataset,
              self).__init__(img_suffix='_leftImg8bit.png',
                             seg_map_suffix='_gtFine_vl_emb_idxs.npz',
                             reduce_zero_label=False,
                             **kwargs)


#    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
#                         split):
#        """Load annotation from directory.
#
#        Args:
#            img_dir (str): Path to image directory
#            img_suffix (str): Suffix of images.
#            ann_dir (str|None): Path to annotation directory.
#            seg_map_suffix (str|None): Suffix of segmentation maps.
#            split (str|None): Split txt file. If split is specified, only file
#                with suffix in the splits will be loaded. Otherwise, all images
#                in img_dir/ann_dir will be loaded. Default: None
#
#        Returns:
#            list[dict]: All image info of dataset.
#        """
#
#        img_infos = []
#        if split is not None:
#            with open(split) as f:
#                for line in f:
#                    img_name = line.strip()
#                    img_info = dict(filename=img_name + img_suffix)
#                    if ann_dir is not None:
#                        seg_map = img_name + seg_map_suffix
#                        img_info['ann'] = dict(seg_map=seg_map)
#                    img_infos.append(img_info)
#        else:
#            for img in mmcv.scandir(img_dir, img_suffix, recursive=True):
#                img_info = dict(filename=img)
#                if ann_dir is not None:
#                    seg_map = img.replace(img_suffix, seg_map_suffix)
#                    img_info['ann'] = dict(seg_map=seg_map)
#                img_infos.append(img_info)
#            img_infos = sorted(img_infos, key=lambda x: x['filename'])
#
#        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
#        return img_infos