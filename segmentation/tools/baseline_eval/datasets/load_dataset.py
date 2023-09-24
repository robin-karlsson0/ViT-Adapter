from .coco_cseg import COCOStuffCSeg
from .coco_orig import COCOStuffOrig
from .mapillary_vistas import MapillaryVistasV1_2Dataset


def load_dataset(name: str, path: str, split: str,
                 img_target_size: int) -> tuple:
    '''
    Args:
        name: Dataset name
    '''
    if name == 'mapillary':
        dataset = MapillaryVistasV1_2Dataset(path, split, img_target_size)
    elif name == 'coco':
        dataset = COCOStuffOrig(path, split, img_target_size)
    elif name == 'coco_cseg':
        dataset = COCOStuffCSeg(path, split, img_target_size)
    else:
        raise IOError(f'Dataset not implemented ({name})')

    cls_txts, rgbs = dataset.get_clss_and_rgbs()

    return dataset, cls_txts, rgbs
