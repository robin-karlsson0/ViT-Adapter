# Copyright (c) OpenMMLab. All rights reserved.
from .ade20k import ADE20KVLDataset  # noqa: F401,F403
from .adechallenge import ADEChallengeVLDataset  # noqa: F401,F403
from .bdd100k import BDD100KVLDataset  # noqa: F401,F403
from .builder import *  # noqa: F401,F403
from .cityscapes_vl import CityscapesVLDataset  # noqa: F401,F403
from .coco_stuff164k import COCOStuffVLDataset  # noqa: F401,F403
from .comp_sem_cityscapes import CompSemCityscapesDataset  # noqa: F401,F403
from .concat_dataset import ConcatVLDataset  # noqa: F401,F403
from .idd import IDDVLDataset  # noqa: F401,F403
from .mapillary import MapillaryDataset  # noqa: F401,F403
from .mapillary_vistas_vl import \
    MapillaryVistasV1_2VLDataset  # noqa: F401,F403
from .pipelines import *  # noqa: F401,F403
from .potsdam import PotsdamDataset  # noqa: F401,F403
from .sun_rgbd import SUNRGBDVLDataset  # noqa: F401,F403
