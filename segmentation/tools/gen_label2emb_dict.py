import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)
# model, preprocess = clip.load("RN50", device=device)

query = [
    "a picture of a road", "a picture of a road marking",
    "a picture of a sidewalk", "a picture of a car", "a picture of a banana",
    "a picture of a boat"
]

cityscapes_labels = [
    'road',
    ''
]

query = [f'a picture of a {label}' for label in labels]

text = clip.tokenize(query).to(device)

with torch.no_grad():

    txt_embs = model.encode_text(text)  # (B, 1024)

    road_road_marking = torch.sum(txt_embs[0:2, :], dim=0, keepdim=True)
    txt_embs = torch.concat((road_road_marking, txt_embs))

    txt_embs = txt_embs / txt_embs.norm(dim=1, keepdim=True)



labels = [
#name                     id    trainId
'road'              ,    0
'sidewalk'          ,    1
'building'          ,    2
'wall'              ,    3
'fence'             ,    4
'pole'              ,    5
'traffic light'     ,    6
'traffic sign'      ,    7
'vegetation'        ,    8
'terrain'           ,    9
'sky'               ,   10
'person'            ,   11
'rider'             ,   12
'car'               ,   13
'truck'             ,   14
'bus'               ,   15
'train'             ,   16
'motorcycle'        ,   17
'bicycle'           ,   18
]