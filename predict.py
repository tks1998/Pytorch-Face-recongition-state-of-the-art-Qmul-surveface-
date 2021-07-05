from scipy.io import loadmat
import albumentations as A
import torch
import torchvision.models as models
import cv2
from albumentations.pytorch import ToTensorV2
from torch import nn
import math
import glob
import sys
import timm
import pandas as pd
from torch.nn.parameter import Parameter
from tqdm import tqdm
import torch.nn.functional as F
from typing import Dict, List, Optional

import faiss
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from .ultis import *

class CFG:
    model_name = 'efficienet_b3'
    original = '' # path train folder
    pretrain_model ="..." # pretrain
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes= 1000 # total class
    input_size = 112 # input size
    ### model config
    use_pretrained = True
    model_name = "efficientnet_b3"
    embedding_size = 512
    dropout = 0.5
    metric = 'arcface' # arcface, cosface , softmax
    use_fc = True
    s = 30
    margin = 0.5
    ls_eps = 0.0
    theta_zero = 0.785
    batch_size = 256
    worker = 2


class Model(nn.Module):

    def __init__(self, n_classes, model_name='efficientnet_b3', use_fc=False, fc_dim=512,\
                 dropout=0.0, metric='softmax', s=30.0, margin=0.50, ls_eps=0.0,\
                 theta_zero=0.785, pretrained=False):
        """
        :param n_classes:
        :param model_name: name of model from pretrainedmodels
            e.g. resnet50, resnext101_32x4d, pnasnet5large
        :param pooling: One of ('SPoC', 'MAC', 'RMAC', 'GeM', 'Rpool', 'Flatten', 'CompactBilinearPooling')
        :param metric: One of ('arcface', 'cosface', 'softmax')
        """
        super(Model, self).__init__()
        print('Building Model Backbone for {} model'.format(model_name))

        self.backbone = timm.create_model(model_name, pretrained=True)
        # self.backbone.load_state_dict(torch.load(CFG.orginal_pretrain_model, map_location=CFG.device))
        final_in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()

        self.pooling = nn.AdaptiveAvgPool2d(1)

        self.use_fc = use_fc
        if use_fc:
            self.dropout = nn.Dropout(p=dropout)
            self.fc = nn.Linear(final_in_features, fc_dim)
            self.bn = nn.BatchNorm1d(fc_dim)
            self._init_params()
            final_in_features = fc_dim

        self.metric = metric
        if metric == 'arcface':
            self.final = ArcMarginProduct(final_in_features, n_classes,
                                          s=s, m=margin, device=CFG.device, easy_margin=False, ls_eps=ls_eps)
        elif metric == 'cosface':
            self.final = AddMarginProduct(final_in_features, n_classes, s=s, m=margin, device=CFG.device)
        elif metric == 'adacos':
            self.final = AdaCos(final_in_features, n_classes, m=margin, theta_zero=theta_zero)
        elif metric == 'adaptive_arcface':
            self.final = Arcface_adaptive_margin(final_in_features, n_classes, s=s, m=margin,\
                                                 device=CFG.device, easy_margin=False, ls_eps=ls_eps)
        else:
            self.final = nn.Linear(final_in_features, n_classes)

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x, label):
        feature = self.extract_feat(x)
        if self.metric in ('arcface', 'cosface', 'adacos','adaptive_arcface'):
            logits = self.final(feature, label)
        else:
            logits = self.final(feature)
        return logits

    def extract_feat(self, x):
        batch_size = x.shape[0]
        x = self.backbone(x)
        x = self.pooling(x).view(batch_size, -1)

        if self.use_fc:
            x = self.dropout(x)
            x = self.fc(x)
            x = self.bn(x)

        return x


class MyFolder(DatasetFolder):

    def __init__(self,root: str,transform = None,target_transform = None,loader = default_loader,is_valid_file= None):
        super(MyFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,\
                                       transform=transform,target_transform=target_transform, is_valid_file=is_valid_file)
        self.imgs = self.samples

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        # sample = self.loader(path)
        sample = cv2.imread(path)
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            sample = self.transform(image=sample)["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

if __name__ == '__main__':

    model = Model(**{
        'n_classes': CFG.num_classes,
        'model_name': CFG.model_name,
        'use_fc': CFG.use_fc,
        'fc_dim': CFG.embedding_size,
        'dropout': CFG.dropout,
        'metric': CFG.metric,
        's': CFG.s,
        'margin': CFG.margin,
        'ls_eps': CFG.ls_eps,
        'theta_zero': CFG.theta_zero,
        'pretrained': False
    })

    if CFG.pretrain_model is not None:
        print("load pretrain")
        checkpoint = torch.load(CFG.pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])

    trans = A.Compose([
        A.Resize(CFG.input_size, CFG.input_size),
        A.Normalize(),
        A.pytorch.transforms.ToTensorV2()
    ])

    trans = A.Compose([
        A.Resize(CFG.input_size, CFG.input_size),
        #     A.CenterCrop(224,224),
        A.Normalize(),
        ToTensorV2()
    ])
    train_loader = MyFolder(eval('CFG.path_' + phase), transform=get_augmentation(phase=phase))

    dataloader = {phase: Data.DataLoader(
        dataset=dataset[phase],
        num_workers=CFG.worker,
        batch_size=CFG.batch_size,
        shuffle=(phase == 'train'),
        pin_memory=(phase == 'train'),
        drop_last=(phase == 'train')
    ) for phase in ['train']}

    model.to(CFG.device)
    model.eval()


