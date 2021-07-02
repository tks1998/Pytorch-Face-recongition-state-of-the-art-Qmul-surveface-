# ==================================================== Library ==================================================== #
# sys
import sys
import os
import time
import random
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import glob

# dataloader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision.datasets.folder import *
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset

import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

### import algorithm
import math
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

### visulization
import tensorflow as tf
from matplotlib import pyplot as plt
torch.manual_seed(1234)
### timm module
sys.path.append('./pytorch-image-models')
from . import timm

# Utils
from ultis import *
from margin import *
from optimizer_ import SAM
from adamp import AdamP

class CFG:
    # path
    path_train = "./data/train/"                           # folder train = train/class1, train/class2....
    path_valid = "./data/valid/"                           # folder valid = valid/class1, valid/class2....
    pretrain_model = "../input/no-agu/best_model (27).pt"  # pretrain name
    orginal_pretrain_model = "./pretrain_model/efficientnet_b3_ra2-cf984f9c.pth" # pretrain efficienet

    # dataloader
    input_size = 112
    batch_size = 128
    worker = 32

    # model
    use_pretrained = True
    model_name = "efficientnet_b3"
    num_classes = len(glob.glob(path_train + "*"))
    embedding_size = 512
    train = True

    dropout = 0.5
    metric = 'arcface' # AdaCos,
    use_fc = True
    s = 30
    margin = 0.4
    ls_eps = 0.0
    theta_zero = 0.785

    # optimizer
    SAM = True
    lr = 1e-4
    min_lr = 1e-6
    weight_decay = 1e-6
    scheduler = 'CosineAnnealingWarmRestarts'
    epochs = 100
    T_max = 6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    T_MULT = 2
    T_0 = 5
    gamma = 1.5

    # other parameter
    seed = 42
    print_freq = 100

    train_continue = False
    freq_save = 10
    valid_every = 5
    loss_name = "cross_entropy"  # or focal loss
    use_amp = False

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tensor_log = "./log/"      + model_name + "_" + metric + "_" + loss_name + "_" + scheduler
    path_save = "./out_model/" + model_name + "_" + metric + "_" + loss_name + "_" + scheduler +".pt"

    # make dir
    os.makedirs('log', exist_ok=True)
    os.makedirs('out_model', exist_ok=True)

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

        self.backbone = timm.create_model(model_name, pretrained=False)
        self.backbone.load_state_dict(torch.load(CFG.orginal_pretrain_model, map_location=CFG.device))
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
        if self.metric in ('arcface', 'cosface', 'adacos'):
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

### augmentation

def get_augmentation(phase):
    if phase == "train":
        return  A.Compose([
                    A.Resize(CFG.input_size, CFG.input_size),
                    A.GaussianBlur(),
#                     A.HorizontalFlip(p=0.5),
#                     A.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2)),
#                     A.HueSaturationValue(p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
#                     A.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
#                     A.CoarseDropout(p=0.5),
                    A.Normalize(),
                    ToTensorV2()
                ])
    elif phase in ['test','valid']:
        return A.Compose([
            A.Resize(CFG.input_size, CFG.input_size),
            A.Normalize(),
            ToTensorV2()
        ])

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

    dataset = {
        phase: MyFolder(eval('CFG.path_'+phase), transform = get_augmentation(phase=phase)) for phase in ['train','valid']
    }

    dataloader = {  phase: Data.DataLoader(
                            dataset=dataset[phase],
                            num_workers=CFG.worker,
                            batch_size=CFG.batch_size,
                            shuffle=(phase=='train'),
                            pin_memory = (phase=='train'),
                            drop_last=(phase=='train')
                    ) for phase in ['train','valid'] }

    # Defining Model for specific fold

    model = Model(**{
                'n_classes':CFG.num_classes,
                'model_name':CFG.model_name,
                'use_fc':CFG.use_fc,
                'fc_dim':CFG.embedding_size,
                'dropout':CFG.dropout,
                'metric':CFG.metric,
                's':CFG.s,
                'margin':CFG.margin,
                'ls_eps':CFG.ls_eps,
                'theta_zero':CFG.theta_zero,
                'pretrained':False
            }
    )

    model.to(CFG.device)

    dataset = {
        phase: MyFolder(eval('CFG.path_'+phase),transform = get_augmentation(phase=phase)) for phase in ['train','valid']
    }

    dataloader = {phase: Data.DataLoader( dataset=dataset[phase],
                                          num_workers=CFG.worker,
                                          batch_size=CFG.batch_size,
                                          shuffle=(phase=='train'),
                                          pin_memory = (phase=='train'),
                                          drop_last=(phase=='train') ) for phase in ['train','valid']}

    criterion = FocalLoss(gamma=CFG.gamma) if CFG.loss_name == "focal_loss" else nn.CrossEntropyLoss()
    criterion = criterion.to(CFG.device)

    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    if CFG.SAM=='True':
        bese_optimizer = AdamP
        optimizer = SAM(model.parameters(), bese_optimizer, lr=CFG.lr, weight_decay=CFG.weight_decay)

    if CFG.scheduler=='CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)
    elif CFG.scheduler=='CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=len(dataloader['train']) * CFG.T_0 * CFG.T_MULT,
                T_mult=CFG.T_MULT,
                eta_min=CFG.min_lr,
        )

    if CFG.train_continue==True:
        print("continue train")
        checkpoint = torch.load(CFG.pretrain_model)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        criterion = checkpoint['criterion']
        scheduler = checkpoint['scheduler']
    current_acc = -np.inf

    visualization = tf.summary.create_file_writer(CFG.tensor_log)

    best_loss = np.inf
    with visualization.as_default():
        for epoch in range(0, CFG.epochs+1):
            for phase in ['train','valid']:
                if phase == 'train':
                    loss = train_fn(dataloader['train'], model,criterion, optimizer, CFG.device, scheduler=scheduler, epoch=epoch)
                else:
                    loss = eval_fn(dataloader['valid'], model, criterion, CFG.device)
                tf.summary.scalar('./loss/', loss.avg, step=epoch)
                print('epoch {} loss {}'.format(epoch,loss.avg))
                if phase == 'valid' and best_loss > loss.avg:
                    best_loss = loss.avg
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'criterion': criterion,
                        'scheduler': scheduler
                    }, CFG.path_save)
                    scheduler.step()
