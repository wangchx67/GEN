import os
import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import random
from utils.dataset_utils import Augment_RGB_torch

augment = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')]


def make_dataset(path):  # 读取自己的数据的函数

    dataset_list = []
    dirgt = os.path.join(path, 'gt')
    dirimg = os.path.join(path, 'input')

    for fGT in glob.glob(os.path.join(dirgt, '*.*')):
        fName = os.path.basename(fGT)
        dataset_list.append([os.path.join(dirimg, fName), os.path.join(dirgt, fName)])

    return dataset_list


class Traindata(data.Dataset):

    def __init__(self, path, transform=None):  # 初始化文件路進或文件名
        self.train_list = make_dataset(path)
        self.size = 256
        self.transform = transform

        print("Total examples:", len(self.train_list))

    def __getitem__(self, idx):
        img_path, gt_path = self.train_list[idx]

        img = Image.open(img_path)
        gt = Image.open(gt_path)
        img = img.resize((self.size, self.size), Image.ANTIALIAS)
        gt = gt.resize((self.size, self.size), Image.ANTIALIAS)

        img = (np.asarray(img)/255.)
        gt = (np.asarray(gt)/255.)

        img = torch.from_numpy(img).float().permute(2, 0, 1)
        gt = torch.from_numpy(gt).float().permute(2, 0, 1)

        apply_trans = transforms_aug[random.getrandbits(3)]
        img = getattr(augment, apply_trans)(img)
        gt = getattr(augment, apply_trans)(gt)


        return img, gt, img_path, gt_path

    def __len__(self):
        return len(self.train_list)


class Valdata(data.Dataset):

    def __init__(self, path):  # 初始化文件路進或文件名
        self.train_list = make_dataset(path)
        self.size = 256

        print("Total examples:", len(self.train_list))

    def __getitem__(self, idx):
        img_path, gt_path = self.train_list[idx]

        img = Image.open(img_path)
        #img = img.resize((self.size, self.size), Image.ANTIALIAS)
        img = (np.asarray(img) / 255.0)
        img = torch.from_numpy(img).float()

        gt = Image.open(gt_path)
        #gt = gt.resize((self.size, self.size), Image.ANTIALIAS)
        gt_rgb = np.asarray(gt)
        gt = (gt_rgb / 255.0)
        gt = torch.from_numpy(gt).float()

        return img.permute(2, 0, 1), gt.permute(2, 0, 1), img_path, gt_path

    def __len__(self):
        return len(self.train_list)
