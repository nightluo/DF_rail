import os
import cv2
import numpy as np
import torch
import imgcrop
import random
import math
from PIL import Image, ImageDraw

from torchvision import transforms
from torch.utils.data import Dataset

import utils

class InpaintDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist = sorted(utils.get_files(opt.baseroot, test=True), key=lambda d:int(d.split('/')[-1].split('.')[0]))
        self.masklist = sorted(utils.get_files(opt.baseroot_mask, test=True), key=lambda d:int(d.split('/')[-1].split('.')[0]))

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # image
        img = cv2.imread(self.imglist[index])
        img_name = self.imglist[index]
        img_name = img_name.split('/')[-1].split('.jpg')[0]
        print(img_name)
        mask = cv2.imread(self.masklist[index])[:, :, 0]
        # find the Minimum bounding rectangle in the mask
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1440, 816))
        mask = cv2.resize(mask, (1440, 816))
        # img = cv2.resize(img, (960, 544))
        # mask = cv2.resize(mask, (960, 544))
        # img = cv2.resize(img, (640, 360))
        # mask = cv2.resize(mask, (640, 360))
        img = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0).unsqueeze(0).contiguous()
        return img, mask, img_name
