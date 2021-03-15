# dataloader add 3.0 scale
# dataloader add filer text
import numpy as np
from PIL import Image
from torch.utils import data
import util
import cv2
import random
import torchvision.transforms as transforms
import torch
import os

test_root_dir = './data/'
test_data_dir = test_root_dir + 'img_not_use/'
test_gt_dir = test_root_dir + 'txt_not_use/'

random.seed(123456)


def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print(img_path)
        raise
    return img


def scale(img, long_size=2240):
    h, w = img.shape[0:2]
    scale = long_size * 1.0 / max(h, w)
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img


class DataLoader(data.Dataset):
    def __init__(self, part_id=0, part_num=1, long_size=2240, file=None):
        data_dirs = [test_data_dir]
        # print("data_dirs:",data_dirs)

        self.img_paths = []

        if file is not None:
            self.img_paths.extend([file])
        else:
            for data_dir in data_dirs:
                img_names = util.io.ls(data_dir, '.jpg')
                img_names.extend(util.io.ls(data_dir, '.png'))
                img_names = sorted(img_names)
                img_paths = []
                for idx, img_name in enumerate(img_names):
                    img_path = data_dir + img_name
                    img_paths.append(img_path)

                self.img_paths.extend(img_paths)

        part_size = len(self.img_paths) // part_num #  24 张图片
        l = part_id * part_size # 0 * 24
        r = (part_id + 1) * part_size # 1 * 24
        self.img_paths = self.img_paths[l:r]
        self.long_size = long_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path)

        scaled_img = scale(img, self.long_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)

        return img[:, :, [2, 1, 0]], scaled_img

    def scale(self, org_img):
        img = org_img[:, :, [2, 1, 0]]
        scaled_img = scale(img, self.long_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)
        scaled_img = torch.unsqueeze(scaled_img, 0)

        return scaled_img