import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def get_CLEVR_data(root_dir, mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_dir, 'train.json')))
        obj = json.load(open(os.path.join(root_dir, 'objects.json')))
        img = list(data.keys())
        label = list(data.values())
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return np.squeeze(img), np.squeeze(label)
    else:
        data = json.load(open(os.path.join(root_dir, 'test.json')))
        obj = json.load(open(os.path.join(root_dir, 'objects.json')))
        label = data
        for i in range(len(label)):
            for j in range(len(label[i])):
                label[i][j] = obj[label[i][j]]
            tmp = np.zeros(len(obj))
            tmp[label[i]] = 1
            label[i] = tmp
        return None, label


class CLEVRDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.root_dir = './dataset'
        self.mode = mode
        self.imgs, self.labels = get_CLEVR_data(self.root_dir, self.mode)
        self.trans = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.mode == 'train':
            print(f'> Found {len(self.imgs)} images...')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if self.mode == 'train':
            img = Image.open(os.path.join(self.root_dir, 'iclevr', self.imgs[index])).convert('RGB')
            img = self.trans(img)
            cond = self.labels[index]
            return img, torch.Tensor(cond)
        else:
            cond = self.labels[index]
            return torch.Tensor(cond)