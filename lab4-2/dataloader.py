import numpy as np
import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision.transforms import transforms as T


def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyDataset(data.Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print(f'> {self.mode}: Found {len(self.img_name)} images...')

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        transform = T.Compose([
            T.RandomRotation(degrees=20),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])
        normalization = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = Image.open(f'{self.root}/{self.img_name[index]}.jpeg')
        if self.mode == 'train':
            img = transform(img)
        img = normalization(img)
        label = self.label[index]
        return img, label