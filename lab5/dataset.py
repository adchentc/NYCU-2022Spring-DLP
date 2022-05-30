import os
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


default_transform = transforms.Compose([
        transforms.ToTensor(),
    ])


class bair_robot_pushing_dataset(Dataset):
    def __init__(self, args, mode='train', transform=default_transform):
        assert mode == 'train' or mode == 'test' or mode == 'validate'
        self.root_dir = args.data_root 
        self.frame_num = 30
        if mode == 'train':
            self.data_dir = f'{self.root_dir}/train'
            self.ordered = False
        else:
            self.data_dir = f'{self.root_dir}/{mode}'
            self.ordered = True 
        self.dirs = []
        for d1 in os.listdir(self.data_dir):
            for d2 in os.listdir(f'{self.data_dir}/{d1}'):
                self.dirs.append(f'{self.data_dir}/{d1}/{d2}')
        self.seq_len = args.n_past + args.n_future
        self.seed_is_set = False
        self.d = 0
        self.cur_dirs = ''
        self.transform = transform
        
    def set_seed(self, seed):
        if not self.seed_is_set:
            self.seed_is_set = True
            np.random.seed(seed)
            
    def __len__(self):
        return len(self.dirs)
        
    def get_seq(self):
        if self.ordered:
            self.cur_dirs = self.dirs[self.d]
            if self.d == len(self.dirs) - 1:
                self.d = 0
            else:
                self.d += 1
        else:
            self.cur_dirs = self.dirs[np.random.randint(len(self.dirs))]
        image_seq = []
        for i in range(self.frame_num):
            fname = f'{self.cur_dirs}/{i}.png'
            im = self.transform(Image.open(fname)).reshape((1, 3, 64, 64))
            image_seq.append(im)
        image_seq = torch.Tensor(np.concatenate(image_seq, axis=0))
        return image_seq

    def get_csv(self):
        cond_seq = []
        actions = [row for row in csv.reader(open(os.path.join(f'{self.cur_dirs}/actions.csv'), newline=''))]
        positions = [row for row in csv.reader(open(os.path.join(f'{self.cur_dirs}/endeffector_positions.csv'), newline=''))]
        for i in range(self.frame_num):
            concat = actions[i]
            concat.extend(positions[i])
            cond_seq.append(concat)
        cond_seq = torch.Tensor(np.array(cond_seq, dtype=float))
        return cond_seq
    
    def __getitem__(self, index):  
        self.set_seed(index)
        seq = self.get_seq()
        cond =  self.get_csv()
        return seq, cond