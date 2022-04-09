import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from models.EEGNet import EEGNet
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from models.DeepConvNet import DeepConvNet
from torch.utils.data import Dataset, DataLoader


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]


def inference(model, loader):
    avg_acc = 0.0
    model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100

    return avg_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="Choose the model: (0) EEGNet, (1) DeepConvNet", type=int, default=0)
    parser.add_argument("-acti", help="Choose the activation func: (0) ELU, (1) ReLU, (2) LeakyReLU", type=int, default=0)
    args = parser.parse_args()

    batch_size = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_data, test_label = dataloader.read_bci_data()
    test_dataset = BCIDataset(test_data, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if args.acti == 0:
        activation = nn.ELU(alpha=0.001)
        acti_name = 'ELU'
    elif args.acti == 1:
        activation = nn.ReLU()
        acti_name = 'ReLU'
    elif args.acti == 2:
        activation = nn.LeakyReLU(0.001)
        acti_name = 'LeakyReLU'

    if args.m == 0:
        model = EEGNet(activation)
        model_name = 'EEGNet'
    else:
        model = DeepConvNet(activation)
        model_name = 'DeepConvNet'
    
    model.to(device)
    print(f'./weights/{model_name}_{acti_name}.pt')
    model.load_state_dict(torch.load(f'./weights/{model_name}_{acti_name}.pt'))

    avg_acc = inference(model, test_loader)
    # summary(model, (1, 2, 750))
    print(f'+=======================================================+')
    print(f'                        Inference                        ')
    print(f'                                                         ')
    print(f'      - Model: {model_name}                              ')
    print(f'      - Activation Func.: {acti_name}                    ')
    print(f'      - Inference Acc.(%): {avg_acc:.2f}%                ')
    print(f'                                                         ')
    print(f'+=======================================================+')