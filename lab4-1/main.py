import copy
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
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


def train(model, loader, criterion, optimizer, acti):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    avg_loss_list = []
    # scheduler_1 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    # scheduler_2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    for epoch in range(1, epochs + 1):
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(tqdm(loader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            print(f'Epoch: {epoch}')
            print(f'Loss: {avg_loss}')
            print(f'Training Acc. (%): {avg_acc:3.2f}%')

        # if epoch > 150 and epoch < 300:
        #     scheduler_1.step()
        
        # if epoch > 300:
        #     scheduler_2.step()

        test_acc = test(model, test_loader)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = model.state_dict()
        print(f'Test Acc. (%): {test_acc:3.2f}%')

    torch.save(best_wts, './weights/' + model_name + '_' + acti + 'z.pt')
    return avg_acc_list, avg_loss_list, test_acc_list, best_acc


def test(model, loader):
    avg_acc = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
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
    args = parser.parse_args()

    lr = 0.001
    epochs = 500
    batch_size = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    a = ['ELU', 'ReLU', 'LeakyReLU']
    d = {}

    for i in range(len(a)):
        print('+======================+')
        print('      ' + a[i] + '   ↓↓↓')
        print('+======================+')
        if a[i] == 'ELU':
            activation = nn.ELU(alpha=0.001)
        elif a[i] == 'ReLU':
            activation = nn.ReLU()
        elif a[i] == 'LeakyReLU':
            activation = nn.LeakyReLU(0.001)
        
        if args.m == 0:
            model = EEGNet(activation)
            model_name = 'EEGNet'
        else:
            model = DeepConvNet(activation)
            model_name = 'DeepConvNet'

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

        model.to(device)
        criterion.to(device)

        train_acc_list, train_loss_list, test_acc_list, best_acc = train(model, train_loader, criterion, optimizer, a[i])
        d[a[i]] = [train_acc_list, train_loss_list, test_acc_list, best_acc]

    print(f'+========================================================+')
    print(f'                   Best Test Acc.(%)                      ')
    print(f'      - Model: {model_name}                               ')
    print(f'      - Epoch: {epochs}                                   ')
    print(f'      - learning rate: {lr}                               ')
    print(f'      - activation function:                              ')
    print('\n'.join([f'         - {k} Test Acc.(%): {d[k][-1]:3.2f}%' for k in d]))
    print(f'+========================================================+')

    # _e = np.linspace(1, epochs, epochs)
    # plt.figure()
    # plt.title('Activation function comparision' + '(' + model_name + ')')
    # plt.plot(_e, d['ELU'][0], label='elu_train')
    # plt.plot(_e, d['ELU'][2], label='elu_test')
    # plt.plot(_e, d['ReLU'][0], label='relu_train')
    # plt.plot(_e, d['ReLU'][2], label='relu_test')
    # plt.plot(_e, d['LeakyReLU'][0], label='leaky_relu_train')
    # plt.plot(_e, d['LeakyReLU'][2], label='leaky_relu_test')
    # plt.xlabel('epoch number')
    # plt.ylabel('acc (%)')
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.legend()
    # plt.savefig('./images/' + model_name +'_acc_' + str(epochs) + 'z.png')
    # plt.show()

    # plt.figure()
    # plt.title('Activation function comparision' + '(' + model_name + ')')
    # plt.plot(_e, d['ELU'][1], label='elu')
    # plt.plot(_e, d['ReLU'][1], label='relu')
    # plt.plot(_e, d['LeakyReLU'][1], label='leaky_relu')
    # plt.xlabel('epoch number')
    # plt.ylabel('loss')
    # plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    # plt.legend()
    # plt.savefig('./images/' + model_name +'_loss_' + str(epochs) + 'z.png')
    # plt.show()