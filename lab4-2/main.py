import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from ResNet import ResNet50, ResNet18
from torch.utils.data import DataLoader
from matplotlib.ticker import MaxNLocator
from dataloader import RetinopathyDataset

def train(model, loader, criterion, optimizer):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    avg_loss_list = []
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.96)
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


        # scheduler.step()
        test_acc = test(model, test_loader)
        test_acc_list.append(test_acc)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = model.state_dict()
        print(f'Test Acc. (%): {test_acc:3.2f}%')

    torch.save(best_wts, f'./weights/{exp_name}_{pretrained}.pt')
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
    parser.add_argument('-model', help='Choose the model: (0) ResNet18, (1) ResNet50', type=int, default=0)
    parser.add_argument('-name', help='', type=str, default='my_exp')
    args = parser.parse_args()

    exp_name = args.name
    print(f'> Start training experiment ## {exp_name} ##...')

    e = 0
    previous_exp = ''

    lr = 0.001
    epochs = 20
    batch_size = 16
    num_classes = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'> Data will compute with {device} device...')

    root = './dataset'
    train_dataset = RetinopathyDataset(root, 'train')
    test_dataset = RetinopathyDataset(root, 'test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    plot_list = []
    best_acc_list = []

    for i in range(2):
        if i % 2 == 0:
            pretrained = True
        else:
            pretrained = False

        print(f'+=================================+')
        print(f'      pretrained: {pretrained}      ↓↓↓')
        print(f'+=================================+')

        if args.model == 0:
            if pretrained:
                model = models.resnet18(pretrained=pretrained)
                fc_inputs = model.fc.in_features
                model.fc = nn.Linear(fc_inputs, num_classes)
            else:
                model = ResNet18()
            model_name = 'ResNet18'
        else:
            if pretrained:
                model = models.resnet50(pretrained=pretrained)
                fc_inputs = model.fc.in_features
                model.fc = nn.Linear(fc_inputs, num_classes)
            else:
                model = ResNet50
            model_name = 'ResNet50'

        # sample = [20656, 1955, 4210, 698, 581]
        # nweights = [1 - (x / sum(sample)) for x in sample]
        # nweights = torch.FloatTensor(nweights).to(device)
        # criterion = nn.CrossEntropyLoss(weight=nweights)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

        model.to(device)
        criterion.to(device)
        
        '''
        pre-trained
        '''
        # model.load_state_dict(torch.load(f'./weights/{previous_exp}_{pretrained}.pt'))
        # print(f'Keep using pre-trained...name: ./weights/{previous_exp}_{pretrained}.pt')

        train_acc_list, train_loss_list, test_acc_list, best_acc = train(model, train_loader, criterion, optimizer)
        plot_list.append([train_acc_list, train_loss_list, test_acc_list])
        best_acc_list.append(best_acc)

    print(f'+========================================================+')
    print(f'                   Best Test Acc.(%)                      ')
    print(f'      - Model: {model_name}                               ')
    print(f'      - Epoch: {epochs}                                   ')
    print(f'      - learning rate: {lr}                               ')
    print(f'      - w Pretrained Acc.(%): {best_acc_list[0]:3.2f}%    ')
    print(f'      - w/o Pretrained Acc.(%): {best_acc_list[1]:3.2f}%  ')
    print(f'+========================================================+')

    _e = np.linspace(1+e, epochs+e, epochs)
    baseline = np.empty(epochs)
    baseline.fill(82)
    plt.figure()
    plt.title(f'Result Comparison({model_name})')
    plt.plot(_e, plot_list[0][0], color='royalblue', label='train (with pretraining)')
    plt.plot(_e, plot_list[0][2], linestyle=(0, (4, 1, 2, 1)), color='royalblue', label='test (with pretraining)')
    plt.plot(_e, plot_list[1][0], color='limegreen', label='train (w/o pretraining)')
    plt.plot(_e, plot_list[1][2], linestyle=(0, (4, 1, 2, 1)), color='limegreen', label='test (w/o pretraining)')
    plt.plot(_e, baseline, linestyle='--', color='red')
    plt.gca().text(epochs+e+0.2, 82, f'{82}%', color='red')
    plt.xlabel('epoch number')
    plt.ylabel('acc (%)')
    plt.ylim(70, 90)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.savefig(f'./images/acc_{exp_name}.png')
    plt.show()