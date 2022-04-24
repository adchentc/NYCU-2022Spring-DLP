import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn as nn
from torchvision import models
import matplotlib.pyplot as plt
from torchsummary import summary
from ResNet import ResNet50, ResNet18
from dataloader import RetinopathyDataset
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader


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

    cm = confusion_matrix(labels, pred.cpu())
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return avg_acc, cmn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", help="Choose the model: (0) ResNet18, (1) ResNet50", type=int, default=0)
    parser.add_argument("-pretrained", help="Choose the model: (0) No, (1) Yes", type=int, default=0)
    parser.add_argument('-name', help='', type=str, default='my_exp')
    args = parser.parse_args()

    exp_name = args.name
    if args.pretrained == 1:
        pretrained = True
    else:
        pretrained = False

    batch_size = 64
    num_classes = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_dataset = RetinopathyDataset('./dataset', 'test')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
    
    model.to(device)
    model.load_state_dict(torch.load(f'./weights/{exp_name}_{pretrained}.pt'))

    avg_acc, cmn = inference(model, test_loader)
    
    # plt.figure()
    df_cmn = pd.DataFrame(cmn, 
        index = ['No DR', 'Mild', 'Moderate', 'Severe', 'P. DR'],
        columns = ['No DR', 'Mild', 'Moderate', 'Severe', 'P. DR'])
    midpoint = (df_cmn.values.max() - df_cmn.values.min()) / 2
    heatmap = sns.heatmap(df_cmn, annot=True, cmap='Blues', center=midpoint)
    fig = heatmap.get_figure()
    plt.title(f'Confusion Matrix({model_name})')
    plt.xlabel('PREDICTION', fontweight='bold')
    plt.ylabel('GRUOND TRUTH', fontweight='bold')
    plt.yticks(rotation=360)
    plt.tight_layout()
    plt.savefig(f'./images/mtx_{exp_name}_{pretrained}.png')
    plt.show()

    print(f'+=======================================================+')
    print(f'                        Inference                        ')
    print(f'                                                         ')
    print(f'      - Model: {model_name}                              ')
    print(f'      - Pre-trained: {pretrained}                        ')
    print(f'      - Experiment: {exp_name}                           ')
    print(f'      - Inference Acc.(%): {avg_acc:.2f}%                ')
    print(f'                                                         ')
    print(f'+=======================================================+')