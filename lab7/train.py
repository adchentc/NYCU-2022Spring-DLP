import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import CLEVRDataset
from argparse import ArgumentParser
from evaluator import evaluation_model
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from matplotlib.ticker import MaxNLocator
from model import Generator, Discriminator, weights_init


def plot(eval_acc_list, avg_loss_G_list, avg_loss_D_list, args):
    _e = np.linspace(1, args.epochs, args.epochs)

    plt.figure()
    plt.title('Loss G/Loss D')
    plt.plot(_e, avg_loss_G_list, color='royalblue', label='Generator loss')
    plt.plot(_e, avg_loss_D_list, color='limegreen', label='Discriminator loss')
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./images/{args.exp_name}/loss_curve_{args.exp_name}.png')

    plt.figure()
    plt.title('Scores')
    plt.plot(_e, eval_acc_list, color='crimson', marker='.', linestyle='', label='F1-score')
    plt.xlabel('epoch number')
    plt.ylabel('F1-score')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./images/{args.exp_name}/score_curve_{args.exp_name}.png')


def sample_z(bs, nz, mode='normal'):
    if mode == 'normal':
        return torch.normal(torch.zeros((bs, nz)), torch.ones((bs, nz)))
    elif mode == 'uniform':
        return torch.randn(bs, nz)
    else:
        raise NotImplementedError()


def evaluate(G, loader, eval_model, nz):
    G.eval()
    avg_acc = 0
    gen_images = None
    with torch.no_grad():
        for _, conds in enumerate(loader):
            conds = conds.to(device)
            z = sample_z(conds.shape[0], nz).to(device)
            fake_images = G(z, conds)
            if gen_images is None:
                gen_images = fake_images
            else:
                gen_images = torch.vstack((gen_images, fake_images))
            acc = eval_model.eval(fake_images, conds)
            avg_acc += acc * conds.shape[0]
    avg_acc /= len(loader.dataset)
    return avg_acc, gen_images


def train(G, D, opt_G, opt_D, adversarial_loss, auxiliary_loss, train_loader, test_loader, args):
    eval_model = evaluation_model()
    best_acc = 0
    eval_acc_list = []
    avg_loss_G_list = []
    avg_loss_D_list = []
    scheduler = optim.lr_scheduler.ExponentialLR(opt_G, gamma = 0.96)
    for epoch in range(1, args.epochs + 1):
        G.train()
        D.train()
        loss_Gs = 0
        loss_Ds = 0
        for _, (images, conds) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            conds = conds.to(device)
            bs = images.shape[0]

            real_label_G = torch.ones(bs).to(device)

            real_label_D = torch.normal(torch.ones(bs), torch.ones(bs)*0.01).to(device)
            torch.clamp(real_label_D, max=1)
            fake_label_D = torch.normal(torch.zeros(bs), torch.ones(bs)*0.01).to(device)
            torch.clamp(fake_label_D, min=0)
            if random.random() < 0.1:
                real_label_D, fake_label_D = fake_label_D, real_label_D
          
            # train discriminator
            opt_D.zero_grad()
            outputs, real_aux = D(images, conds)
            loss_real = (adversarial_loss(outputs, real_label_D) + args.gamma * auxiliary_loss(real_aux, conds)) / 2
            z = sample_z(bs, args.nz).to(device)
            fake_images = G(z, conds)
            outputs, fake_aux = D(fake_images.detach(), conds)
            loss_fake = (adversarial_loss(outputs, fake_label_D) + args.gamma * auxiliary_loss(fake_aux, conds)) / 2
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            opt_D.step()

            for _ in range(args.proportion):
                # train generator
                opt_G.zero_grad()
                z = sample_z(bs, args.nz).to(device)
                fake_images = G(z, conds)
                outputs, pred_label = D(fake_images, conds)
                loss_G = (adversarial_loss(outputs, real_label_G) + args.gamma * auxiliary_loss(pred_label, conds)) / 2
                loss_G.backward()
                opt_G.step()

            loss_Gs += loss_G.item()
            loss_Ds += loss_D.item()

        # if epoch > (args.epochs*0.5):
        #     scheduler.step()
        avg_loss_G = loss_Gs / len(train_loader)
        avg_loss_D = loss_Ds / len(train_loader)
        eval_acc, gen_images = evaluate(G, test_loader, eval_model, args.nz)
        gen_images = 0.5*gen_images + 0.5
        eval_acc_list.append(eval_acc)
        avg_loss_G_list.append(avg_loss_G)
        avg_loss_D_list.append(avg_loss_D)
        print(f'[Epoch {epoch}/{args.epochs}] acc: {eval_acc:.3f}, loss G: {avg_loss_G:.3f}, loss D: {avg_loss_D:.3f}')
        save_image(gen_images, os.path.join(f'./images/{args.exp_name}', f'epoch{epoch}.png'), nrow=8)
        if eval_acc > best_acc:
            best_acc = eval_acc
            torch.save(G.state_dict(), os.path.join(f'./models/{args.exp_name}', f'epoch{epoch}-acc{eval_acc:.3f}.pth'))
    plot(eval_acc_list, avg_loss_G_list, avg_loss_D_list, args)


if __name__ == '__main__':

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = ArgumentParser()
    parser.add_argument('--nz', type=int, default=100, help='Size of z latent vector')
    parser.add_argument('--num_conditions', type=int, default=24, help='Number of condition')
    parser.add_argument('--nc', type=int, default=300)
    parser.add_argument('--ngf', type=int, default=128, help='Size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=256, help='Size of feature maps in discriminator')
    parser.add_argument('--img_size', type=int, default=64, help='Size of image')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--dis_mode', type=str, default='proj')
    parser.add_argument('--proportion', type=int, default=4)
    parser.add_argument('--gamma', type=int, default=1, help='Adjust auxiliary loss\'s weights')
    parser.add_argument('--exp_name', type=str, default='tmp')
    args = parser.parse_args()

    os.makedirs(f'./images/{args.exp_name}', exist_ok=True)
    os.makedirs(f'./models/{args.exp_name}', exist_ok=True)

    train_dataset = CLEVRDataset(args=args, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    test_dataset = CLEVRDataset(args=args, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    G = Generator(args).to(device)
    G.apply(weights_init)

    D = Discriminator(args).to(device)
    D.apply(weights_init)

    adversarial_loss = torch.nn.BCEWithLogitsLoss()
    auxiliary_loss = torch.nn.CrossEntropyLoss()
    opt_G = torch.optim.Adam(G.parameters(), args.lr, betas=(args.beta1, args.beta2))
    opt_D = torch.optim.Adam(D.parameters(), args.lr, betas=(args.beta1, args.beta2))

    train(G, D, opt_G, opt_D, adversarial_loss, auxiliary_loss, train_loader, test_loader, args)