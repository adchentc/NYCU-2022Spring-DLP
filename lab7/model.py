import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.rgb = 3
        self.nz = args.nz
        self.nc = args.nc

        self.map= nn.Sequential(
            nn.Linear(args.num_conditions, args.nc),
            nn.ReLU(True))

        self.main = nn.Sequential(  
            nn.ConvTranspose2d(args.nz + args.nc, args.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(args.ngf, self.rgb, 4, 2, 1, bias=False),
            nn.Tanh())

    def forward(self, z, c):
        z = z.reshape(-1, self.nz, 1, 1)
        c = self.map(c).reshape(-1, self.nc, 1, 1)
        x = torch.cat((z, c), dim=1)
        out = self.main(x)
        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.img_size = args.img_size
        self.input_size = 3 if args.dis_mode == 'proj' else 4

        self.map = nn.Sequential(
            nn.Linear(args.num_conditions, args.img_size * args.img_size),
            nn.ReLU(inplace=True))

        self.main = nn.Sequential(
            nn.Conv2d(self.input_size, args.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(args.ndf * 8, args.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True))

        self.adv_layer = nn.Sequential(
            nn.Conv2d(args.ndf * 16, 1, 4, 2, 1, bias=False),
            nn.Sigmoid())

        self.aux_layer = nn.Sequential(
            nn.Conv2d(args.ndf * 16, args.num_conditions, 4, 2, 1, bias=False),
            nn.Sigmoid())

        self.embedded_layer = nn.Linear(args.num_conditions, args.ndf * 16)

    def forward(self, image, c):
        if self.input_size == 3:
            phi_x = self.main(image)
            out = torch.sum(self.adv_layer(phi_x), dim=(2, 3))
            labels = torch.sum(self.aux_layer(phi_x), dim=(2, 3))
            c = self.embedded_layer(c)
            out = out + torch.sum(c * torch.sum(phi_x, dim=(2, 3)), dim=1, keepdim=True)
            return out.reshape(-1), labels
        else:
            c = self.map(c).reshape(-1, 1, self.img_size, self.img_size)
            x = torch.cat((image, c), dim=1)
            phi_x = self.main(x)
            validity = self.adv_layer(phi_x).reshape(-1)
            labels = self.aux_layer(phi_x).squeeze()
            return validity, labels
