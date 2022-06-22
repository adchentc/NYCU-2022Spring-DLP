import os
import json
import torch
from argparse import ArgumentParser
from evaluator import evaluation_model
from torchvision.utils import save_image
from model import Generator, weights_init


def get_conditions(path):
    with open(os.path.join('dataset', 'objects.json'), 'r') as file:
        num_conditions = json.load(file)
    with open(path, 'r') as file:
        conditions_list = json.load(file)

    labels = torch.zeros(len(conditions_list),len(num_conditions))

    for i in range(len(conditions_list)):
        for condition in conditions_list[i]:
            labels[i,int(num_conditions[condition])] = 1.

    return labels


if __name__=='__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = ArgumentParser()
    parser.add_argument('--nz', type=int, default=100, help='Size of z latent vector')
    parser.add_argument('--num_conditions', type=int, default=24, help='Number of condition')
    parser.add_argument('--nc', type=int, default=300)
    parser.add_argument('--ngf', type=int, default=128, help='Size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=128, help='Size of feature maps in discriminator')
    parser.add_argument('--img_size', type=int, default=64, help='Size of image')
    parser.add_argument('--exp_name', type=str, default='tmp')
    parser.add_argument('--json', type=str, default='./dataset/new_test.json')
    parser.add_argument('--weights', type=str, default='./models/best/epoch151-acc0.861.pth')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of inference smaple')
    args = parser.parse_args()

    conditions = get_conditions(args.json).to(device)

    G = Generator(args).to(device)
    G.apply(weights_init)
    G.load_state_dict(torch.load(args.weights))

    avg_score = 0
    for _ in range(args.num_samples):
        z = torch.normal(torch.zeros((len(conditions), args.nz)), torch.ones((len(conditions), args.nz))).to(device)
        gen_images = G(z, conditions)
        eval_model = evaluation_model()
        score = eval_model.eval(gen_images, conditions)
        print(f'score: {score:.3f}')
        avg_score += score

    avg_score = avg_score / args.num_samples
    filename = args.json.split('./dataset/')[1].split('.json')[0]
    save_image(gen_images, f'{filename}_eval-{avg_score:.3f}.png', nrow=8, normalize=True)
    print(f'avg score: {avg_score:.3f}')