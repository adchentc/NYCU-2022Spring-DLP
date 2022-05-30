import os
import torch
import utils
import random
import argparse
import progressbar
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import bair_robot_pushing_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=24, type=int, help='batch size')
    parser.add_argument('--data_root', default='./data/processed_data', help='root directory for data')
    parser.add_argument('--model_path', default='/adc/NYCU/DLP/lab5/logs/fp', help='path to model')
    parser.add_argument('--log_dir', default='./logs/fp', help='directory to save generations to')
    parser.add_argument('--seed', default=1, type=int, help='manual seed')
    parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
    parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
    parser.add_argument('--num_threads', type=int, default=0, help='number of data loading threads')
    parser.add_argument('--nsample', type=int, default=100, help='number of samples')
    parser.add_argument('--N', type=int, default=24, help='number of samples')
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--z_dim', type=int, default=64, help='dimensionality of z_t')
    parser.add_argument('--g_dim', type=int, default=128, help='dimensionality of encoder output vector and decoder input vector')
    parser.add_argument('--exp_name', type=str, default='a')
    parser.add_argument('--model_weights', type=str, default='model_246.pth')

    parser.add_argument('--tfr', type=float, default=1.0, help='teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_start_decay_epoch', type=int, default=150, help='The epoch that teacher forcing ratio become decreasing')
    parser.add_argument('--tfr_decay_step', type=float, default=0.001, help='The decay step size of teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--tfr_lower_bound', type=float, default=0.8, help='The lower bound of teacher forcing ratio for scheduling teacher forcing ratio (0 ~ 1)')
    parser.add_argument('--kl_anneal_cyclical', default=False, action='store_true', help='use cyclical mode')
    parser.add_argument('--kl_anneal_ratio', type=float, default=0.0, help='The decay ratio of kl annealing')
    parser.add_argument('--kl_anneal_cycle', type=int, default=4, help='The number of cycle for kl annealing (if use cyclical mode)')
    parser.add_argument('--last_frame_skip', action='store_true', help='if true, skip connections go between frame t and frame t+t rather than last ground truth frame')

    args = parser.parse_args()
    return args


def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]


def normalize_data(dtype, sequence):
    sequence.transpose_(0, 1)
    sequence.transpose_(3, 4).transpose_(2, 3)


    return sequence_input(sequence, dtype)


def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = normalize_data(dtype, sequence)
            yield batch 


def make_gifs(x, cond, modules, idx, name, args):
    # get approx posterior sample
    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    modules['posterior'].hidden = modules['posterior'].init_hidden()
    modules['prior'].hidden = modules['prior'].init_hidden()
    posterior_gen = []
    posterior_gen.append(x[0])
    h_seq = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]
    for i in range(1, args.n_past+args.n_future):
        h_target = h_seq[i][0].detach()
        if args.last_frame_skip or i < args.n_past:	
            h, skip = h_seq[i-1]
        else:
            h, _ = h_seq[i-1]
        h = h.detach()
        if i < args.n_past:
            _, mu, _ = modules['posterior'](h_target)
            modules['frame_predictor'](torch.cat([h, mu, cond[i-1]], 1)) 
            posterior_gen.append(x[i])
        else:
            _, mu_p, _ = modules['prior'](h)
            h_pred = modules['frame_predictor'](torch.cat([h, mu_p, cond[i-1]], 1)).detach()
            x_pred = modules['decoder']([h_pred, skip]).detach()
            h_seq[i] = modules['encoder'](x_pred)
            posterior_gen.append(x_pred)
  
    nsample = args.nsample
    psnr = np.zeros((args.batch_size, nsample, args.n_future))
    progress = tqdm(total=nsample)
    all_gen = []
    for s in range(nsample):
        progress.update(s+1)
        gen_seq = []
        gt_seq = []
        modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
        modules['posterior'].hidden = modules['posterior'].init_hidden()
        modules['prior'].hidden = modules['prior'].init_hidden()
        all_gen.append([])
        all_gen[s].append(x[0])
        h_seq = [modules['encoder'](x[i]) for i in range(args.n_past+args.n_future)]
        for i in range(1, args.n_past+args.n_future):
            h_target = h_seq[i][0].detach()
            if args.last_frame_skip or i < args.n_past:	
                h, skip = h_seq[i-1]
            else:
                h, _ = h_seq[i-1]
            h = h.detach()
            if i < args.n_past:
                _, mu, _ = modules['posterior'](h_target)
                modules['frame_predictor'](torch.cat([h, mu, cond[i-1]], 1))
                all_gen[s].append(x[i])
            else:
                _, mu_p, _ = modules['prior'](h)
                h_pred = modules['frame_predictor'](torch.cat([h, mu_p, cond[i-1]], 1)).detach()
                x_pred = modules['decoder']([h_pred, skip]).detach()
                gen_seq.append(x_pred.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())
                all_gen[s].append(x_pred)
        _, _, psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)

    ###### psnr ######
    for i in range(args.batch_size):
        gifs = [ [] for t in range(args.n_eval) ]
        text = [ [] for t in range(args.n_eval) ]
        mean_psnr = np.mean(psnr[i], 1)
        ordered = np.argsort(mean_psnr)
        rand_sidx = [np.random.randint(nsample) for s in range(3)]
        for t in range(args.n_eval):
            # gt 
            gifs[t].append(add_border(x[t][i], 'green'))
            text[t].append('Ground\ntruth')
            #posterior 
            if t < args.n_past:
                color = 'green'
            else:
                color = 'red'
            gifs[t].append(add_border(posterior_gen[t][i], color))
            text[t].append('Approx.\nposterior')
            # best 
            if t < args.n_past:
                color = 'green'
            else:
                color = 'red'
            sidx = ordered[-1]
            gifs[t].append(add_border(all_gen[sidx][t][i], color))
            text[t].append('Best PSNR')
            # random 3
            for s in range(len(rand_sidx)):
                gifs[t].append(add_border(all_gen[rand_sidx[s]][t][i], color))
                text[t].append('Random\nsample %d' % (s+1))

        fname = f'{args.log_dir}/test/{name}/{name}_{idx+i}.gif'
        utils.save_gif_with_text(fname, gifs, text)
    
    return np.array(psnr[:, s, :])


def add_border(x, color, pad=1):
    w = x.size()[1]
    nc = x.size()[0]
    px = Variable(torch.zeros(3, w+2*pad+30, w+2*pad))
    if color == 'red':
        px[0] =0.7 
    elif color == 'green':
        px[1] = 0.7
    if nc == 1:
        for c in range(3):
            px[c, pad:w+pad, pad:w+pad] = x
    else:
        px[:, pad:w+pad, pad:w+pad] = x
    return px


if __name__ == '__main__':
    args = parse_args()
    if args.cuda:
        assert torch.cuda.is_available(), 'CUDA is not available.'
        device = 'cuda'
    else:
        device = 'cpu'

    args.n_eval = args.n_past + args.n_future

    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    dtype = torch.cuda.FloatTensor

    # ---------------- load the models  ----------------
    if args.model_path != '' and args.exp_name != '' and args.model_weights != '':
        os.makedirs(f'{args.log_dir}/test/{args.exp_name}', exist_ok=True)
        model_path = f'{args.model_path}/{args.exp_name}/{args.model_weights}'
        tmp = torch.load(model_path)
        frame_predictor = tmp['frame_predictor']
        posterior = tmp['posterior']
        prior = tmp['prior']
        encoder = tmp['encoder']
        decoder = tmp['decoder']
        frame_predictor.eval()
        posterior.eval()
        prior.eval()
        encoder.eval()
        decoder.eval()
        frame_predictor.batch_size = args.batch_size
        posterior.batch_size = args.batch_size

        args.last_frame_skip = tmp['args'].last_frame_skip
        args.niter = tmp['args'].niter
        args.epoch_size = tmp['args'].epoch_size
        args.tfr_start_decay_epoch = tmp['args'].tfr_start_decay_epoch
        args.tfr_decay_step = tmp['args'].tfr_decay_step
        args.tfr_lower_bound = tmp['args'].tfr_lower_bound
        args.kl_anneal_cyclical = tmp['args'].kl_anneal_cyclical
        args.kl_anneal_ratio = tmp['args'].kl_anneal_ratio
        args.kl_anneal_cycle = tmp['args'].kl_anneal_cycle

        print(args)
    else:
        raise NotImplementedError

    # --------- transfer to gpu ------------------------------------
    frame_predictor.to(device)
    posterior.to(device)
    prior.to(device)
    encoder.to(device)
    decoder.to(device)

    # --------- load a dataset ------------------------------------
    test_data = bair_robot_pushing_dataset(args, 'test')
    test_loader = DataLoader(test_data,
                            num_workers=args.num_threads,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)

    # testing_batch_generator = get_testing_batch()
    test_iterator = iter(test_loader)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'prior': prior,
        'encoder': encoder,
        'decoder': decoder,
    }

    psnr_list = []
    ave_psnr = 0
    for i in range(0, args.N, args.batch_size):
        print(f'Start Number {i} Iterations...')
        try:
            test_seq, test_cond = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_loader)
            test_seq, test_cond = next(test_iterator)

        test_seq = test_seq.permute(1, 0, 2, 3, 4).to(device)
        test_cond = test_cond.permute(1, 0, 2).to(device)
        psnr = make_gifs(test_seq, test_cond, modules, i, args.exp_name, args)
        psnr = np.mean(np.concatenate(psnr))
        ave_psnr += psnr
    ave_psnr = ave_psnr / (args.N//args.batch_size)
    
    print(f'+---------------------------------------------------------+')
    print(f'                        Generate LP                        ')
    print(f'      - Epoch: {args.niter}                                ')
    print(f'      - tfr_start_decay_epoch: {args.tfr_start_decay_epoch}')
    print(f'      - tfr_decay_step: {args.tfr_decay_step}              ')
    print(f'      - tfr_lower_bound: {args.tfr_lower_bound}            ')
    print(f'      - kl_anneal_cyclical: {args.kl_anneal_cyclical}      ')
    print(f'      - kl_anneal_cycle: {args.kl_anneal_cycle}            ')
    print(f'                                                           ')
    print(f'   ---> Best PSNR: {ave_psnr}                              ')
    print(f'+---------------------------------------------------------+')