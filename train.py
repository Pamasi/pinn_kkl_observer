import sys
import os
import argparse

import torch
from torch import nn
import systems
import numpy as np
import random
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import wandb


from loss import *
from typing import TYPE_CHECKING, Optional
from tqdm import trange
from utils.dataset import DataSet
from utils.common import save_ckpt
from neural_network import MainNetwork


from neural_network import MainNetwork

from utils.dataset import DataSet
from utils.common import get_args_parser, config_wandb
from normalizer import Normalizer
from trainer import Trainer


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# to enable reproducibility
torch.backends.cudnn.benchmark = False


os.environ["WANDB_START_METHOD"] = "thread"


def train(net, loss_calc, n_epoch, train_set, optimizer, scheduler,
          device, args: argparse.Namespace, normalizer=None, with_pde=False, pde1=None, wandb_run: wandb.wandb_run.Run = None) -> None:
    """
    Training loop.
    """
    MSE = MSELoss(loss_calc)

    loss_train = 0.0
    loss_best = 0.0

    for epoch in trange(n_epoch):
        loss_epoch = 0.0

        for idx, data in enumerate(train_set):

            # Normal and physics data
            x, z, y, x_ph, y_ph = data

            x, z, y = x.to(device), z.to(device), y.to(device)

            if with_pde:
                x_ph, y_ph = x_ph.to(device), y_ph.to(device)

                assert x_ph.device.type == "cuda", "not cuda as default device"
                assert y_ph.device.type == "cuda", "not cuda as default device"

            optimizer.zero_grad()
            net.mode = 'normal'

            z_hat, x_hat, norm_z_hat, norm_x_hat = net(x)
            if normalizer != None:
                label_x = normalizer.Normalize(
                    x, mode='normal').float()
                label_z = normalizer.Normalize(
                    z, mode='normal').float()
            else:
                label_x = x
                label_z = z

            # Compute MSE loss
            loss_normal = MSE(norm_x_hat, norm_z_hat, label_x, label_z)

            # Compute physics loss
            net.mode = 'physics'
            z_hat_ph = net(x_ph)[0]
            loss_pde1 = pde1(x_ph, y_ph, z_hat_ph)
            # loss_pde2 = pde2(x_ph, z_hat_ph)
            loss = loss_normal + loss_pde1  # + loss_pde2

            loss_epoch += loss
            loss.backward()
            optimizer.step()
            # print(pde1.lagrange)

        # mean loss
        loss_epoch = (loss_epoch / idx).item()

        if scheduler is not None:
            scheduler.step(loss_epoch)

        if epoch % 5 == 0:
            save_ckpt(net, epoch, loss_epoch, scheduler,
                      args.ckpt_dir, optimizer.state_dict())

        if (epoch > 0) and (loss_epoch < loss_best):
            loss_best = loss_epoch

            save_ckpt(net, epoch, loss_epoch, scheduler, args.ckpt_dir,
                      optimizer.state_dict(), save_best=True)

        else:
            loss_best = loss_epoch

        dict_log = {
            "lr": scheduler.get_last_lr()[0],
            "train/loss/mse": loss_epoch,
            "train/loss/pde": loss_pde1
        }

        if args.no_track == False:
            wandb.log(dict_log)

            wandb_run.log_code()

        else:
            print(dict_log)


def experiment(args: argparse.Namespace):
    torch.manual_seed(9)

    # save_dir = args.ckpt_dir
    method = args.method

    # --------------------- System Setup ---------------------

    print('Generating Data.', '\n')
    # set of initial condition among which the LHS is performed
    limits = limits = np.array(
        [[-1000, 1000], [-100, 100], [-1000, 1000], [-100, 100]])    # Sample space

    # parameter for LHS
    a = 0   # start
    b = args.t_sim  # end
    N = args.n_sample      # Number of intervals for runge_kutta4
    num_ic = 50

    if args.add_noise:
        print(
            f'Noise mean:({args.noise_mean})\tvariance:({args.noise_var})')
    if str(args.system).endswith('radar'):
        system = systems.TrackingRadar(
            add_noise=args.add_noise, noise_mean=args.noise_mean, noise_std=math.sqrt(args.noise_var))
        # A and B matricies
        A = np.array(np.diag(-np.arange(1, system.z_size + 1, 1)))

        B = np.ones([system.z_size, system.y_size])

    else:
        raise ValueError('System still not implemented')

    dataset = DataSet(system, A, B, a, b, N, num_ic, limits, seed=args.seed)

    print('Dataset sucessfully generated.', '\n')

    # --------------------- Training Setup ---------------------
    x_size = dataset.system.x_size
    z_size = dataset.system.z_size

    if str(args.activation_fcn) == 'relu':
        activation = F.relu

    else:
        activation = F.mish

    device = torch.device(args.device)

    normalizer = Normalizer(dataset, device)
    main_net = MainNetwork(x_size, z_size, args.n_hidden,
                           args.hidden_size, activation, normalizer)

    optimizer = torch.optim.Adam(main_net.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=args.factor_scheduler,
        patience=args.patiente_scheduler, threshold=args.threshold_scheduler,
        verbose=True)
    loss_fn = nn.MSELoss(reduction='mean')

    # reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ------ Wandb Loggin ------------------------
    if args.no_track == False:
        wandb_run = config_wandb(args)

    # trainer = Trainer(dataset, args.n_epoch, optimizer, main_net,
    #                   loss_fn, args.batch, args.lmbda, method, scheduler=scheduler)

    train_set = torch.utils.data.DataLoader(dataset, args.batch, shuffle=True)

    main_net.to(device)
    loss_calculator = LossCalculator(
        loss_fn, main_net, dataset, device, method)

    pde1 = PdeLoss_xz(dataset.M, dataset.K, dataset.system,
                      loss_calculator, args.lmbda, reduction='mean')

    # we analyze only PINN network
    with_pde = True  # False if method == 'supervised_NN' else True

    print('Device:', device)

    print('Training is starting.', '\n')

    train(main_net, loss_calculator, args.n_epoch, train_set, optimizer,
          scheduler, device, args, normalizer, with_pde, pde1, wandb_run)

    print('Training complete.', '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'PINN-TrackingRadar', parents=[get_args_parser()])
    args = parser.parse_args()

    experiment(args)
