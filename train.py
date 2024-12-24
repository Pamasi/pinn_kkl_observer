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
from typing import TYPE_CHECKING, Optional, Tuple
from tqdm import trange
from utils.dataset import DataSet
from utils.common import save_ckpt
from neural_network import EncoderDecoder


from neural_network import EncoderDecoder

from utils.common import get_args_parser, config_wandb
from normalizer import Normalizer


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# to enable reproducibility
torch.backends.cudnn.benchmark = False


os.environ["WANDB_START_METHOD"] = "thread"


def val_step(model, loss_calc, val_loader, device, normalizer=None,
             with_pde=False, pde1=None) -> Tuple[torch.Tensor]:
    """Generate the validation of different episode
    """
    MSE = MSELoss(loss_calc)

    loss_tot = 0.0
    loss_mse = 0.0
    loss_pde = 0.0
    n_batch = len(val_loader)

    model.eval()
    # to reduce memory footprint
    with torch.no_grad():
        for (x, z, y, x_ph, y_ph) in val_loader:

            x, z, y = x.to(device), z.to(device), y.to(device)

            if with_pde:
                x_ph, y_ph = x_ph.to(device), y_ph.to(device)

                # sanity check
                assert x_ph.device.type == "cuda", "not cuda as default device"
                assert y_ph.device.type == "cuda", "not cuda as default device"

            model.mode = 'normal'

            z_hat, x_hat, norm_z_hat, norm_x_hat = model(x)
            if normalizer != None:
                label_x = normalizer.Normalize(
                    x, mode='normal').float()
                label_z = normalizer.Normalize(
                    z, mode='normal').float()
            else:
                label_x = x
                label_z = z

            # Compute MSE loss
            loss_normal_batch = MSE(norm_x_hat, norm_z_hat, label_x, label_z)

            loss_mse += loss_normal_batch
            # Compute physics loss
            model.mode = 'physics'
            z_hat_ph = model(x_ph)[0]
            loss_pde1_batch = pde1(x_ph, y_ph, z_hat_ph)
            # loss_pde2 = pde2(x_ph, z_hat_ph)

            loss_pde += loss_pde1_batch

            loss_batch = loss_normal_batch + loss_pde1_batch  # + loss_pde2

            loss_tot += loss_batch

            # print(pde1.lagrange)

        # mean  batch loss
        loss_tot /= n_batch
        loss_mse /= n_batch
        loss_pde /= n_batch

    return (loss_tot, loss_mse, loss_pde)


def train_step(model, loss_calc, train_loader, optimizer,
               device,  normalizer=None, with_pde=False, pde1=None, clip_norm=0.1) -> Tuple[torch.Tensor]:
    """
    Training loop.
    """
    MSE = MSELoss(loss_calc)

    loss_tot = 0.0
    loss_mse = 0.0
    loss_pde = 0.0
    n_batch = len(train_loader)

    model.train()
    for data in train_loader:
        # Normal and physics data
        x, z, y, x_ph, y_ph = data

        x, z, y = x.to(device), z.to(device), y.to(device)

        if with_pde:
            x_ph, y_ph = x_ph.to(device), y_ph.to(device)

            assert x_ph.device.type == "cuda", "not cuda as default device"
            assert y_ph.device.type == "cuda", "not cuda as default device"

        optimizer.zero_grad()
        model.mode = 'normal'

        z_hat, x_hat, norm_z_hat, norm_x_hat = model(x)
        if normalizer != None:
            label_x = normalizer.Normalize(
                x, mode='normal').float()
            label_z = normalizer.Normalize(
                z, mode='normal').float()
        else:
            label_x = x
            label_z = z

        # Compute MSE loss
        loss_normal_batch = MSE(norm_x_hat, norm_z_hat, label_x, label_z)

        # Compute physics loss
        model.mode = 'physics'
        z_hat_ph = model(x_ph)[0]
        loss_pde1_batch = pde1(x_ph, y_ph, z_hat_ph)
        # loss_pde2 = pde2(x_ph, z_hat_ph)

        loss_mse += loss_normal_batch
        loss_pde += loss_pde1_batch

        loss_batch = loss_normal_batch + loss_pde1_batch  # + loss_pde2

        loss_tot += loss_batch
        loss_batch.backward()

        # velocity too, gradient is clipped
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        # print(pde1.lagrange)

    # mean  batch loss
    loss_tot /= n_batch
    loss_mse /= n_batch
    loss_pde /= n_batch

    return loss_tot, loss_mse, loss_pde


def experiment(args: argparse.Namespace):
    # reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # save_dir = args.ckpt_dir
    method = args.method

    # --------------------- System Setup ---------------------

    print('Generating Data.', '\n')
    # set of initial condition among which the LHS is performed
    limits = np.array(
        [[-1, 1], [-1, 1], [-1, 1], [-1, 1]])    # Sample space

    # parameter for LHS

    # mumber of intervals for runge_kutta4
    # define the sampling time
    n_sample = args.n_sample

    # reduce the number of initial condition, to lower the computational burden
    n_init_cond = args.n_init_cond
    # training simulation time
    # split traning and validation following literature
    t_init_train = 0
    t_end_train = t_init_train + args.t_sim/2

    # validation simulation time
    t_init_val = args.t_sim/2
    t_end_val = args.t_sim

    if args.add_noise:
        print(f'Noise mean:({args.noise_mean})\tvariance:({args.noise_var})')

    if str(args.system).endswith('radar'):
        system = systems.TrackingRadar(
            add_noise=args.add_noise, noise_mean=args.noise_mean, noise_std=math.sqrt(args.noise_var))

    else:
        raise ValueError('System  is still not implemented')

    # A and B matricies
    A = np.array(np.diag(-np.arange(1, system.z_size + 1, 1)))

    B = np.ones([system.z_size, system.y_size])

    # use split based on trajector (2-step episod)
    train_set = DataSet(system, A, B, t_init_train, t_end_train,
                        n_sample, n_init_cond, limits, seed=args.seed, data_gen_mode='backward sim')

    val_set = DataSet(system, A, B, t_init_val, t_end_val,
                      n_sample, n_init_cond, limits, seed=args.seed, data_gen_mode='backward sim')

    print('Dataset sucessfully generated.', '\n')

    # --------------------- Wandb Logging ---------------------
    if args.no_track == False:
        wandb_run, args.ckpt_dir = config_wandb(args)

    # --------------------- Training Setup ---------------------
    x_size = train_set.system.x_size
    z_size = train_set.system.z_size

    if str(args.activation_fcn) == 'relu':
        activation = F.relu

    else:
        activation = F.mish

    device = torch.device(args.device)

    if args.normalize:
        normalizer = Normalizer(train_set, device)

    else:
        normalizer = None

    model = EncoderDecoder(x_size, z_size, args.n_hidden,
                           args.hidden_size, activation, normalizer)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                  factor=args.factor_scheduler,
                                  patience=args.patiente_scheduler,
                                  threshold=args.threshold_scheduler,
                                  verbose=True)

    loss_fn = nn.MSELoss(reduction='mean')

    train_loader = torch.utils.data.DataLoader(
        train_set, args.batch, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, args.batch, shuffle=True)

    model.to(device)
    loss_train = LossCalculator(loss_fn, model, train_set, device, method)
    loss_val = LossCalculator(loss_fn, model, val_set, device, method)

    pde1_train = PdeLoss_xz(train_set.M, train_set.K, train_set.system,
                            loss_train, args.lmbda, reduction='mean')

    pde1_val = PdeLoss_xz(val_set.M, val_set.K, val_set.system,
                          loss_val, args.lmbda, reduction='mean')

    # we analyze only PINN network
    with_pde = True  # False if method == 'supervised_NN' else True

    print('Device:', device)
    print('Training is starting.', '\n')

    loss_min = 0.0

    for epoch in trange(args.n_epoch):
        loss_train_tot, loss_mse_train, loss_pde_train = train_step(
            model, loss_train, train_loader, optimizer, device, normalizer, with_pde, pde1_train)

        (loss_val_tot, loss_mse_val, loss_pde_val) = val_step(
            model, loss_val, val_loader, device, normalizer, with_pde, pde1_val)

        dict_log = {
            "train/loss/total": loss_train_tot.item(),
            "train/loss/mse": loss_mse_train.item(),
            "train/loss/pde": loss_pde_train.item(),
            "val/loss/total": loss_val_tot.item(),
            "val/loss/mse": loss_mse_val.item(),
            "val/loss/pde": loss_pde_val.item()
        }

        if args.no_track == False:
            wandb.log(dict_log)

            wandb_run.log_code()

        else:
            print(dict_log)

        # when to apply scheduling:
        # ref https://discuss.pytorch.org/t/on-which-dataset-learning-rate-scheduler-is-applied/131259
        if scheduler is not None:
            scheduler.step(loss_val_tot)

        # intermidiate saving for crash
        if epoch % 5 == 0:
            save_ckpt(model, epoch, loss_mse_val, optimizer,
                      scheduler, args.ckpt_dir, torch.get_rng_state())

        if (epoch > 0 and loss_mse_val < loss_min) or (epoch == 0):
            loss_min = loss_mse_val

            save_ckpt(model, epoch, loss_mse_val, optimizer,
                      scheduler, args.ckpt_dir, torch.get_rng_state(), save_best=True)

    print('Training is completed.', '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'PINN-Obsever', parents=[get_args_parser()])
    args = parser.parse_args()

    experiment(args)
