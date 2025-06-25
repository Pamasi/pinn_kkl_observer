import os
import argparse

import math
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb.wandb_run
import systems
import numpy as np
import random
import math
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import wandb


from loss import *
from typing import TYPE_CHECKING, Optional, Tuple, Dict
from tqdm import trange, tqdm
from utils.dataset import DataSet
from utils.common import save_ckpt
from neural_network import EncoderDecoder

from utils.common import get_args_parser, config_wandb, load_ckpt
from normalizer import Normalizer


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# to enable reproducibility
torch.backends.cudnn.benchmark = False


os.environ["WANDB_START_METHOD"] = "thread"


def val_step(model, loss_calc, val_loader, device, normalizer=None,
             with_pde=False, pde=None, w_enc=None, w_dec=None ) -> Tuple[torch.Tensor]:
    """Generate the validation of different episode
    """
    MSE = MSELoss(loss_calc)

    loss_tot = 0.0
    loss_mse = 0.0
    loss_pde_encoder = 0.0
    loss_pde_decoder = 0.0
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
            loss_pde_enc_batch = pde[0](x_ph, y_ph, z_hat_ph)

            loss_pde_encoder += loss_pde_enc_batch
            loss_batch = loss_normal_batch + w_enc*loss_pde_enc_batch   

            if w_dec is not None:
                loss_pde_dec_batch = pde[1](x_ph, z_hat_ph)
                
                loss_pde_decoder += loss_pde_dec_batch
                loss_batch +=  w_dec*loss_pde_dec_batch

  
            loss_tot += loss_batch

            # print(pde1.lagrange)

        # mean  batch loss
        loss_tot /= n_batch
        loss_mse /= n_batch
        
        loss_pde_encoder /= n_batch

        if w_dec is not None:
            loss_pde_decoder /= n_batch

    return (loss_tot, loss_mse, loss_pde_encoder, loss_pde_decoder)


def train_step(model, loss_calc, train_loader, optimizer, device,  normalizer=None, with_pde=False, 
               pde=None, w_enc=None, w_dec=None, to_clip=False, clip_norm=0.1, scheduler=None) -> Tuple[torch.Tensor]:
    """
    Training loop.
    """
    MSE = MSELoss(loss_calc)

    loss_tot = 0.0
    loss_mse = 0.0
    loss_pde_encoder = 0.0
    loss_pde_decoder = 0.0
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
        loss_pde_enc_batch = pde[0](x_ph, y_ph, z_hat_ph)


        loss_mse += loss_normal_batch
        loss_pde_encoder += loss_pde_enc_batch


        loss_batch = loss_normal_batch   + w_enc*loss_pde_enc_batch 

        if w_dec is not None:
            loss_pde_dec_batch = pde[1](x_ph, z_hat_ph)

            loss_pde_decoder += loss_pde_dec_batch

            loss_batch += w_dec*loss_pde_dec_batch


        loss_tot += loss_batch
        loss_batch.backward()

        # velocity too  , gradient is clipped
        if to_clip == True:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()


        if scheduler is not None:
            scheduler.step()
        # print(pde1.lagrange)

    # mean  batch loss
    loss_tot /= n_batch
    loss_mse /= n_batch

    if w_dec is not None:
        loss_pde_encoder /= n_batch

    loss_pde_decoder /= n_batch

    return (loss_tot, loss_mse, loss_pde_encoder, loss_pde_decoder)

def lr_range_test(model, loss_calc_train, loss_calc_val, train_loader, val_loader, 
                  optimizer, scheduler, device,  normalizer=None, with_pde=False, pde1_train=None, 
                  pde1_val=None, to_clip=False, clip_norm=0.1, wandb_run:wandb.wandb_run.Run=None) -> None:
    """ execute a step of one epoch
    """
    MSE = MSELoss(loss_calc_train)



   
    for data in tqdm(train_loader):
        model.train()
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
        loss_pde1_batch = pde1_train(x_ph, y_ph, z_hat_ph)
        #loss_pde2 = pde1_train[1](x_ph, z_hat_ph)

     
        loss_train_batch = loss_normal_batch + loss_pde1_batch   #+ loss_pde2

        loss_train_batch.backward()

        # velocity too, gradient is clipped
        if to_clip == True:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()


        (loss_tot_val, loss_score_mse, loss_score_pde) = val_step(
            model, loss_calc_val, val_loader, device, normalizer, with_pde, pde1_val)


        dict_log = {
            "lr": scheduler.get_last_lr()[0],
            "train/loss/batch/total": loss_train_batch.item(),
            "train/loss/batch/mse": loss_normal_batch.item(),
            "train/loss/batch/pde": loss_pde1_batch.item(),
            "val/loss/batch/total": loss_tot_val.item(),
            "val/loss/batch/mse": loss_score_mse.item(),
            "val/loss/batch/pde": loss_score_pde.item()
        }


        scheduler.step()
        if wandb_run == None:
            print(dict_log)
        
        else:
            wandb.log(dict_log)
            wandb_run.log_code()



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
    # assumed to account also for racing drone
    pos_limit = [-args.max_pos, args.max_pos]
    vel_limit = [-args.max_vel, args.max_vel]

    # state space domain
    limits = np.array([pos_limit, vel_limit, pos_limit, vel_limit])    

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
    # TODO increase trajectory time
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

    # real matrix
    A = np.array(np.diag(-np.arange(1, system.z_size + 1, 1)))

    # using paper https://proceedings.mlr.press/v242/peralez24a/peralez24a.pdf
    # alpha = 0.5
    # omega = 0.5
    # asyn_dynamic = np.asarray([math.cos(omega), -math.sin(omega), math.sin(omega), math.cos(omega)]).reshape(2,2)/(1+np.exp(alpha))
    # complex conjugate
    #A = np.block([  [asyn_dynamic, np.zeros((2, 2))], [ np.zeros((2, 2)), asyn_dynamic ]])

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
        raise ValueError('The only Lipschitz function implemented is the ReLU')

    device = torch.device(args.device)

    if args.normalize:
        normalizer = Normalizer(train_set, device)

    else:
        normalizer = None

    model = EncoderDecoder(x_size, z_size, args.n_hidden,
                           args.hidden_size, activation, normalizer)

    # L2 regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_ridge)

    loss_fn = nn.MSELoss(reduction='mean')

    train_loader = DataLoader(train_set, args.batch, shuffle=True)
    val_loader = DataLoader(val_set, args.batch, shuffle=True)

    model.to(device)
    loss_score = LossCalculator(loss_fn, model, device, method)


    pde_encoder = PdeLoss_xz(train_set.M, train_set.K, train_set.system,
                            loss_score,  reduction='mean')


    if args.enable_pde_dec_loss == True:
        pde_decoder = PdeLoss_zx(loss_score)
        print('[WARNING] Decoder PDE Loss is enabled')

    else:
        args.w_dec = None
        pde_decoder = None
    # we analyze only PINN network
    with_pde = True  # False if method == 'supervised_NN' else True

    print('Device:', device)



    if args.lr_range_test:
        max_lr = 100
        n_iter = len(train_loader)

        print(f'batch size is ={len(train_loader)}')
        lr_step = (max_lr - args.lr)/len(train_loader);
        print(f'lr_step ={lr_step}')
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=lr_step, total_iters=1n_iter)

        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: step*lr_step)
        
        
        print('Performing LR Range Test')

     
        lr_range_test(model, loss_score, loss_score, train_loader, val_loader, 
                        optimizer, scheduler, device,  normalizer, with_pde, pde_encoder, 
                        pde_encoder, to_clip=args.clip_norm, wandb_run=wandb_run)





    else:

        if args.one_cycle_lr:
            scheduler =  torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.c_lr_max,
                                            epochs=args.n_epoch, steps_per_epoch=len(train_loader))
            
            print('Activated OneCycle Learning Rate')
            
        else:
            scheduler = ReduceLROnPlateau(optimizer, mode='min',
                                        factor=args.factor_scheduler,
                                        patience=args.patiente_scheduler,
                                        threshold=args.threshold_scheduler)
            
            print('Activated Reduce Learning Rate On Plateau')
            

        print('Training is started.', '\n')
        
        loss_min = 0.0
        
        if args.load_ckpt:
            offset_epoch = load_ckpt(
                f'{args.dir}/ckpt_best', model, optimizer, scheduler)

        else:
            offset_epoch = 0


        # hack to discard decoder loss since not the best as stated in the paper
   

        for epoch in trange(offset_epoch, args.n_epoch):
            loss_train_tot, loss_mse_train, loss_pde_encoder_train, loss_pde_decoder_train = train_step(
                model, loss_score, train_loader, optimizer, device, 
                normalizer, with_pde, [pde_encoder, pde_decoder],
                args.w_enc, args.w_dec,to_clip=args.clip_norm, 
                scheduler = scheduler if args.one_cycle_lr else None)

            (loss_score_tot, loss_mse_val, loss_pde_encoder_val, loss_pde_decoder_val) = val_step(
                model, loss_score, val_loader, device, normalizer, 
                with_pde, [pde_encoder, pde_decoder],
                args.w_enc, args.w_dec)

            dict_log = {
                "lr": scheduler.get_last_lr()[0] if args.one_cycle_lr else None,
                "train/loss/total": loss_train_tot.item(),
                "train/loss/mse": loss_mse_train.item(),
                "train/loss/pde/encoder": loss_pde_encoder_train.item(),
                "train/loss/pde/decoder": loss_pde_decoder_train.item(),
                "val/loss/total": loss_score_tot.item(),
                "val/loss/mse": loss_mse_val.item(),
                "val/loss/pde/encoder": loss_pde_encoder_val.item(),
                "val/loss/pde/decoder": loss_pde_decoder_val.item()
            }

            if args.no_track == False:
                wandb.log(dict_log)

                wandb_run.log_code()

            else:
                print(dict_log)

            # when to apply scheduling:
            # ref https://discuss.pytorch.org/t/on-which-dataset-learning-rate-scheduler-is-applied/131259


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
