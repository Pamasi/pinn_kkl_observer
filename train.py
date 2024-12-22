import sys
import argparse

import torch
from torch import nn
import systems
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F


from neural_network import MainNetwork

from utils.dataset import DataSet
from utils.common import get_args_parser
from normalizer import Normalizer
from trainer import Trainer


instructions = """To run the script, create a path and download the repository contents.
Run python train_tracking_radar.py arg1 arg2
    arg1 = Location to store the trained model.
    arg2 = Training method, either supervised_NN, unsupervised_AE or supervised_PINN."""


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# to enable reproducibility
torch.backends.cudnn.benchmark = False


def experiment(args: argparse.Namespace):
    torch.manual_seed(9)

    save_dir = args.ckpt_dir
    method = args.method

    # --------------------- System Setup ---------------------

    print('Generating Data.', '\n')
    # set of initial condition among which the LHS is performed
    limits = limits = np.array(
        [[-1000, 1000], [-100, 100], [-1000, 1000], [-100, 100]])    # Sample space

    # parameter for LHS
    a = 0   # start
    b = 50  # end
    N = 1000        # Number of intervals for RK4
    num_ic = 50

    if str(args.system).endswith('radar'):
        system = systems.TrackingRadar()
        # A and B matricies
        A = np.array(np.diag(-np.arange(1, system.z_size + 1, 1)))

        B = np.ones([system.z_size, system.y_size])

    else:
        raise ValueError('System still not implemented')
    dataset = DataSet(system, A, B, a, b, N, num_ic, limits,
                      PINN_sample_mode='split traj', data_gen_mode='negative forward')

    print('Dataset sucessfully generated.', '\n')

    # --------------------- Training Setup ---------------------
    x_size = dataset.system.x_size
    z_size = dataset.system.z_size
    num_hidden = 5
    hidden_size = 50

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

    trainer = Trainer(dataset, args.n_epoch, optimizer, main_net,
                      loss_fn, args.batch, args.lmbda, method, scheduler=scheduler)
    trainer.train()
    torch.save(main_net, save_dir+'/'+method)

    print('Training complete.', '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'PINN-TrackingRadar', parents=[get_args_parser()])
    args = parser.parse_args()

    experiment(args)
