import argparse
import sys
from os import getcwd, mkdir, listdir
import os.path as osp
import numpy as np
import torch
import wandb
from typing import Any, Optional, Callable, Optional

import wandb.wandb_run


def get_args_parser():
    parser = argparse.ArgumentParser('Set PINN-Observer', add_help=False)

    # data
    parser.add_argument('--ckpt_dir', default='debug',
                        type=str, help='directory of checkpoints')
    parser.add_argument('--n_sample', default=1000, type=int,
                        help='number of samples per simulation')
    parser.add_argument('--t_sim', default=50, type=int,
                        help='time of the simulation (second)')

    systems = ['radar', 'monoslam']
    parser.add_argument('--system', default='radar', choices=systems)
    parser.add_argument('--normalize', action='store_true',
                        help='normalize data')
    
    parser.add_argument('--add_noise', action='store_true',
                        help='add gaussian noise to measurements')
    parser.add_argument('--noise_mean', default=0, type=float,
                        help='mean of the gaussian noise')
    parser.add_argument('--noise_var', default=1e-3,
                        type=float, help='variance of the gaussian noise')

    # network

    methods = ['supervised_NN', 'unsupervised_AE', 'supervised_PINN']
    parser.add_argument('--method', default='supervised_PINN', choices=methods)
    parser.add_argument('--n_hidden', default=5, type=int,
                        help='number of hidden layers')
    parser.add_argument('--hidden_size', default=50, type=int,
                        help='number of neurons per hidden layer')
    parser.add_argument('--activation_fcn', default='relu', type=str,
                        choices=['relu', 'mish'],  help='type of activation function')

    parser.add_argument('--load_ckpt', action='store_true', help='load checkpoint from \
                        the directory previously created for the current configuration')

    # config
    parser.add_argument('--batch', default=32, type=int, help='batch size')
    parser.add_argument('--num_workers', default=2,
                        type=int, help='number of workers')
    parser.add_argument('--device', default='cuda',
                        type=str, help='device used to train')
    parser.add_argument('--model_name', default='distilbert-base-uncased',
                        type=str, help='name of the encoder model')

    # hyperparam
    parser.add_argument('--n_epoch', default=15,
                        type=int, help='number of epochs')
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lmbda', default=0.1, type=float)

    parser.add_argument('--factor_scheduler', default=0.1, type=float)
    parser.add_argument('--threshold_scheduler', default=1e-4, type=float)
    parser.add_argument('--patiente_scheduler', default=1, type=float)
    # technicality
    parser.add_argument('--seed', default=23, type=int, help='seed')
    parser.add_argument('--no_track', action='store_true',
                        help='disable experiment tracking')

    return parser


def save_ckpt(
    net: torch.nn.Module,
    epoch: int,
    loss: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.CyclicLR] = None,
    dir: str = '',
    torch_state='',
    save_best: bool = False
) -> None:

    torch.save({'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'lr': scheduler.state_dict() if scheduler is not None else None,
                'torch_state': torch_state
                },
               get_ckpt_dir('best' if save_best else epoch, dir)
               )


def load_ckpt(
    ckpt_dir: str,
    net: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.CyclicLR] = None
) -> int:

    ckpt = torch.load(ckpt_dir)

    net.load_state_dict(ckpt['model_state_dict'])

    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(['lr'])

    torch.set_rng_state(ckpt['torch_state'])

    epoch = ckpt['epoch']

    return epoch


def get_ckpt_dir(epoch: int, dir: str = '') -> str:
    if osp.exists(dir) == False:
        mkdir(osp.join(getcwd(), dir))

    return osp.join(getcwd(), f'{dir}/ckpt_{epoch}', )


def config_wandb(args: argparse.Namespace) -> wandb.wandb_run.Run:
    print(sys.executable)
    wandb.login()
    wandb_run_name = f'PINN_{args.system}_NH{args.n_hidden}_HS{args.hidden_size}_AF{str(args.activation_fcn).upper()}_LR{args.lr}'

    wandb_tag = [f'{param}@{val}' for param, val in vars(args).items()]

    wandb_run = wandb.init(project='pinn_kkl_observer', config=args,
                           name=wandb_run_name,  reinit=True, tags=wandb_tag)

    # automate the name folder
    args.dir = str(wandb_run_name).replace(".", "_").replace('-', 'm')

    return wandb_run


def runge_kutta4(f: Callable, a: int, b: int, N: int, v: int, inputs: Optional[np.array]):
    h = (b-a) / N
    x = [v]
    t = [a]
    u = 0

    for _ in range(0, N):
        if inputs != None:
            u = np.array(inputs(t[-1]))

        k1 = f(u, v)
        k2 = f(u, v + h/2*k1)
        k3 = f(u, v + h/2*k2)
        k4 = f(u, v + h*k3)

        v = v + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
        x.append(np.ndarray.tolist(v))

        time = t[-1] + h
        t.append(time)

    return x, np.array(t)


def KKL_observer_data(M, K, y, a, b, ic, N):
    scalar_y = False
    data = []
    size_z = M.shape[0]
    h = (b-a) / N

    # check if y is scalar or vector
    if y.ndim > 2:
        # Reshape y from (m,) --> (m, 1) for matrix multiplication
        def f(y, z): return np.matmul(M, z) + \
            np.matmul(K, np.expand_dims(y, 1))

    else:
        def f(y, z): return np.matmul(M, z) + K*y
        scalar_y = True

    for output, init in zip(y, ic):
        x = [np.ndarray.tolist(init)]
        v = np.array(x).T

        if scalar_y == True:
            # Ignore the first output value as we already have the initial conditions
            truncated_output = np.delete(output, 0)

        else:
            truncated_output = output[1:, :]

        for i in truncated_output:
            k1 = f(i, v)
            k2 = f(i, v + h/2*k1)
            k3 = f(i, v + h/2*k2)
            k4 = f(i, v + h*k3)

            v = v + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

            a = np.reshape(v.T, size_z)
            x.append(np.ndarray.tolist(a))
        data.append(np.array(x))

    return np.array(data)


def beta_ic(beta, start):
    return beta / np.sqrt(2) + start


def sample_circular(delta: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Sampling of initial conditions from concentric cricles. Returns a 4D array
    of shape (Number of circles, Number of sampeles in each, 1, 2)
    """
    ic = []
    for distance in delta:
        r = distance + np.sqrt(2)
        angles = np.arange(0, 2*np.pi, 2*np.pi / num_samples)
        x = r*np.cos(angles, np.zeros([1, num_samples])).T
        y = r*np.sin(angles, np.zeros([1, num_samples])).T
        init_cond = np.concatenate((x, y), axis=1)
        ic.append(np.expand_dims(init_cond, axis=1))
    return np.array(ic)


def sample_spherical(delta: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Sampling of initial conditions from expanding spherical shells. Returns a 4D array of
    shape (Number of spheres, Number of circles in each sphere, Number of points in each circle, 3)
    """
    r = delta + np.sqrt(0.02)
    theta = np.arange(0, 2*np.pi, 2*np.pi/num_samples)
    phi = np.arange(0, np.pi, (np.pi)/num_samples)
    def x(r, theta, phi): return r*np.cos(theta)*np.sin(phi)
    def y(r, theta, phi): return r*np.sin(theta)*np.sin(phi)
    def z(r, phi): return r*np.cos(phi)

    sphere = []
    for radius in r:
        circles = []
        for angle in phi:
            x_coord = x(radius, theta, np.ones(
                len(theta))*angle).reshape(-1, 1)
            y_coord = y(radius, theta, np.ones(
                len(theta))*angle).reshape(-1, 1)
            z_coord = z(radius, np.ones(len(theta))*angle).reshape(-1, 1)
            circle_coord = np.concatenate((x_coord, y_coord, z_coord), axis=1)
            circles.append(circle_coord)
        sphere.append(circles)

    return np.array(sphere)


def calc_neg_t(M: np.ndarray, z_max: int, e: int) -> int:
    w, v = np.linalg.eig(M)
    min_ev = np.min(np.abs(np.real(w)))
    kappa = np.linalg.cond(v)
    s = np.sqrt(z_max*M.shape[0])
    t = 1/min_ev*np.log(e / (kappa*s))

    return t


def listdir_filter(path: str):
    filtered_listdir = []
    for f in listdir(path):
        if not f.startswith('.'):
            filtered_listdir.append(f)
    return filtered_listdir
