import sys
import torch
from torch import nn
import Systems
import numpy as np
from NN import Main_Network
from Dataset import DataSet
from Normalizer import Normalizer
from Trainer import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

instructions = """To run the script, create a path and download the repository contents.
Run python train_tracking_radar.py arg1 arg2
    arg1 = Location to store the trained model.
    arg2 = Training method, either supervised_NN, unsupervised_AE or supervised_PINN."""


def main():
    torch.manual_seed(9)

    methods = ['supervised_NN', 'unsupervised_AE', 'supervised_PINN']
    save_dir = sys.argv[1]
    method = sys.argv[2]
    if method not in methods:
        raise Exception('Invalid choice of method')

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

    radar_system = Systems.TrackingRadar()
    # A and B matricies
    A = np.array(np.diag(-np.arange(1, radar_system.z_size + 1, 1)))

    B = np.ones([radar_system.z_size, radar_system.y_size])

    dataset = DataSet(radar_system, A, B, a, b, N, num_ic, limits,
                      PINN_sample_mode='split traj', data_gen_mode='negative forward')

    print('Dataset sucessfully generated.', '\n')

    # --------------------- Training Setup ---------------------
    x_size = dataset.system.x_size
    z_size = dataset.system.z_size
    num_hidden = 5
    hidden_size = 50
    activation = F.relu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    normalizer = Normalizer(dataset, device)
    main_net = Main_Network(x_size, z_size, num_hidden,
                            hidden_size, activation, normalizer)

    epochs = 15
    learning_rate = 0.001
    batch_size = 32
    lmbda = 0.1
    optimizer = torch.optim.Adam(main_net.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=1, threshold=0.0001, verbose=True)
    loss_fn = nn.MSELoss(reduction='mean')

    trainer = Trainer(dataset, epochs, optimizer, main_net,
                      loss_fn, batch_size, lmbda, method, scheduler=scheduler)
    trainer.train()
    torch.save(main_net, save_dir+'/'+method)

    print('Training complete.', '\n')


if __name__ == '__main__':
    # try:
    main()
    # except Exception as e:
    #     print(e)
    #     print(instructions)
