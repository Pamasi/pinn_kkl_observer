import torch
from torch import nn
from typing import Tuple
# Model NN1 and NN2


class NN(nn.Module):
    def __init__(self, num_hidden, hidden_size, in_size, out_size, activation, normalizer=None):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.normalizer = normalizer
        self.mode = 'normal'
        current_dim = in_size
        for _ in range(num_hidden):
            self.layers.append(nn.Linear(current_dim, hidden_size))
            current_dim = hidden_size
        self.layers.append(nn.Linear(current_dim, out_size))

    def forward(self, x):
        """
        Forward method of the NN.
        If a normalizer object is passed to the class, the network will normalize the input and denormalize the output.
        """
        # Normalize input here
        if self.normalizer != None:
            x = self.normalizer.Normalize(x, self.mode).float()

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)

        # Denormalize output here
        if self.normalizer != None:
            x = self.normalizer.Denormalize(x, self.mode).float()
        return x

# NN1 and NN2 combined


class Main_Network(nn.Module):
    def __init__(self, x_size, z_size, num_hidden, hidden_size, activation, normalizer=None):
        super().__init__()
        self.normalizer = normalizer
        self.net1 = NN(num_hidden, hidden_size, x_size,
                       z_size, activation, normalizer)
        self.net2 = NN(num_hidden, hidden_size, z_size,
                       x_size, activation, normalizer)
        self.mode = 'normal'

    def forward(self, x) -> Tuple[torch.Tensor]:
        self.net1.mode = self.mode
        self.net2.mode = self.mode
        output_xz = self.net1(x)    # Output from NN1
        # Output from NN2 with NN1 as input
        output_xzx = self.net2(output_xz)

        if self.normalizer != None:
            norm_x_hat = self.normalizer.Normalize(
                output_xzx, self.mode).float()
            norm_z_hat = self.normalizer.Normalize(
                output_xz, self.mode).float()
        else:
            norm_x_hat = output_xzx
            norm_z_hat = output_xz

        return output_xz, output_xzx, norm_z_hat, norm_x_hat
