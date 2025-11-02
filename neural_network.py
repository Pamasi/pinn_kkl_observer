import torch
from torch import nn
from typing import Tuple


class FCN(nn.Module):
    """
    FullyConnectedNeural Network
    """
    def __init__(self, num_hidden, hidden_size, in_size, out_size, 
                 activation, normalizer=None, p_drop=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=p_drop)
        self.activation = activation
        self.normalizer = normalizer
        self.mode = None
 
        current_dim = in_size
        for _ in range(num_hidden):
            self.layers.append(nn.Linear(current_dim, hidden_size))
            current_dim = hidden_size
        self.layers.append(nn.Linear(current_dim, out_size))



    def forward(self, x):
        """
        Forward method of the FCN.
        If a normalizer object is passed to the class, the network will normalize the input and denormalize the output.
        """
        # Normalize input here
        if self.normalizer is not None:
            x = self.normalizer.Normalize(x, self.mode).float()
 
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))

        
        x = self.dropout(self.layers[-1](x))

        # Denormalize output here
        if self.normalizer is not None:
            x = self.normalizer.Denormalize(x, self.mode).float()
        return x


class EncoderDecoder(nn.Module):
    """
    Encoder-Decoder architecture use to observer in a latent space 
    """
    def __init__(self, x_size, z_size, num_hidden, hidden_size, activation, normalizer=None):
        super().__init__()
        self.normalizer = normalizer
        self.encoder = FCN(num_hidden, hidden_size, x_size,
                           z_size, activation, normalizer)
        self.decoder = FCN(num_hidden, hidden_size, z_size,
                           x_size, activation, normalizer)
        self.mode = None

    def forward(self, x) -> Tuple[torch.Tensor]:
        self.encoder.mode = self.mode
        self.decoder.mode = self.mode
        output_xz = self.encoder(x)    # Output from NN1
        # Output from NN2 with NN1 as input
        output_xzx = self.decoder(output_xz)

        if self.normalizer != None:
            norm_x_hat = self.normalizer.Normalize(
                output_xzx, self.mode).float()
            norm_z_hat = self.normalizer.Normalize(
                output_xz, self.mode).float()
        else:
            norm_x_hat = output_xzx
            norm_z_hat = output_xz

        return output_xz, output_xzx, norm_z_hat, norm_x_hat
