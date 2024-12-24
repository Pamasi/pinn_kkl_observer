import torch


class Normalizer:
    """
    Normalize do avoid numerical issue between different state scalings
    """
    def __init__(self, dataset, device):
        self.x_size = dataset.system.x_size
        self.z_size = dataset.system.z_size

        self.mean_x = dataset.mean_x
        self.std_x = dataset.std_x
        self.mean_z = dataset.mean_z
        self.std_z = dataset.std_z

        self.mean_x_ph = dataset.mean_x_ph
        self.std_x_ph = dataset.std_x_ph
        self.mean_z_ph = dataset.mean_z_ph
        self.std_z_ph = dataset.std_z_ph

        self.sys = dataset.system

        self.device = device

    def check_sys(self, x, mode):
        """
        Checks if the x is x or z data. Then if the x belongs to the 
        physics or normal dataset. The correct mean and standard deviations are chosen 
        according to those parameters.
        """
        if x.size()[1] == self.sys.x_size:     # Check if x or z input
            if mode == 'physics':       # Check if physics or normal data point
                mean = self.mean_x_ph
                std = self.std_x
            else:
                mean = self.mean_x
                std = self.std_x
        elif x.size()[1] == self.sys.z_size:
            if mode == 'physics':
                mean = self.mean_z_ph
                std = self.std_z_ph
            else:
                mean = self.mean_z
                std = self.std_z
        else:
            raise Exception('Size of x unmatched with any system.')

        return mean.to(self.device), std.to(self.device)

    def Normalize(self, x, mode):
        mean, std = self.check_sys(x, mode)

        # move to same sensor
        normalized_x = (x.to(self.device) - mean.to(self.device)) / std
        return normalized_x

    def Denormalize(self, x, mode):
        mean, std = self.check_sys(x, mode)
        return x*std + mean
