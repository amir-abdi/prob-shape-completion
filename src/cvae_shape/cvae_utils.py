"""This code is based on: https://github.com/SimonKohl/probabilistic_unet"""

import torch
from torch import nn
from torch.distributions import Normal, Independent


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block
    convolutional layers, after each block a pooling operation is performed. And after each convolutional
    layer a Leaky ReLU activation function.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block,
                 padding=True, posterior=False, bias=True):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            self.input_channels += 1

        layers = []
        output_dim = 0
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]
            layers.append(nn.Conv3d(input_dim, output_dim, kernel_size=3, padding=int(padding), stride=2, bias=bias))
            layers.append(nn.LeakyReLU(inplace=True))

            for _ in range(no_convs_per_block - 1):
                layers.append(nn.Conv3d(output_dim, output_dim, kernel_size=3, padding=int(padding), bias=bias))
                layers.append(nn.LeakyReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input_tensor):
        x = input_tensor
        for i, module in enumerate(self.layers):
            x = module(x)
        return x


class AxisAlignedConvGaussian(nn.Module):
    """A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix."""

    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, posterior=False, padding=True):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior

        # todo: revert bias to True
        self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block,
                               posterior=self.posterior, padding=padding, bias=False)

        # todo: revert bias to True
        self.conv_layer = nn.Conv3d(num_filters[-1], 2 * self.latent_dim, kernel_size=1, stride=1,
                                    bias=False)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, input_mask, target_mask=None):
        # If target mask is not none, concatenate the mask to the channel axis of the input
        if target_mask is not None:
            self.show_img = input_mask
            self.show_seg = target_mask
            input_mask = torch.cat([input_mask, target_mask], dim=1)
            self.show_concat = input_mask
            self.sum_input = torch.sum(input_mask)

        encoding = self.encoder(input_mask)
        self.show_enc = encoding
        mu_log_sigma = self.conv_layer(encoding)

        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)
        mu_log_sigma = torch.squeeze(mu_log_sigma, dim=2)

        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        dist = Independent(Normal(loc=mu, scale=log_sigma + 0.0001), 1)
        return dist
