"""This code is based on: https://github.com/SimonKohl/probabilistic_unet"""

import torch
from torch import nn
from torch.distributions import Normal, Independent

from cvae_shape.vnet_model import _G as Vnet
from cvae_shape.vnet_model import PGen
from cvae_shape.vnet_utils import init_weights, init_weights_orthogonal_normal
from common.utils import var_or_cuda
from cvae_shape.cvae_utils import AxisAlignedConvGaussian
from common.torch_utils import kl_divergence, repeat_cube

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Pcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the VNet (the feature map) by concatenating them along their channel axis.
    """

    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers):
        super(Pcomb, self).__init__()
        self.num_classes = num_classes
        self.spatial_axes = [2, 3, 4]

        self.latent_dim = latent_dim
        self.no_convs_fcomb = no_convs_fcomb

        layers = list()
        layers.append(nn.Conv3d(num_output_channels + self.latent_dim, num_filters, kernel_size=1))
        layers.append(nn.LeakyReLU(inplace=True))

        for _ in range(no_convs_fcomb - 2):
            layers.append(nn.Conv3d(num_filters, num_filters, kernel_size=1))
            layers.append(nn.LeakyReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.last_layer = PGen(num_filters, True, False)

        if initializers['w'] == 'orthogonal':
            self.layers.apply(init_weights_orthogonal_normal)
            self.last_layer.apply(init_weights_orthogonal_normal)
        else:
            self.layers.apply(init_weights)
            self.last_layer.apply(init_weights)

    def forward(self, feature_map):
        output = feature_map
        for i, module in enumerate(self.layers):
            output = module(output)

        return self.last_layer(output)


class _G(nn.Module):
    def __init__(self, args):
        super(_G, self).__init__()

        self.args = args
        input_channels = 1
        num_classes = 1
        num_filters = [32, 64, 128, 192]
        latent_dim = args.z_size

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.num_fcomb_filters = args.num_fcomb_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = args.no_convs_per_block_fcomb  # both fcomb and encoder
        self.no_convs_fcomb = args.no_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.z_prior_sample = 0

        self.vnet = Vnet(args, apply_last_layer=False).to(device)
        self.fixed_dist = Independent(Normal(loc=var_or_cuda(torch.zeros(args.z_size)),
                                             scale=var_or_cuda(torch.ones(args.z_size))), 1)

        self.posterior = AxisAlignedConvGaussian(self.input_channels,
                                                 [2, 4, 4, 8, 8, 8],
                                                 no_convs_per_block=1,
                                                 padding=False,
                                                 latent_dim=self.latent_dim,
                                                 posterior=True).to(device)

        self.prior = AxisAlignedConvGaussian(self.input_channels,
                                             [2, 4, 4, 8, 8, 8],
                                             no_convs_per_block=1,
                                             padding=False,
                                             latent_dim=self.latent_dim,
                                             posterior=False).to(device)

        self.fcomb = Pcomb(self.num_fcomb_filters, self.latent_dim,
                           self.vnet.output_channels, self.num_classes,
                           self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'}).to(device)

    def forward(self, patch, target=None, combine=False, prior_post_latent=None):
        if target is not None:
            s0, s1 = target.size(0), target.size(1)
            target = target.view(s0 * s1, 1,
                                 target.size(2),
                                 target.size(3),
                                 target.size(4))

            input_rep = patch.repeat(1, s1, 1, 1, 1)
            input_rep = input_rep.view(s0 * s1, 1,
                                       target.size(2),
                                       target.size(3),
                                       target.size(4))
            post_dist = self.posterior.forward(input_rep, target)
            loss_dist = kl_divergence(post_dist, self.fixed_dist, analytic=True)

            prior_post_latent = post_dist.rsample()
            prior_post_latent = prior_post_latent.view(s0, s1, -1)

            unet_features = self.vnet.forward(patch)
            reconstructed = self.reconstruct_mult(unet_features, prior_post_latent)
            return reconstructed, loss_dist
        elif prior_post_latent is not None:
            unet_features = self.vnet.forward(patch)
            reconstructed = self.reconstruct_mult(unet_features, prior_post_latent)
            if not combine:
                return reconstructed
            else:
                reconstructed = reconstructed.mean(dim=1, keepdim=False)
                return reconstructed
        else:
            raise NotImplementedError('In CVAE, during inference, the latent should have been sampled from the '
                                      'prior distribution before calling inference.')

    def reconstruct_mult(self, feature_map, latent_sample):

        # turn features into: batch_size x num_variations x last_layer_features x cube_len x cube_len x cube_len
        feature_map = feature_map.unsqueeze(1)
        features_tiled = feature_map.repeat(1, latent_sample.size(1), 1, 1, 1, 1)

        # turn latent into: batch_size x num_variations x latent_size x cube_len x cube_len x cube_len
        cl = self.args.cube_len
        latent_sample = repeat_cube(latent_sample, [cl, cl, cl])

        # concat and reshape
        concat = torch.cat((features_tiled, latent_sample), dim=2)
        concat = concat.view((concat.size(0) * concat.size(1),
                              concat.size(2),
                              concat.size(3),
                              concat.size(4),
                              concat.size(5)))

        combed = self.fcomb(concat)
        combed = combed.view(latent_sample.size(0),  # self.args.batch_size
                             latent_sample.size(1),
                             combed.size(2),
                             combed.size(3),
                             combed.size(4))
        return combed

    def reconstruct_single(self, feature_map, latent_dist=None, use_latent_mean=False,
                           sample_latent=False,
                           latent_sample=None):
        """
        Reconstruct a segmentation from a posterior sample (decoding a posterior sample) and UNet feature map
        use_posterior_mean: use posterior_mean instead of sampling z_q
        calculate_posterior: use a provided sample or sample from posterior latent space
        """
        if latent_sample is not None:
            pass
        elif use_latent_mean:
            latent_sample = latent_dist.loc
        elif sample_latent:
            latent_sample = latent_dist.rsample()

        latent_sample = torch.unsqueeze(latent_sample, -1)
        cl = self.args.cube_len
        latent_sample = repeat_cube(latent_sample, [cl, cl, cl])

        concat = torch.cat((feature_map, latent_sample), dim=1)
        combed = self.fcomb(concat)
        return combed
