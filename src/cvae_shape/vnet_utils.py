import torch.nn as nn
import matplotlib.pyplot as plt
from torch.distributions.kl import register_kl, kl_divergence
from torch.distributions import Independent
from torch.distributions.utils import _sum_rightmost


@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    result = kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)


def truncated_normal_(tensor, mean=0, std=1.0):
    size = tensor.shape

    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]

    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv3d or type(m) == nn.ConvTranspose3d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv3d or type(m) == nn.ConvTranspose3d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg


def save_mask_prediction_example(mask, pred, iteration):
    plt.imshow(pred[0, :, :], cmap='Greys')
    plt.savefig('images/' + str(iteration) + "_prediction.png")
    plt.imshow(mask[0, :, :], cmap='Greys')
    plt.savefig('images/' + str(iteration) + "_mask.png")
