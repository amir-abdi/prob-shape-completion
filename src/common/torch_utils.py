import warnings

import torch
from torch.nn.functional import _Reduction
from torch._C import _infer_size
from torch.nn.modules.loss import _WeightedLoss
from torch import nn
from torch.nn import functional as F
from torch.utils.data.sampler import Sampler
from torch.distributions import kl


def binary_cross_entropy_class_weighted(input, target, weight=None, size_average=None,
                                        reduce=None, reduction='elementwise_mean',
                                        class_weight=None):
    r"""Function that measures the Binary Cross Entropy
    between the target and the output.

    See :class:`~torch.nn.BCELoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'
        class_weight: A list/tuple of two alpha values, summing up to 1, which indicate the relative weight
            of each class.

    Examples::

        >>> input = torch.randn((3, 2), requires_grad=True)
        >>> target = torch.rand((3, 2), requires_grad=False)
        >>> loss = F.binary_cross_entropy(F.sigmoid(input), target)
        >>> loss.backward()
    """

    # import numpy as np
    # print('max input:', np.max(input.cpu().data.numpy()))
    # print('min input:', np.min(input.cpu().data.numpy()))
    eps = 1e-12
    input = torch.clamp(input, min=eps, max=1 - eps)
    # print('max input:', np.max(input.cpu().data.numpy()))
    # print('min input:', np.min(input.cpu().data.numpy()))

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_enum(size_average, reduce)
    else:
        reduction = _Reduction.get_enum(reduction)
    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}) is deprecated. "
                      "Please ensure they have the same size.".format(target.size(), input.size()))
    if input.nelement() != target.nelement():
        raise ValueError("Target and input must have the same number of elements. target nelement ({}) "
                         "!= input nelement ({})".format(target.nelement(), input.nelement()))

    if weight is not None:
        new_size = _infer_size(target.size(), weight.size())
        weight = weight.expand(new_size)

    if class_weight is not None:
        loss = class_weight[1] * (target * torch.log(input)) + \
               class_weight[0] * ((1 - target) * torch.log(1 - input))

        # loss = (target * torch.log(input)) + \
        #        ((1 - target) * torch.log(1 - input))

        mean_loss = torch.neg(torch.mean(loss))
        # print('mean_loss:', mean_loss.cpu().data.numpy())
        return mean_loss

    mean_loss = torch._C._nn.binary_cross_entropy(input, target, weight, reduction)
    # print('mean_loss:', mean_loss.cpu().data.numpy())
    return mean_loss


class BCELossClassWeighted(_WeightedLoss):
    r"""Creates a criterion that measures the Binary Cross Entropy
    between the target and the output:

    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right],

    where :math:`N` is the batch size. If reduce is ``True``, then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if}\; \text{size_average} = \text{True},\\
            \operatorname{sum}(L),  & \text{if}\; \text{size_average} = \text{False}.
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `y` should be numbers
    between 0 and 1.

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size
            "nbatch".
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'elementwise_mean' | 'sum'. 'none': no reduction will be applied,
            'elementwise_mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: 'elementwise_mean'
        class_weight: A list/tuple of two alpha values which indicate the relative weight of each class.
            The weights are enforced to sum up to 2. This is intuitively set so that the weights [1,1]
            correspond to the non-weighted BCE loss.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input
        - Output: scalar. If `reduce` is False, then `(N, *)`, same shape as
          input.

    Examples::

        >>> m = nn.Sigmoid()
        >>> loss = nn.BCELoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(m(input), target)
        >>> output.backward()
    """

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean',
                 class_weight=None):
        super(BCELossClassWeighted, self).__init__(weight, size_average, reduce, reduction)

        if class_weight is not None:
            assert (class_weight[0] + class_weight[1] == 2), "The class_weights (alpha) should sum up to 2."
        self.class_weight = class_weight

    def forward(self, input, target):
        return binary_cross_entropy_class_weighted(input, target, weight=self.weight, reduction=self.reduction,
                                                   class_weight=self.class_weight)


def dice_torch(pred, target):
    """This definition generalize to real valued pred and target vector; i.e. it is an estiamted dice.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    # have to use contiguous since they may from a torch.view op
    # todo(amirabdi): I guess this will only calculate an estimate of the true dice per sample
    # And I ma right as this considers the entire batch as a single nd volume: sigma(2*A1*A2)/sigma(A1+A2)

    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat)
    B_sum = torch.sum(tflat)

    return (2. * intersection) / (A_sum + B_sum)


def dice_torch_per_sample(pred, target):
    dice = 0
    batch_size = pred.size(0)
    for i in range(batch_size):
        iflat = pred[0].contiguous().view(-1)
        tflat = target[0].contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat)
        B_sum = torch.sum(tflat)
        dice += (2. * intersection) / (A_sum + B_sum)

    dice /= batch_size
    return dice


class EuclideanDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        diff = torch.abs(input - target)
        norm2 = torch.norm(diff, 2, dim=2)
        return torch.mean(norm2)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))


class DiceLossPerSample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        smooth = 1.

        # iflat = input.view(-1)
        # tflat = target.view(-1)
        # intersection = (iflat * tflat).sum()

        # return 1 - ((2. * intersection + smooth) /
        #            (iflat.sum() + tflat.sum() + smooth))
        dice = 0
        batch_size = input.size(0)
        for i in range(batch_size):
            iflat = input[0].contiguous().view(-1)
            tflat = target[0].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            A_sum = torch.sum(iflat)
            B_sum = torch.sum(tflat)
            dice += (2. * intersection + smooth) / (A_sum + B_sum + smooth)

        dice /= batch_size
        return 1 - dice


class DiceLossVoxelWeighted(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, weight):
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        wflat = weight.view(-1)
        intersection = (iflat * tflat * wflat).sum()
        denominator = (iflat * wflat).sum() + (tflat * wflat).sum()
        return 1 - ((2. * intersection + smooth) /
                    (denominator + smooth))


class DiceLossSampleWeighted(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, weight):
        smooth = 1.

        iflat = input.view(input.size(0), input.size(1), -1)
        tflat = target.view(target.size(0), target.size(1), -1)

        intersection = torch.sum(iflat * tflat, 2, keepdim=False)
        denominator = torch.sum(iflat + tflat, 2, keepdim=False)
        dice_per_sample = ((2. * intersection + smooth) / (denominator + smooth))

        # print('loss_per_sample ', loss_per_sample.size())
        # print('weight ', weight.size())

        # print('dice_per_sample ', dice_per_sample)
        # print('weight ', weight)

        dice_per_sample *= weight
        loss_batch = 1 - torch.sum(dice_per_sample, dim=1)
        return torch.mean(loss_batch)


class DiceLossSampleVoxelWeighted(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, sample_weight, voxel_weight):
        smooth = 1.

        iflat = input.view(input.size(0), input.size(1), -1)
        tflat = target.view(target.size(0), target.size(1), -1)
        wflat = voxel_weight.view(voxel_weight.size(0), voxel_weight.size(1), -1)

        intersection = torch.sum(iflat * tflat * wflat, 2, keepdim=False)
        denominator = torch.sum((iflat + tflat) * wflat, 2, keepdim=False)
        dice_per_sample = ((2. * intersection + smooth) / (denominator + smooth))

        weighted_dice = dice_per_sample * sample_weight
        total_dice = torch.sum(weighted_dice, dim=1)

        loss_batch = 1 - torch.mean(total_dice)
        return loss_batch


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * self.gamma).exp() * loss

        return loss.mean()


class SubsetIterativeSampler(Sampler):
    r"""Samples elements iteratively from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def kld_loss_fn_torch(mean, log_var):
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return KLD


def get_main_part(input_shape3d_kept, generated_shape3d, phi):
    # generated_shape3d_new = generated_shape3d.clone().detach()
    mask = torch.ones_like(input_shape3d_kept)
    mask[input_shape3d_kept == 1] = 0
    mask[phi > 0] = 0

    generated_shape3d_new = generated_shape3d * mask

    return generated_shape3d_new


def repeat_cube(tensor, dims):
    s = [1] * len(tensor.size())
    s += dims
    tensor = torch.unsqueeze(tensor, -1)
    tensor = torch.unsqueeze(tensor, -1)
    tensor = torch.unsqueeze(tensor, -1)
    tensor = tensor.repeat(s)
    return tensor


def kl_divergence(posterior_dist, prior_dist, analytic=True):
    """
    Calculate the KL divergence between the posterior and prior KL(Q||P)
    analytic: calculate KL analytically or via sampling from the posterior
    calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
    """
    if analytic:
        # Need to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
        result = kl.kl_divergence(posterior_dist, prior_dist)
    else:
        z_posterior = posterior_dist.rsample()
        log_posterior_prob = posterior_dist.log_prob(z_posterior)
        log_prior_prob = prior_dist.log_prob(z_posterior)
        result = log_posterior_prob - log_prior_prob

    return torch.mean(result)
