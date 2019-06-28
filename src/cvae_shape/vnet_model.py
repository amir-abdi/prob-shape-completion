import torch
import torch.nn as nn
import torch.nn.functional as F


def passthrough(x, **kwargs):
    return x


def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        conv1 = self.conv1(x)
        out = self.relu1(self.bn1(conv1))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(nchan, elu))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu, args=None):
        super(InputTransition, self).__init__()
        self.args = args
        s = args.net_size
        self.conv1 = nn.Conv3d(1, int(16 * s), kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(int(16 * s))
        self.relu1 = ELUCons(elu, int(16 * s))

    def forward(self, x):
        conv1 = self.conv1(x)
        out = self.bn1(conv1)
        x16 = x
        s = self.args.net_size
        for i in range(int(16 * s) - 1):
            x16 = torch.cat((x16, x), 1)

        out = self.relu1(torch.add(out, x16))
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False, args=None):
        super(DownTransition, self).__init__()
        self.args = args
        outChans = 2 * inChans
        self.down_conv = nn.Conv3d(inChans, outChans, kernel_size=2, stride=2, padding=0)
        self.bn1 = ContBatchNorm3d(outChans)
        self.do1 = passthrough
        self.relu1 = ELUCons(elu, outChans)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x):
        down = self.relu1(self.bn1(self.down_conv(x)))
        out = self.do1(down)
        out = self.ops(out)
        out = self.relu2(torch.add(out, down))
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False, args=None):
        super(UpTransition, self).__init__()
        self.args = args
        self.up_conv = nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=3, stride=2)
        self.bn1 = ContBatchNorm3d(outChans // 2)
        self.do1 = passthrough
        self.do2 = nn.Dropout3d()
        self.relu1 = ELUCons(elu, outChans // 2)
        self.relu2 = ELUCons(elu, outChans)
        if dropout:
            self.do1 = nn.Dropout3d()
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):

        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv(out)))

        cube_len = self.args.cube_len
        if out.size(2) > cube_len:
            out = out[:, :, 1:cube_len + 1, 1:cube_len + 1, 1:cube_len + 1]

        if skipxdo.size(2) > out.size(2):
            xcat = torch.cat((out, skipxdo[:, :, :out.size(2), :out.size(3), :out.size(4)]), 1)
        elif skipxdo.size(2) == out.size(2) - 1:
            xcat = torch.cat((out, F.pad(skipxdo, [0, 1, 0, 1, 0, 1], 'constant', 0)), 1)
        elif skipxdo.size(2) == out.size(2) - 2:
            xcat = torch.cat((out, F.pad(skipxdo, [1, 1, 1, 1, 1, 1], 'constant', 0)), 1)
        elif skipxdo.size(2) == out.size(2) - 4:
            xcat = torch.cat((out, F.pad(skipxdo, [2, 2, 2, 2, 2, 2], 'constant', 0)), 1)
        else:
            xcat = torch.cat((out, skipxdo), 1)

        out = self.ops(xcat)
        out = self.relu2(torch.add(out, xcat))
        return out


class PGen(nn.Module):
    def __init__(self, inChans, elu, nll, args=None):
        super(PGen, self).__init__()
        self.args = args
        self.conv1 = nn.Conv3d(inChans, 2, kernel_size=2, padding=1)
        self.bn1 = ContBatchNorm3d(2)
        self.conv2 = nn.Conv3d(2, 1, kernel_size=2)
        self.relu1 = ELUCons(elu, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        conv1 = self.conv1(x)
        out = self.relu1(self.bn1(conv1))
        out = self.conv2(out)
        out = self.sigmoid(out)
        return out


class _G(nn.Module):
    def __init__(self, args, elu=True, nll=False, apply_last_layer=True):
        super(_G, self).__init__()
        self.args = args
        s = args.net_size
        self.apply_last_layer = apply_last_layer
        self.sc = args.vnet_skip_conn

        self.output_channels = int(32 * s)

        self.in_tr = InputTransition(int(16 * s), elu, args=args)
        self.down_tr32 = DownTransition(int(16 * s), 1, elu, args=args)
        self.down_tr64 = DownTransition(int(32 * s), 1, elu, args=args)
        self.down_tr128 = DownTransition(int(64 * s), 1, elu, dropout=True, args=args)
        self.down_tr256 = DownTransition(int(128 * s), 1, elu, dropout=True, args=args)

        self.up_tr256 = UpTransition(int(256 * s), int(256 * s), 2, elu, dropout=True, args=args)
        self.up_tr128 = UpTransition(int(256 * s), int(128 * s), 2, elu, dropout=True, args=args)
        self.up_tr64 = UpTransition(int(128 * s), int(64 * s), 1, elu, args=args)
        self.up_tr32 = UpTransition(int(64 * s), int(32 * s), 1, elu, args=args)
        self.out_tr = PGen(int(32 * s), elu, nll, args=args)

        if apply_last_layer:
            self.output_channels = 1
        else:
            self.output_channels = int(32 * s)

    def forward(self, x):
        if len(x.size()) == 4:
            x = x.view(x.size(0), 1, x.size(1), x.size(2), x.size(3))
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)
        # ------- Up -------
        if self.sc:
            out = self.up_tr256(out256, out128)
        else:
            out = self.up_tr256(out256, torch.zeros_like(out128))

        if self.sc:
            out = self.up_tr128(out, out64)
        else:
            out = self.up_tr128(out, torch.zeros_like(out64))

        if self.sc:
            out = self.up_tr64(out, out32)
        else:
            out = self.up_tr64(out, torch.zeros_like(out32))

        if self.sc:
            out = self.up_tr32(out, out16)
        else:
            out = self.up_tr32(out, torch.zeros_like(out16))

        if self.apply_last_layer:
            out = self.out_tr(out)

        return out
