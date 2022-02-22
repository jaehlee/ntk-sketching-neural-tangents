import math
import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
from neural_tangents import stax


class AcosFeatureMap(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(AcosFeatureMap, self).__init__()
        assert input_dim > 0 and output_dim > 0
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = torch.randn(input_dim, output_dim)
        self.norm_const = np.sqrt(2.0 / self.output_dim)

    def forward(self, x, order):
        assert x.shape[1] == self.input_dim
        xw = x @ self.W
        if order == 0:
            return (xw > 0) * self.norm_const
        elif order == 1:
            return (abs(xw) + xw) * (self.norm_const / 2.0)
        else:
            raise NotImplementedError


class CountSketch2(nn.Module):

    def __init__(self, input_dim1, input_dim2, output_dim):
        super(CountSketch2, self).__init__()
        if input_dim1 == 0 or input_dim2 == 0 or output_dim == 0:
            import pdb
            pdb.set_trace()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sign1 = torch.randint(2, (input_dim1,)) * 2 - 1
        self.indx1 = torch.randint(output_dim, (input_dim1,))
        self.sign2 = torch.randint(2, (input_dim2,)) * 2 - 1
        self.indx2 = torch.randint(output_dim, (input_dim2,))

    # From https://github.com/gdlg/pytorch_compact_bilinear_pooling/blob/master/compact_bilinear_pooling/__init__.py
    def count_sketch_forward(self, x, indx, sign):
        x_size = tuple(x.size())
        s_view = (1,) * (len(x_size) - 1) + (x_size[-1],)
        out_size = x_size[:-1] + (self.output_dim,)
        sign = sign.view(s_view)
        xs = x * sign
        indx = indx.view(s_view).expand(x_size)
        out = x.new(*out_size).zero_()
        return out.scatter_add_(-1, indx, xs)

    def forward(self, x, y):
        assert (x.shape[0] == y.shape[0])
        assert (x.shape[1] == self.input_dim1)
        assert (y.shape[1] == self.input_dim2)
        n = x.shape[0]
        x_cs = self.count_sketch_forward(x, self.indx1, self.sign1)
        y_cs = self.count_sketch_forward(y, self.indx2, self.sign2)
        return fft.ifft(fft.fft(x_cs, dim=-1) * fft.fft(y_cs, dim=-1)).real


class CntkFeatureMapOps(nn.Module):

    def __init__(self, num_layers, stride, C, m1, m0, m_cs):
        super(CntkFeatureMapOps, self).__init__()
        if m0 < 0:
            m0 = m1
        self.num_layers = num_layers
        self.stride = stride
        self.num_channels = C
        self.input_dim = (stride**2) * C
        self.m1 = m1
        self.m0 = m0
        self.m_cs = m_cs
        self.arccos0 = [AcosFeatureMap(self.input_dim, m0)]
        self.arccos1 = [AcosFeatureMap(self.input_dim, m1)]
        self.sketches = [CountSketch2(self.input_dim, m0, m_cs)]
        for _ in range(num_layers - 1):
            self.arccos0.append(AcosFeatureMap((stride**2) * m1, m0))
            self.arccos1.append(AcosFeatureMap((stride**2) * m1, m1))
            self.sketches.append(
                CountSketch2((stride**2) * (m_cs + m1), m0, m_cs))

    def forward(self, layer_idx, z_nngp, z_ntk, concat=True):
        tmp = self.arccos0[layer_idx](z_nngp, order=0)
        z_nngp = self.arccos1[layer_idx](z_nngp, order=1)
        z_ntk = self.sketches[layer_idx](z_ntk, tmp)
        if concat:
            z_ntk = torch.cat((z_nngp, z_ntk), axis=1)
        return z_nngp, z_ntk


def conv_feat(X, filter_size):
    N, H, W, C = X.shape
    out = torch.zeros((N, H, W, C * filter_size))
    out[:, :, :, :C] = X
    j = 1
    for i in range(1, min((filter_size + 1) // 2, W)):
        out[:, :, :-i, j * C:(j + 1) * C] = X[:, :, i:]
        j += 1
        out[:, :, i:, j * C:(j + 1) * C] = X[:, :, :-i]
        j += 1
    return out


def conv2d_feat(X, filter_size):
    return conv_feat(torch.moveaxis(conv_feat(X, filter_size), 1, 2),
                     filter_size)


def _avg_pool2d(x, filter_size, stride):
    # Input shape (N,H,W,C) should be changed to (N,C,H,W) for pytorch average
    # pooling (torhc.nn.functional.avg_pool2d) operation.
    return torch.moveaxis(
        torch.nn.functional.avg_pool2d(torch.moveaxis(x, (1, 2, 3), (2, 3, 1)),
                                       filter_size,
                                       stride=stride), (2, 3, 1), (1, 2, 3))


def cntk_feat(x, ops, filter_size, num_layers, global_pool):
    z_nngp = x / np.sqrt(x.shape[-1])
    N_ = x.shape[0]
    for i in range(num_layers):
        # Conv
        z_nngp = conv2d_feat(z_nngp, filter_size) / filter_size
        if i == 0:
            z_ntk = z_nngp
        else:
            z_ntk = conv2d_feat(z_ntk, filter_size) / filter_size
            z_ntk = torch.cat((z_ntk, z_nngp), axis=-1)

        # ReLU
        N, P, Q, _ = z_nngp.shape
        assert (z_nngp.shape[:-1] == z_ntk.shape[:-1])
        z_nngp_new, z_ntk_new = ops(i,
                                    z_nngp.reshape(-1, z_nngp.shape[-1]),
                                    z_ntk.reshape(-1, z_ntk.shape[-1]),
                                    concat=False)
        z_nngp = z_nngp_new.reshape((N, P, Q, -1))
        z_ntk = z_ntk_new.reshape((N, P, Q, -1))
        # Pooling
        if i != num_layers - 1:
            z_nngp = _avg_pool2d(z_nngp, 2, 2)
            z_ntk = _avg_pool2d(z_ntk, 2, 2)

    if global_pool:
        # Global average pooling
        z_nngp = z_nngp.reshape(N_, -1, z_nngp.shape[-1]).mean(axis=1)
        z_ntk = z_ntk.reshape(N_, -1, z_ntk.shape[-1]).mean(axis=1)
    else:
        # Flatten
        _, P_out, Q_out, _ = z_ntk.shape
        z_nngp = z_nngp.reshape(N, -1) / np.sqrt(P_out * Q_out)
        z_ntk = z_ntk.reshape(N, -1) / np.sqrt(P_out * Q_out)

    return z_nngp, z_ntk


def ntk_feat(z_nngp_orig, z_ntk_orig, a1_dim, a0_dim, cs_dim, num_layers=1):

    if a0_dim < 0:
        a0_dim = a1_dim

    phi_1 = z_nngp_orig
    phi_0 = torch.cat((z_ntk_orig, z_nngp_orig), axis=1)

    for _ in range(num_layers):
        tmp_0 = AcosFeatureMap(phi_1.shape[1], a0_dim)(phi_1, 0)
        phi_1 = AcosFeatureMap(phi_1.shape[1], a1_dim)(phi_1, 1)
        mu = CountSketch2(phi_0.shape[1], tmp_0.shape[1], cs_dim)(phi_0, tmp_0)
        phi_0 = torch.cat((phi_1, mu), axis=1)

    return phi_1, phi_0


def myrtle5_feat(x, ops):
    # Myrtle5 network:
    #   conv1-relu1-conv2-relu2-avgpool-conv3-relu3-avgpool-conv4-relu4-(avgpoolx3)
    filter_size = 3
    z_nngp = x / np.sqrt(x.shape[-1])
    N = x.shape[0]

    for i in range(4):
        # Conv
        print(f"i={i} | z_nngp.shape = {z_nngp.shape} ", end='')
        z_nngp = conv2d_feat(z_nngp, filter_size) / filter_size
        if i == 0:
            z_ntk = z_nngp
        else:
            z_ntk = conv2d_feat(z_ntk, filter_size) / filter_size
            z_ntk = torch.cat((z_ntk, z_nngp), axis=-1)
        print(f" || z_nngp.shape = {z_nngp.shape}")

        # ReLU
        _, P, Q, _ = z_nngp.shape
        z_nngp_new, z_ntk_new = ops(i,
                                    z_nngp.reshape(-1, z_nngp.shape[-1]),
                                    z_ntk.reshape(-1, z_ntk.shape[-1]),
                                    concat=False)
        z_nngp = z_nngp_new.reshape((N, P, Q, -1))
        z_ntk = z_ntk_new.reshape((N, P, Q, -1))

        if i > 0:
            # 2x2 AvgPool
            z_nngp = _avg_pool2d(z_nngp, 2, 2)
            z_ntk = _avg_pool2d(z_ntk, 2, 2)

    for _ in range(2):
        z_nngp = _avg_pool2d(z_nngp, 2, 2)
        z_ntk = _avg_pool2d(z_ntk, 2, 2)

    z_ntk = z_ntk.reshape(N, -1)
    z_nngp = z_nngp.reshape(N, -1)
    z_ntk = torch.cat((z_ntk, z_nngp), axis=-1)
    return z_nngp, z_ntk


def TEST_myrtle_network():

    xx = torch.randn(2, 32, 32, 3)
    ops = CntkFeatureMapOps(num_layers=4, stride=3, C=3, m1=400, m0=300, m_cs=600)

    z_nngp, z_ntk = myrtle5_feat(xx, ops)


if __name__ == "__main__":
    TEST_myrtle_network()
   