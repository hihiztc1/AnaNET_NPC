import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.mvmd_torch import mvmd

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """

    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class FD_block(nn.Module):

    """
    Series decomposition block
    """
    def __init__(self,alpha):
        super(FD_block, self).__init__()
        self.alpha = alpha
        self.tau = 0
        self.K = 2
        self.DC = 0
        self.init = 1
        self.tol = 5e-5
        self.max_N = 5

    def forward(self, x):
        with torch.no_grad():
            x_permuted = x.permute(0, 2, 1)
            x_reshaped = x_permuted.reshape(-1, x.shape[1])

            result = mvmd(x_reshaped, self.alpha, self.tau, self.K, self.DC, self.init, self.tol, self.max_N)

            x_l_reshaped = result[0, :, :].reshape(x.shape[0], x.shape[2], x.shape[1])
            x_l = x_l_reshaped.permute(0, 2, 1)

            x_h = x - x_l

        return x_h, x_l


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    AnaNET encoder
    """

    def __init__(self, attention, d_model, alpha, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention

        self.decomp1 = FD_block(alpha)
        self.decomp2 = FD_block(alpha)

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, index, attn_mask=None):

        h_1, l_1 = self.decomp1(x)

        new_h_1, attn = self.attention(
            h_1, h_1, h_1, index,
            attn_mask=attn_mask
        )
        h_1 = h_1 + self.dropout(new_h_1)
        h_2, l_2 = self.decomp2(h_1)

        return h_2, h_1, attn


class Encoder(nn.Module):
    """
    AnaNET encoder
    """

    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, index, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                h_2, h_1, attn = attn_layer(x, index, attn_mask=attn_mask)
                x = conv_layer(h_2)
                attns.append(attn)
            h_2, _, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                h_2, h_1, attn = attn_layer(x, index, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            h_2 = self.norm(h_2)
            h_1 = self.norm(h_1)

        return h_2, h_1, attns


class DecoderLayer(nn.Module):
    """
    AnaNET decoder Layer
    """

    def __init__(self, self_attention, cross_attention, d_model, c_out, alpha, d_ff=None,
                 dropout=0.1, activation="relu", size=24):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention

        # Decomp
        self.decomp2 = FD_block(alpha)
        self.decomp1 = series_decomp(size)

        self.dropout = nn.Dropout(dropout)


        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, h_1, index_de, indexq, indexk, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x, index_de,
            attn_mask=x_mask
        )[0])

        x, trend1 = self.decomp1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross, indexq, indexk,
            attn_mask=cross_mask
        )[0])

        x, trend2 = self.decomp2(x)

        residual_trend = trend1 + trend2
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    AnaNET encoder
    """

    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, h_1, index_de, indexq, indexk, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, h_1, index_de, indexq, indexk, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
