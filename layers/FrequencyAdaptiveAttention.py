# coding=utf-8

import numpy as np
import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(CrossAttention, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, indexq,indexk, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)


        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            indexq,
            indexk,
            attn_mask
        )

        out = out.view(B, L, -1)


        return self.out_projection(out), attn

class Attention(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(Attention, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, index, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)


        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            index,
            attn_mask
        )

        out = out.view(B, L, -1)


        return self.out_projection(out), attn

class FrequencyAdaptiveAttention(nn.Module):
    # 1.2
    def __init__(self, in_channels, out_channels, seq_len, modes, topindex, sign):
        super(FrequencyAdaptiveAttention, self).__init__()

        self.topk = min(modes, seq_len // 2)
        self.topindex = topindex
        # 设置标志 为了固化index
        self.sign = sign
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, self.topk, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)


    # 1.2
    def forward(self, q, k, v, index, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        x = q.permute(0, 2, 3, 1)
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x, dim=-1)
        x_mean = abs(x_ft).mean(0).mean(0).mean(0)
        # x_mean[0] = 0
        _, indexk = torch.topk(x_mean, self.topk, largest=True, sorted=True)
        if self.sign == 1:
            self.topindex = indexk
            self.sign = self.sign-1
        else:
            self.topindex = index
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.topindex):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)


class AlignmentFrequencyAttention(nn.Module):
    # 1.2
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, topkindex,topqindex,qsign,ksign, modes=64,
                 activation='tanh', policy=0):
        super(AlignmentFrequencyAttention, self).__init__()

        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.topk = min(modes, seq_len_q // 2, seq_len_kv // 2)
        self.topkindex = topkindex
        self.topqindex = topqindex
        self.ksign = qsign
        self.qsign = ksign

        self.scale = (1 / (in_channels * out_channels))

        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, self.topk, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)


    # 1.2
    def forward(self, q, k, v,indexq,indexk, mask):
        # size = [B, L, H, E]
        B, L, H, E = q.shape
        xq = q.permute(0, 2, 3, 1)  # size = [B, H, E, L]
        xk = k.permute(0, 2, 3, 1)
        xv = v.permute(0, 2, 3, 1)

        # Compute Fourier coefficients
        xq_ft_ = torch.zeros(B, H, E, self.topk, device=xq.device, dtype=torch.cfloat)
        xq_ft = torch.fft.rfft(xq, dim=-1)
        xq_mean = abs(xq_ft).mean(0).mean(0).mean(0)
        # xq_mean[0] = 0
        _, qindex = torch.topk(xq_mean, self.topk, largest=True, sorted=True)
        if self.qsign == 1:
            self.topqindex = qindex
            self.qsign = self.qsign-1
        else:
            self.topqindex = indexq
        for i, j in enumerate(self.topqindex):
            xq_ft_[:, :, :, i] = xq_ft[:, :, :, j]
        xk_ft_ = torch.zeros(B, H, E, self.topk, device=xq.device, dtype=torch.cfloat)
        xk_ft = torch.fft.rfft(xk, dim=-1)
        xk_mean = abs(xk_ft).mean(0).mean(0).mean(0)
        xk_mean[0] = 0
        _, kindex = torch.topk(xk_mean, self.topk, largest=True, sorted=True)
        if self.ksign:
            self.topkindex = kindex
            self.ksign = self.ksign - 1
        else:
            self.topkindex = indexk
        for i, j in enumerate(kindex):
            xk_ft_[:, :, :, i] = xk_ft[:, :, :, j]


        xqk_ft = (torch.einsum("bhex,bhey->bhxy", xq_ft_, xk_ft_))
        if self.activation == 'tanh':
            xqk_ft = xqk_ft.tanh()
        elif self.activation == 'softmax':
            xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
            xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        else:
            raise Exception('{} actiation function is not implemented'.format(self.activation))
        xqkv_ft = torch.einsum("bhxy,bhey->bhex", xqk_ft, xk_ft_)
        xqkvw = torch.einsum("bhex,heox->bhox", xqkv_ft, self.weights1)
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=xq.device, dtype=torch.cfloat)
        for i, j in enumerate(self.topqindex):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]
        # Return to time domain
        out = torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out, None)
    



