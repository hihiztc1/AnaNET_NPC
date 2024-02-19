from layers.Embed import DataEmbedding_wo_pos
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.mvmd_torch import mvmd

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Mode part
class Model(nn.Module):
    """
    AnaNET
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.modes = configs.major
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.dropout = configs.dropout
        self.activation = configs.activation
        self.frep = configs.freq
        self.size = configs.size
        self.topindex_encoder = []
        self.topindex_decoder = []
        self.sign_encoder = configs.is_training
        self.sign_decoder = configs.is_training
        self.topkindex = []
        self.topqindex = []
        self.ksign = configs.is_training
        self.qsign = configs.is_training

        self.decomp = series_decomp(self.size)

        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, self.d_model, configs.embed, self.frep,
                                                  self.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, self.d_model, configs.embed, self.frep,
                                                  self.dropout)

        self.encoder_self_att = FrequencyAdaptiveAttention(in_channels=self.d_model,
                                                      out_channels=self.d_model,
                                                      seq_len=self.seq_len,
                                                      modes=self.modes,
                                                      topindex=self.topindex_encoder,
                                                      sign=self.sign_encoder)
        self.decoder_self_att = FrequencyAdaptiveAttention(in_channels=self.d_model,
                                                      out_channels=self.d_model,
                                                      seq_len=self.seq_len // 2 + self.pred_len,
                                                      modes=self.modes,
                                                      topindex=self.topindex_decoder,
                                                      sign=self.sign_decoder)
        self.decoder_cross_att = AlignmentFrequencyAttention(in_channels=self.d_model,
                                                        out_channels=self.d_model,
                                                        seq_len_q=self.seq_len // 2 + self.pred_len,
                                                        seq_len_kv=self.seq_len,
                                                        topqindex=self.topqindex,
                                                        topkindex=self.topkindex,
                                                        qsign=self.qsign,
                                                        ksign=self.ksign,
                                                        modes=self.modes)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Attention(
                        self.encoder_self_att,
                        self.d_model, self.n_heads),

                    self.d_model,
                    configs.alpha,
                    configs.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(self.d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    Attention(
                        self.decoder_self_att,
                        self.d_model, self.n_heads),
                    CrossAttention(
                        self.decoder_cross_att,
                        self.d_model, self.n_heads),
                    self.d_model,
                    configs.c_out,
                    configs.alpha,
                    configs.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                    size = self.size
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(self.d_model),
            projection=nn.Linear(self.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        high_init, low_init = self.decomp(x_enc)

        # decoder input
        low_init = torch.cat([low_init[:, -self.label_len:, :], mean], dim=1)
        high_init = F.pad(high_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))

        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, h_1, attns = self.encoder(enc_out, self.topindex_encoder, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(high_init, x_mark_dec)
        high_part, low_part = self.decoder(dec_out, enc_out, h_1, self.topindex_decoder,self.topqindex,self.topkindex, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=low_init)
        self.topindex_encoder = self.encoder_self_att.topindex
        self.topindex_decoder = self.decoder_self_att.topindex
        self.topqindex = self.decoder_cross_att.topqindex
        self.topkindex = self.decoder_cross_att.topkindex

        # final
        dec_out = low_part + high_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]




# Attention part
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

    def forward(self, queries, keys, values, indexq, indexk, attn_mask):
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
    def __init__(self, in_channels, out_channels, seq_len, modes, topindex, sign):
        super(FrequencyAdaptiveAttention, self).__init__()

        self.topk = min(modes, seq_len // 2)
        self.topindex = topindex
        self.sign = sign
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, self.topk, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)

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
            self.sign = self.sign - 1
        else:
            self.topindex = index
        out_ft = torch.zeros(B, H, E, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.topindex):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return (x, None)


class AlignmentFrequencyAttention(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len_q, seq_len_kv, topkindex, topqindex, qsign, ksign, modes=64,
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

    def forward(self, q, k, v, indexq, indexk, mask):
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
            self.qsign = self.qsign - 1
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




# Decomposition part
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




# Encoder part
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




# Decoder part
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



if __name__ == '__main__':
    pass
