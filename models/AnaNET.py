import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.FrequencyAdaptiveAttention import FrequencyAdaptiveAttention, AlignmentFrequencyAttention, Attention, \
    CrossAttention
from layers.AnaNET_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, \
    VMD_block
import math
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


        # Decomp
        if configs.fd_method == 'MA':
            self.decomp = series_decomp(self.size)
        elif configs.fd_method == 'VMD':
            self.decomp = series_decomp(self.size)
            # self.decomp = VMD_block(configs.K, configs.alpha)
        else:
            raise ValueError("fd_method not in ['MA', 'VMD']")

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
        # Encoder
        # enc_modes = int(min(self.modes, self.seq_len // 2))
        # dec_modes = int(min(self.modes, (self.seq_len // 2 + self.pred_len) // 2))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Attention(
                        self.encoder_self_att,
                        self.d_model, self.n_heads),

                    self.d_model,
                    configs.fd_method,
                    configs.K,
                    configs.alpha,
                    configs.d_ff,
                    dropout=self.dropout,
                    activation=self.activation,
                    size = self.size
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
                    configs.fd_method,
                    configs.K,
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
        low_init = torch.zeros_like(low_init)
        high_init = F.pad(x_enc[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
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


if __name__ == '__main__':
    pass
