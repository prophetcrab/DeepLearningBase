import torch
import torch.nn as nn
import torch.nn.functional as F

from TransformerBlock.SupportBlock.FFN import *
from TransformerBlock.SupportBlock.PositionEmbedding import *
from TransformerBlock.SupportBlock.MultiHeadAttention import *
from TransformerBlock.SupportBlock.TokenEmbedding import *
from TransformerBlock.SupportBlock.LayerNorm import *

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob=0.1):
        super(DecoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)

        self.cross_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc, t_mask, s_mask):
        dec = dec.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        enc = enc.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        _x = dec
        #t_mask 掩码 ：因果掩码，下三角矩阵掩码
        x = self.attention1(dec, dec, dec, t_mask)

        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            _x = x
            #s_mask : 填充掩码
            x = self.cross_attention(x, enc, enc, s_mask)

            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x
