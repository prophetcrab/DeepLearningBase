import torch
import torch.nn as nn
import torch.nn.functional as F

from TransformerBlock.SupportBlock.FFN import *
from TransformerBlock.SupportBlock.PositionEmbedding import *
from TransformerBlock.SupportBlock.MultiHeadAttention import *
from TransformerBlock.SupportBlock.TokenEmbedding import *
from TransformerBlock.SupportBlock.LayerNorm import *

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = LayerNorm(d_model)
        self.drop1 = nn.Dropout(drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm(d_model)
        self.drop2 = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        _x = x
        x = self.attention(x, x, x, mask)

        x = self.drop1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.ffn(x)

        x = self.drop2(x)
        x = self.norm2(x + _x)

        return x