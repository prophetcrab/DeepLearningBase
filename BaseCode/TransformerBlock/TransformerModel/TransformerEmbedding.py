import torch
import torch.nn as nn
import torch.nn.functional as F

from TransformerBlock.SupportBlock.FFN import *
from TransformerBlock.SupportBlock.PositionEmbedding import *
from TransformerBlock.SupportBlock.MultiHeadAttention import *
from TransformerBlock.SupportBlock.TokenEmbedding import *
from TransformerBlock.SupportBlock.LayerNorm import *

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model, device)
        self.pos_emb = PositionEmbedding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)
        self.device = device

    def forward(self, x):
        x = x.to(self.device)
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)



if __name__ == '__main__':
    d_model = 512
    max_len = 5000
    drop_prob = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = 5893
    model = TransformerEmbedding(vocab_size, d_model, max_len, drop_prob, device)
    src = torch.load("D:/PythonProject/BaseCode/Data/tensor_src.pt")
    src = torch.cat((src, torch.zeros(src.shape[0], 2, dtype=torch.int)), dim=-1)
    res = model(src)
    print(res)

