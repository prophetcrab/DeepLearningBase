import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# class TokenEmbedding(nn.Embedding):
#     def __init__(self, vocab_size, d_model):
#         super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
#
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, device):
        super(TokenEmbedding, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model, padding_idx=1, device=device)
        self.device = device

    def forward(self, x):
        return self.embeddings(x)


if __name__ == '__main__':
    d_model = 512
    vocab_size = 2000
    x = torch.randn(128, 64, 512)
    embedding = TokenEmbedding(vocab_size, d_model)
    res = embedding(x)
    print(res)