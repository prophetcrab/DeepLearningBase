import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)