import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, maxlen=5000, device='cpu'):
        super(PositionEmbedding, self).__init__()
        self.encoding = torch.zeros(maxlen, d_model, device)
        self.encoding.requires_grad_(False)

