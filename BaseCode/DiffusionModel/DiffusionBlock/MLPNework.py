import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DiffusionModel.DiffusionBlock.SinusoidalPosEmb import *


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device, t_dim=16):
        super(MLP, self).__init__()

        self.t_dim = t_dim
        self.a_dim = action_dim
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim*2),
            nn.Mish(),
            nn.Linear(t_dim*2, t_dim)
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.final_layer = nn.Linear(hidden_dim, action_dim)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, time, state):
        time = time.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        t_mlp = self.time_mlp(time)
        x = torch.cat([x, state, t_mlp], dim=1)
        x = self.mid_layer(x)
        return self.final_layer(x)

