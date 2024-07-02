import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from DiffusionModel.DiffusionBlock.SinusoidalPosEmb import *
from DiffusionModel.DiffusionBlock.MLPNework import *
from DiffusionModel.DiffusionBlock.Losses import *

Losses = {
    'l1' : WeightedL1,
    'l2' : WeightedL2,
}

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
class Diffusion(nn.Module):
    def __init__(self, loss_type, beta_schedule="linear", clip_denoised=True, **kwargs):
        super(Diffusion, self).__init__()
        self.state_dim = kwargs['obs_dim']
        self.action_dim = kwargs['act_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.T = kwargs['T']
        self.device = torch.device(kwargs['device'])
        self.model = MLP(self.state_dim, self.action_dim, self.hidden_dim, self.device)

        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, self.T, dtype=torch.float32)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        #forward process
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        #back process
        posterior_variance = (
            betas * (1 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))

        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0/alphas_cumprod))
        self.register_buffer("sqrt_recipm_alphas_cumprod", torch.sqrt(1.0/alphas_cumprod - 1))

        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    def q_posterior(self, x_start, x, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        posterior_variance = extract(self.posterior_variance, t, x.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)

        return posterior_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x, t, pred_noise):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
                - extract(self.sqrt_recipm_alphas_cumprod, t, x.shape) * pred_noise)

    def p_mean_variance(self, x, t, s):
        pred_noise = self.model(x, t, s)
        x_recon = self.predict_start_from_noise(x, t, pred_noise)
        x_recon.clamp_(-1,1)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, model_log_variance = self.p_mean_variance(x, t, s)
        noise = torch.randn_like(x)

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,)*(len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    def p_sample_loop(self, state, shape, *args, **kwargs):
        device = self.device
        batch_size = state.shape[0]
        x = torch.randn(shape, device=device, requires_grad=True)

        for i in reversed(range(0, self.T)):
            t = torch.full((batch_size, ), i, device=device, dtype=torch.long)
            x = self.p_sample(x, t, state)

        return x

    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = [batch_size, self.action_dim]
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        # return action.clamp_(-1, 1)
        return action.clamp_(-1,1)

    def q_sample(self, x_start, t, noise):
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample
    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        x_recon = self.model(x_noisy, t, state)

        loss = self.loss_fn(x_recon, noise, weights)
        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(256, 2).to(device)
    state = torch.randn(256, 11).to(device)
    model = Diffusion(loss_type="l2", obs_dim=11, act_dim=2, hidden_dim=256, T=100, device=device).to(device)
    action = model(state)

    loss = model.loss(x, state)

    print(f"action: {action.shape}; loss: {loss.item()}")

