os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from torchdiffeq import odeint
import random

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0, use_bn: bool = True):
        super().__init__()
        self.use_bn = use_bn
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.scale = scale
        if use_bn:
            self.bn1 = nn.BatchNorm1d(dim)
            self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        out = self.fc1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = F.silu(out)
        out = self.fc2(out)
        if self.use_bn:
            out = self.bn2(out)
        return x + self.scale * out

class FlowModelResNet(nn.Module):
    def __init__(self, input_dim=2, time_embed_dim=64, hidden_dim=256, num_blocks=4, use_bn=True):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        in_feat = input_dim + time_embed_dim
        layers = [nn.Linear(in_feat, hidden_dim)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_blocks):
            layers.append(ResidualMLPBlock(hidden_dim, use_bn=use_bn))

        layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        t_embed = self.time_embed(t)
        xt = torch.cat([x, t_embed], dim=-1)
        return self.net(xt)


class FlowMatchingModel:
    def __init__(self, X, dim=10, sigma_min=0.01, device=None,
                 hidden_dim=64, time_embed_dim=32, num_blocks=1, use_bn=False, seed=42, gpu_index=0):
        set_seed(seed)

        if device is None:
            device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.seed = seed
        self.rng_cpu = torch.Generator(device="cpu").manual_seed(seed)
        self.rng_cuda = torch.Generator(device=self.device).manual_seed(seed) if self.device.type == "cuda" else None
        self.X = torch.from_numpy(X).to(torch.float32).to(self.device)
        self.dim = dim
        self.sigma_min = sigma_min

        self.model = FlowModelResNet(
            input_dim=dim,
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks,
            use_bn=use_bn
        ).to(self.device)

    def _sample_source(self, batch_size):
        gen = self.rng_cuda if self.device.type == "cuda" else self.rng_cpu
        return torch.randn(batch_size, self.dim, device=self.device, generator=gen)

    def _sample_target(self, batch_size):
        gen = self.rng_cuda if self.device.type == "cuda" else self.rng_cpu
        idx = torch.randint(0, self.X.size(0), (batch_size,), device=self.device, generator=gen)
        return self.X[idx]

    def _sample_t(self, batch_size):
        gen = self.rng_cuda if self.device.type == "cuda" else self.rng_cpu
        return torch.rand(batch_size, 1, device=self.device, generator=gen)

    def _loss_fn(self, x0, x1, t):
        coeff_x0 = 1 - (1 - self.sigma_min) * t
        xt = coeff_x0 * x0 + t * x1
        v_target = x1 - (1 - self.sigma_min) * x0
        v_pred = self.model(xt, t)
        return (v_pred - v_target).pow(2).mean()

    def flow_matching_loss(self, x0, x1, t):
        xt = (1 - t) * x0 + t * x1
        v_target = x1 - x0
        v_pred = self.model(xt, t)
        return ((v_pred - v_target) ** 2).mean()

    def fit(self, num_steps=20000, batch_size=512, lr=5e-4, show_plot=True):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        pbar = tqdm(range(num_steps), desc="Training", ncols=100)

        for _ in pbar:
            x0 = self._sample_source(batch_size)
            x1 = self._sample_target(batch_size)
            t = self._sample_t(batch_size)

            loss = self._loss_fn(x0, x1, t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            l = float(loss.detach().cpu())
            losses.append(l)
            pbar.set_postfix(loss=f"{l:.4f}")

        if show_plot:
            plt.plot(losses)
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.grid(True)
            plt.show()

    def sample(self, x0, t_span=(0, 1)):
        self.model.eval()
        if isinstance(x0, np.ndarray):
            x0 = torch.tensor(x0, dtype=torch.float32, device=self.device)

        def ode_func(t, x):
            x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)
            t_tensor = torch.tensor([[t]], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                v = self.model(x_tensor, t_tensor)
            return v.squeeze(0).detach().cpu().numpy()

        sol = solve_ivp(ode_func, t_span, x0.detach().cpu().numpy(), t_eval=[t_span[1]])
        return sol.y[:, -1]

    def sample_batch(self, x0, t_span=(0, 1)):
        self.model.eval()
        if isinstance(x0, np.ndarray):
            x0 = torch.tensor(x0, dtype=torch.float32)
        x0 = x0.to(self.device)
        if x0.ndim == 1:
            x0 = x0.unsqueeze(0)

        t = torch.tensor(t_span, dtype=torch.float32, device=self.device)

        def odefunc(t_scalar, x):
            t_expand = torch.full((x.size(0), 1), float(t_scalar), device=self.device)
            return self.model(x, t_expand)

        out = odeint(odefunc, x0, t, rtol=1e-3, atol=1e-5, method='dopri5')
        return out[-1]

    def Jacobi_N(self, y0, t_span=(0, 1)):
        self.model.eval()
        if isinstance(y0, torch.Tensor):
            y0 = y0.detach().cpu().numpy()

        x0 = torch.tensor(y0[:self.dim], dtype=torch.float32, device=self.device)
        J0 = torch.eye(self.dim, dtype=torch.float32, device=self.device).flatten()
        y0_torch = torch.cat([x0, J0])

        def odefunc_aug(t_scalar, y_aug):
            x = y_aug[:self.dim]
            J = y_aug[self.dim:].reshape(self.dim, self.dim)

            x = x.detach().requires_grad_(True)
            t_tensor = torch.tensor([[float(t_scalar)]], dtype=torch.float32, device=self.device)

            v = self.model(x.unsqueeze(0), t_tensor).squeeze(0)
            from torch.func import jacrev
            A = jacrev(lambda x_: self.model(x_, t_tensor).squeeze(0))(x.unsqueeze(0)).squeeze(0)

            dxdt = v
            dJdt = A @ J
            return torch.cat([dxdt, dJdt.reshape(-1)])

        t = torch.tensor(t_span, dtype=torch.float32, device=self.device)
        y_aug_out = odeint(odefunc_aug, y0_torch, t, rtol=1e-3, atol=1e-5, method='dopri5')
        y1 = y_aug_out[-1]
        J1 = y1[self.dim:].reshape(self.dim, self.dim).detach().cpu().numpy()
        return J1

