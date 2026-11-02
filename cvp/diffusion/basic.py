from copy import deepcopy
from typing import Optional, Union

import torch
import torch.nn as nn

from cvp.nn_condition import BaseNNCondition, IdentityCondition
from cvp.nn_diffusion import BaseNNDiffusion


class DiffusionModel:
    """Base class of diffusion models."""

    def __init__(
            self,
            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,

            # ------------------ Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            diffusion_steps: int = 1000,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,

            device: Union[torch.device, str] = "cpu"
    ):
        if optim_params is None:
            optim_params = {"lr": 2e-4, "weight_decay": 1e-5}

        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.diffusion_steps = diffusion_steps
        self.ema_rate = ema_rate

        if nn_condition is None:
            nn_condition = IdentityCondition()

        self.model = nn.ModuleDict({
            "diffusion": nn_diffusion.to(self.device),
            "condition": nn_condition.to(self.device)})
        self.model_ema = deepcopy(self.model).requires_grad_(False)

        self.model.train()
        self.model_ema.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optim_params)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1. - self.ema_rate)

    def update(self, x0, condition=None, update_ema=True, **kwargs):
        raise NotImplementedError

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "model_ema": self.model_ema.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])
