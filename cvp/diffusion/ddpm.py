from typing import Optional, Union

import torch
import torch.nn as nn

from cvp.nn_condition import BaseNNCondition
from cvp.nn_diffusion import BaseNNDiffusion
from cvp.utils import at_least_ndim, cosine_beta_schedule, linear_beta_schedule

from .basic import DiffusionModel
from .utils import space_timesteps


class DDPM(DiffusionModel):
    """DDPM with a discrete noise schedule. The samplers are in `CVP`."""

    def __init__(
            self,
            # ------------------ Horizon ------------------------ #
            inst_horizon: int,  # number of tokens the network denoises at once
            # ----------------- Neural Networks ----------------- #
            nn_diffusion: BaseNNDiffusion,
            nn_condition: Optional[BaseNNCondition] = None,

            # ------------------ Params ---------------- #
            grad_clip_norm: Optional[float] = None,
            diffusion_steps: int = 1000,
            ema_rate: float = 0.995,
            optim_params: Optional[dict] = None,
            x_max: Optional[Union[float, torch.Tensor]] = None,
            x_min: Optional[Union[float, torch.Tensor]] = None,

            # ------------------- DPM Params ------------------- #
            predict_noise: bool = True,
            beta_schedule: str = "cosine",  # or linear
            beta_schedule_params: Optional[dict] = None,

            device: Union[torch.device, str] = "cpu"
    ):
        super().__init__(
            nn_diffusion, nn_condition, grad_clip_norm,
            diffusion_steps, ema_rate, optim_params, device)

        self.inst_horizon = inst_horizon

        if beta_schedule_params is None:
            beta_schedule_params = {}
        beta_schedule_params["T"] = self.diffusion_steps

        if beta_schedule == "linear":
            beta = linear_beta_schedule(**beta_schedule_params)
        elif beta_schedule == "cosine":
            beta = cosine_beta_schedule(**beta_schedule_params)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.predict_noise = predict_noise

        # Training parameters
        self.beta = torch.tensor(beta, device=self.device, dtype=torch.float32)
        self.alpha = 1 - self.beta
        self.bar_alpha = torch.cumprod(self.alpha.clone(), 0)
        self.x_max, self.x_min = x_max, x_min

    def get_sample_info(self, sample_steps: int):
        """
        Noise schedule of the `sample_steps` steps used at inference.
        `timestep_map[i]` is the diffusion step of sampling step i.
        """
        use_time_step = space_timesteps(num_timesteps=self.diffusion_steps, section_counts=str(sample_steps))
        last_alpha_bar = 1.0
        new_beta = []
        timestep_map = []
        for i, bar_alpha in enumerate(self.bar_alpha):
            if i in use_time_step:
                new_beta.append(1. - bar_alpha / last_alpha_bar)
                last_alpha_bar = bar_alpha
                timestep_map.append(i)

        sample_beta = torch.tensor(new_beta, device=self.device, dtype=torch.float32)
        sample_alpha = 1. - sample_beta
        sample_bar_alpha = torch.cumprod(sample_alpha.clone(), 0)
        return sample_alpha, sample_beta, sample_bar_alpha, timestep_map

    @property
    def clip_pred(self):
        return (self.x_max is not None) or (self.x_min is not None)

    # ---------------------------------------------------------------------------
    # Training

    def add_noise(self, x0, t=None, eps=None):
        t = torch.randint(self.diffusion_steps, (x0.shape[0],), device=self.device) if t is None else t
        eps = torch.randn_like(x0) if eps is None else eps
        bar_alpha = at_least_ndim(self.bar_alpha[t], x0.dim())
        xt = x0 * bar_alpha.sqrt() + eps * (1 - bar_alpha).sqrt()
        return xt, t, eps

    def loss(self, x0, condition=None):
        xt, t, eps = self.add_noise(x0)
        condition = self.model["condition"](condition) if condition is not None else None
        if self.predict_noise:
            loss = (self.model["diffusion"](xt, t, condition) - eps) ** 2
        else:
            loss = (self.model["diffusion"](xt, t, condition) - x0) ** 2
        return loss

    def update(self, x0, condition=None, update_ema=True, **kwargs):
        loss = self.loss(x0, condition)
        loss = loss.mean()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) \
            if self.grad_clip_norm else None
        self.optimizer.step()
        self.optimizer.zero_grad()
        if update_ema: self.ema_update()
        log = {"loss": loss.item(), "grad_norm": grad_norm}
        return log
