from typing import List

import numpy as np
import torch
from tqdm import tqdm

from cvp.diffusion.ddpm import DDPM


class CVP(DDPM):
    '''
    Compositional sampling with a short-horizon DDPM.
    The canonical sequence of `canon_horizon` tokens is split into overlapping instance chunks of `inst_horizon`
    tokens. The chunks are denoised together as one batch, conditioned on the first token (start) and the last
    token (goal) of `priors`.
        `cvp_sample`: ours, guidance on the Tweedie estimates couples the chunks.
        `gsc_sample`: DiffCollage / Generative Skill Chaining baseline, the overlaps are averaged after every step.
    '''

    def init_chunk_params(self, prior: torch.Tensor, canon_horizon: int, stride: int, horizon_ctx: List[List[int]]) -> None:
        '''
        Initialize the mapper between the canonical sequence and the instance chunks.
        '''
        self.canonical_shape = prior.shape[1:]  # (T, dim)
        self.inst_shape = (self.inst_horizon, *self.canonical_shape[1:])  # (t, dim)
        self.segments = self.get_windows(canon_horizon=canon_horizon, stride=stride)
        self.canon_horizon = canon_horizon
        self.horizon_ctx = horizon_ctx

    def get_windows(self, canon_horizon: int, stride: int) -> List[tuple]:
        assert (canon_horizon - self.inst_horizon) % stride == 0, 'The total time steps must be divisible by the stride'
        num_segments = (canon_horizon - self.inst_horizon) // stride + 1
        segments = []
        for i in range(num_segments):
            start = i * stride
            end = start + self.inst_horizon
            segments.append((start, end))
        assert segments[0][0] == 0, 'The first segment must start at 0'
        assert segments[-1][-1] == canon_horizon, 'The last segment must end at the total time steps'
        return segments

    def canonical_to_instance(self, zt: torch.Tensor) -> torch.Tensor:
        '''
        Projects from canonical space into overlapped instance segments.

        Args:
            zt: Canonical sequence tensor of shape (1, t, *), where t is sequence length

        Returns:
            xts: Instance segments tensor of shape (num_segments, *), where num_segments
            is the number of overlapping chunks
        '''
        instances = []
        for start, end in self.segments:
            seg = zt[:, start:end]
            instances.append(seg)
        xts = torch.cat(instances, dim=0)
        return xts

    def replace_instance_to_canonical(self, xts: torch.Tensor) -> torch.Tensor:
        '''
        Unproject instance segments back into canonical sequence with horizon contexts

        Args:
            xts: Instance segments of shape (num_segments, *)

        Returns:
            zt: Canonical sequence of shape (1, t, *) with overlapping segments replaced
        '''
        segments = []
        for i, (start, end) in enumerate(self.segments):
            start = start if i == 0 else start + self.horizon_ctx[i][0]
            length = end - start
            seg = xts[i:i+1, -length:, ...]
            segments.append(seg)

        zt = torch.cat(segments, dim=1)
        return zt

    def average_instance_to_canonical(self, xts: torch.Tensor) -> torch.Tensor:
        '''
        xts: (num_segments, *)
        * indicates instance at each time step's shape
        return: (1, t, *)
        '''
        count = torch.zeros((1, *self.canonical_shape), dtype=xts.dtype, device=self.device)
        value = torch.zeros((1, *self.canonical_shape), dtype=xts.dtype, device=self.device)

        for i, (start, end) in enumerate(self.segments):
            count[:, start:end] += 1
            value[:, start:end] += xts[i:i+1, ...]

        zt = torch.where(count > 0, value / count, value)
        return zt

    def gsc_sample(self,
            priors: torch.Tensor,
            # ----------------- sampling ----------------- #
            temperature: float = 1.0,
            sample_steps: int = 50,
            # ------------------ chunks ------------------ #
            canon_horizon: int = 20,
            stride: int = 6,
            horizon_ctx: List[List[int]] = [[1, 2], [2, 2], [2, 1]],
            # ------------------ others ------------------ #
            preserve_history: bool = False,
        ):
        '''
        DiffCollage / Generative Skill Chaining baseline. After every DDIM step the overlapping tokens of the
        chunks are averaged and the (noised) start and goal tokens are written back.

        Returns:
            zt: canonical sequence of shape (1, t, dim)
            xts: instance segments laid side by side, (1, num_segments*inst_horizon, dim)
            history: noisy chunks and Tweedie estimates of every step if `preserve_history`, else None
        '''
        self.init_chunk_params(prior=priors, canon_horizon=canon_horizon, stride=stride, horizon_ctx=horizon_ctx)
        zt = torch.randn((1, self.canon_horizon, *self.canonical_shape[1:]), device=self.device) * temperature

        xts = self.canonical_to_instance(zt)  # (num_segments, inst, dim)

        sample_alpha, sample_beta, sample_bar_alpha, timestep_map = self.get_sample_info(sample_steps)

        map_tensor = torch.tensor(timestep_map, device=self.device, dtype=torch.long)
        history = {"xts": [], "x0s": []} if preserve_history else None
        for t in tqdm(range(sample_steps - 1, -1, -1), desc='gsc_sample'):
            t_batch = map_tensor[t].repeat(len(self.segments))
            t_batch_prev = map_tensor[t - 1].repeat(len(self.segments)) if t > 0 else torch.tensor(0, device=self.device).repeat(len(self.segments))
            bar_alpha = sample_bar_alpha[t]
            bar_alpha_prev = sample_bar_alpha[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            beta = sample_beta[t]
            with torch.no_grad():
                x0s = self.model["diffusion"](xts, t_batch, None)

                # --------------------- DDIM Step -------------------- #
                eta = 0
                sigma = eta * ((1 - bar_alpha_prev) * beta / (1 - bar_alpha)).sqrt()
                xts = bar_alpha_prev.sqrt() * x0s + \
                    (1 - bar_alpha_prev - sigma**2).sqrt() * (xts - bar_alpha.sqrt() * x0s) / (1 - bar_alpha).sqrt() + \
                    torch.randn_like(xts) * sigma

                # ------------------ Average Overlaps ------------------ #
                zt = self.average_instance_to_canonical(xts)  # (1, t, dim)
                fix_part, _, _ = self.add_noise(priors, t_batch_prev[0], eps=torch.randn_like(priors))  # (1, t, dim)

                zt[:, :self.horizon_ctx[0][0], ...] = fix_part[:, :self.horizon_ctx[0][0], ...]
                zt[:, -self.horizon_ctx[-1][1]:, ...] = fix_part[:, -self.horizon_ctx[-1][1]:, ...]
                xts = self.canonical_to_instance(zt)  # (num_segments, inst, dim)
                if preserve_history:
                    history["xts"].append(xts.reshape(-1, *self.inst_shape[1:])[None, ...].cpu().numpy())  # (1, num_segments*8, dim)
                    history["x0s"].append(x0s.reshape(-1, *self.inst_shape[1:])[None, ...].cpu().numpy())  # (1, num_segments*8, dim)

        xts = xts.reshape(-1, *self.inst_shape[1:])[None, ...]  # (1, num_segments*8, dim)
        return zt, xts, history

    def cvp_sample(self,
            priors: torch.Tensor,
            # ----------------- sampling ----------------- #
            temperature: float = 1.0,
            sample_steps: int = 300,
            # ------------------ chunks ------------------ #
            canon_horizon: int = 20,
            stride: int = 6,
            horizon_ctx: List[List[int]] = [[1, 2], [2, 2], [2, 1]],
            # ------------------ guidance ----------------- #
            wg: float = 0.2,
            discount: float = 0.6,
            w_sync: float = 0.05,
            # ------------------ others ------------------ #
            preserve_history: bool = False,
        ):
        '''
        Ours. At every step the chunks are guided towards agreeing on their boundaries.
        The EMA network predicts the Tweedie estimates `x0s`, which are guided, and the online network predicts
        the targets `x0s_tgt`. The fwd loss compares the start of each chunk with the end of the previous chunk
        (the first chunk with the start token), the bwd loss compares the end of each chunk with the start of the
        next chunk (the last chunk with the goal token), and the sync loss covers all boundaries at once. The
        gradient of the losses w.r.t. the noisy chunks replaces a share `wg` of the noise in the stochastic DDIM
        step (eta = 1).

        horizon_ctx: List of lists, each sublist contains the context size for each segment.
            e.g., [[1, 4], [4, 4], [4, 4], [4, 1]],
            for [1, 4], 1 select the start of the first segment, 4 select the end of the first segment;
            for [4, 1], 4 select the start of the last segment, 1 select the end of the last segment.
        wg: share of the guidance direction in the injected noise, 0 gives a plain stochastic DDIM step.
        discount: the fwd / bwd losses of chunk i are weighted by discount**i / discount**(num_segments-1-i).
        w_sync: weight of the sync loss.

        Returns:
            zt: canonical sequence of shape (1, t, dim)
            xts: instance segments laid side by side, (1, num_segments*inst_horizon, dim)
            history: noisy chunks and Tweedie estimates of every step if `preserve_history`, else None
        '''
        self.init_chunk_params(prior=priors, canon_horizon=canon_horizon, stride=stride, horizon_ctx=horizon_ctx)

        def build_A_B(horizon_ctx, horizon):
            '''
            B selects the start part and A selects the end part of each chunk
            '''
            A_list = []
            B_list = []
            for _, ctx in enumerate(horizon_ctx):
                B = torch.zeros((ctx[0], horizon), dtype=torch.float32, device=self.device)
                B[:ctx[0], :ctx[0]] = torch.eye(ctx[0], dtype=torch.float32, device=self.device)
                B_list.append(B)
                A = torch.zeros((ctx[1], horizon), dtype=torch.float32, device=self.device)
                A[-ctx[1]:, -ctx[1]:] = torch.eye(ctx[1], dtype=torch.float32, device=self.device)
                A_list.append(A)
            return A_list, B_list

        def build_lamda(A_list, B_list, horizon):
            '''
            Block tri-diagonal operator of the boundary constraints between neighboring chunks
            '''
            num_chunks = len(A_list)
            lamda = torch.zeros((num_chunks*horizon, num_chunks*horizon), dtype=torch.float32, device=self.device)
            for i in range(num_chunks):
                At = A_list[i]
                Bt = B_list[i]
                lamda[i*horizon:(i+1)*horizon, i*horizon:(i+1)*horizon] = \
                    Bt.t() @ Bt + At.t() @ At

            for i in range(num_chunks - 1):
                At = A_list[i]
                Bt_next = B_list[i + 1]
                lamda[i*horizon:(i+1)*horizon, (i+1)*horizon:(i+2)*horizon] = \
                    -At.t() @ Bt_next

            for i in range(num_chunks - 1):
                At = A_list[i]
                Bt_next = B_list[i + 1]
                lamda[(i+1)*horizon:(i+2)*horizon, i*horizon:(i+1)*horizon] = \
                    -Bt_next.t() @ At
            return lamda

        A_list, B_list = build_A_B(horizon_ctx, horizon=self.inst_horizon)
        lamda = build_lamda(A_list, B_list, horizon=self.inst_horizon)

        def build_ETA(A_list, B_list, prior, horizon_ctx):
            '''
            Target of the sync loss: the start token enters through the first chunk, the goal token through the last one
            '''
            num_chunks = len(A_list)
            ETA = torch.zeros(num_chunks*self.inst_horizon, *self.inst_shape[1:], dtype=torch.float32, device=self.device)
            B = B_list[0]
            A = A_list[-1]
            ETA[:self.inst_horizon] = torch.einsum('ij, jd -> id', B.t(), prior[0, 0].unsqueeze(0).expand(horizon_ctx[0][0], -1))
            ETA[-self.inst_horizon:] = torch.einsum('ij, jd -> id', A.t(), prior[0][-1].unsqueeze(0).expand(horizon_ctx[-1][1], -1))
            return ETA

        ETA = build_ETA(A_list, B_list, prior=priors, horizon_ctx=horizon_ctx)

        def build_T(horizon_ctx):
            '''
            Select the start parts of each chunks
            '''
            num_chunks = len(horizon_ctx)
            t_ = sum([ctx[0] for ctx in horizon_ctx])
            T = torch.zeros((t_, num_chunks*self.inst_horizon), dtype=torch.float32, device=self.device)
            start = 0
            for i, ctx in enumerate(horizon_ctx):
                shift = ctx[0]
                T[start:start + shift, i*self.inst_horizon:i*self.inst_horizon + shift] = torch.eye(shift, dtype=torch.float32, device=self.device)
                start += shift
            return T

        def build_M(horizon_ctx):
            '''
            Select the end parts of each chunks
            '''
            num_chunks = len(horizon_ctx)
            m_ = sum([ctx[1] for ctx in horizon_ctx])
            M = torch.zeros((m_, num_chunks*self.inst_horizon), dtype=torch.float32, device=self.device)
            start = 0
            for i, ctx in enumerate(horizon_ctx):
                shift = ctx[1]
                M[start:start + shift, (i+1)*self.inst_horizon-shift:(i+1)*self.inst_horizon] = torch.eye(shift, dtype=torch.float32, device=self.device)
                start += shift
            return M

        def build_measurement_fwd(M, x0s_batch, prior, horizon_ctx):
            '''
            Targets of the start parts: the start token, then the end parts of the previous chunks
            '''
            assert x0s_batch.dim() == 2, 'x0s_batch must be of shape (num_segments*t, dim)'
            measurement_ = torch.einsum('ij, jd -> id', M, x0s_batch)
            lhs = prior[0, 0].unsqueeze(0).expand(horizon_ctx[0][0], -1)
            rhs = measurement_[:-horizon_ctx[-1][1]]  # remove goal frames
            measurement = torch.cat((lhs, rhs), dim=0)
            return measurement

        def build_measurement_bwd(T, x0s_batch, prior, horizon_ctx):
            '''
            Targets of the end parts: the start parts of the next chunks, then the goal token
            '''
            assert x0s_batch.dim() == 2, 'x0s_batch must be of shape (num_segments*t, dim)'
            measurement_ = torch.einsum('ij, jd -> id', T, x0s_batch)
            lhs = measurement_[horizon_ctx[0][0]:]  # remove start frames
            rhs = prior[0, -1].unsqueeze(0).expand(horizon_ctx[-1][1], -1)
            measurement = torch.cat((lhs, rhs), dim=0)
            return measurement

        T = build_T(horizon_ctx)
        M = build_M(horizon_ctx)

        zt = torch.randn_like(priors, device=self.device) * temperature

        sample_alpha, sample_beta, sample_bar_alpha, timestep_map = self.get_sample_info(sample_steps)
        map_tensor = torch.tensor(timestep_map, device=self.device, dtype=torch.long)

        xts = self.canonical_to_instance(zt)  # (num_segments, inst, dim)

        history = {"xts": [], "x0s": []} if preserve_history else None
        pbar = tqdm(range(sample_steps - 1, -1, -1), desc='cvp_sample')
        for t in pbar:
            t_batch = map_tensor[t].repeat(len(self.segments))
            bar_alpha = sample_bar_alpha[t]
            bar_alpha_prev = sample_bar_alpha[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            beta = sample_beta[t]
            with torch.set_grad_enabled(True):
                xts = xts.detach().requires_grad_(True)

                x0s = self.model_ema["diffusion"](xts, t_batch, None)  # Tweedie estimates that are guided
                x0s_tgt = self.model["diffusion"](xts, t_batch, None)  # targets of the fwd / bwd losses

                x0s_batch = x0s.reshape(-1, *self.inst_shape[1:])
                x0s_tgt_batch = x0s_tgt.reshape(-1, *self.inst_shape[1:])
                if self.clip_pred:
                    x0s_batch = x0s_batch.clip(self.x_min, self.x_max)
                    x0s_tgt_batch = x0s_tgt_batch.clip(self.x_min, self.x_max)

                # ------------------- Calculate Loss --------------------- #
                # 1. sync norm
                sync_difference = torch.einsum('ij, jd -> id', lamda.detach(), x0s_batch) - ETA.detach()
                sync_loss = - torch.linalg.vector_norm(sync_difference, ord=2, dim=(0, 1))

                # 2. fwd norm
                fwd_measurement = build_measurement_fwd(M, x0s_tgt_batch, priors, horizon_ctx)
                fwd_difference = torch.einsum('ij, jd -> id', T.detach(), x0s_batch) - fwd_measurement.detach()
                weight = torch.ones_like(fwd_difference, device=self.device)
                p = 0
                for i, ctx in enumerate(horizon_ctx):
                    k = ctx[0]
                    weight[p:p+k] = (discount) ** i
                    p += k
                fwd_difference = weight * fwd_difference
                fwd_loss = -torch.linalg.vector_norm(fwd_difference, ord=2, dim=(0, 1))

                # 3. bwd norm
                bwd_measurement = build_measurement_bwd(T, x0s_tgt_batch, priors, horizon_ctx)
                bwd_difference = torch.einsum('ij, jd -> id', M.detach(), x0s_batch) - bwd_measurement.detach()
                weight = torch.ones_like(bwd_difference, device=self.device)
                p = 0
                for i, ctx in enumerate(horizon_ctx):
                    k = ctx[1]
                    weight[p:p+k] = (discount) ** (len(horizon_ctx) - i - 1)
                    p += k
                bwd_difference = weight * bwd_difference
                bwd_loss = -torch.linalg.vector_norm(bwd_difference, ord=2, dim=(0, 1))

                loss = bwd_loss + fwd_loss + w_sync * sync_loss
                guidance = torch.autograd.grad(loss, xts)[0]
                pbar.set_postfix(loss=f'{-loss.item():.4f}')

            with torch.set_grad_enabled(False):
                eta = 1.0
                sigma = eta * ((1 - bar_alpha_prev) * beta / (1 - bar_alpha)).sqrt()

                # --------------------- DDIM Step -------------------- #
                xts = bar_alpha_prev.sqrt() * x0s + (1 - bar_alpha_prev - sigma**2).sqrt() * (xts - bar_alpha.sqrt() * x0s) / (1 - bar_alpha).sqrt()

                # -------------------- Guided Noise -------------------- #
                # replace the DDIM noise by a mix of random noise and the guidance direction, keeping the same norm
                guidance_norm = torch.linalg.vector_norm(guidance, dim=(1, 2), keepdim=False)  # (num_segments,)
                guidance_norm = guidance_norm.view(-1, 1, 1)  # (num_segments, 1, 1)

                local_r = torch.sqrt(torch.tensor(np.prod(self.inst_shape), device=self.device)) * sigma
                local_d_star = local_r / (guidance_norm + 1e-8) * guidance
                local_d_sample = torch.randn_like(xts) * sigma

                mix_local_d = local_d_sample + wg * (local_d_star - local_d_sample)
                mix_local_d_norm = torch.linalg.vector_norm(mix_local_d, dim=(1, 2), keepdim=False)
                mix_local_d_norm = mix_local_d_norm.view(-1, 1, 1)  # (num_segments, 1, 1)
                mix_local_step = mix_local_d / (mix_local_d_norm + 1e-8) * local_r

                xts = xts + mix_local_step
                if preserve_history:
                    history["xts"].append(xts.reshape(-1, *self.inst_shape[1:])[None, ...].cpu().numpy())  # (1, num_segments*8, dim)
                    history["x0s"].append(x0s.reshape(-1, *self.inst_shape[1:])[None, ...].cpu().numpy())  # (1, num_segments*8, dim)

        zt = self.replace_instance_to_canonical(xts)  # (1, t, dim)
        xts = xts.reshape(-1, *self.inst_shape[1:])[None, ...]  # (1, num_segments*8, dim)
        return zt, xts, history
