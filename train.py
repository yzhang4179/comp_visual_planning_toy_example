from pathlib import Path

import hydra
import numpy as np
import torch
from tqdm import tqdm

from cvp.dataset import ToyDataset
from cvp.diffusion import DDPM
from cvp.nn_diffusion import DiT1d
from cvp.utils import count_parameters, loop_dataloader, seed_everything
from cvp.utils.hub import resolve_file


@hydra.main(config_path="configs", config_name="toy", version_base=None)
def pipeline(args):
    seed_everything(args.seed, deterministic=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = args.device if torch.cuda.is_available() else 'cpu'

    in_dim = 2 * args.points_per_token
    inst_horizon = args.horizon // args.points_per_token
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- Create Dataset ----------------------#
    dataset = ToyDataset(dataset_path=resolve_file(args.data_path, args.hf_repo),
                         horizon=args.horizon, pad_before=0, pad_after=0)
    print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                shuffle=True, pin_memory=True, persistent_workers=args.num_workers > 0, drop_last=True)

    # ---------------------- Create Diffusion Model ----------------------#
    nn_diffusion = DiT1d(in_dim=in_dim,
        emb_dim=args.emb_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        dropout=args.dropout,)
    print(f"DiT1d parameters: {count_parameters(nn_diffusion) / 1e6:.2f}M")

    agent = DDPM(inst_horizon=inst_horizon,
        nn_diffusion=nn_diffusion, nn_condition=None, diffusion_steps=args.diffusion_steps,
        predict_noise=args.predict_noise, beta_schedule=args.beta_schedule,
        optim_params={'lr': args.lr}, device=device)
    agent.train()
    n_gradient_step = 0
    diffusion_loss_list = []

    # ---------------------- Training ----------------------#
    for batch in tqdm(loop_dataloader(dataloader), desc="Training", total=args.gradient_steps):
        # (batch, horizon, 2) points -> (batch, inst_horizon, in_dim) tokens of `points_per_token` points
        nstates = batch['states'].to(device).reshape(-1, inst_horizon, in_dim)

        # update diffusion
        diffusion_loss = agent.update(x0=nstates, condition=None)['loss']
        diffusion_loss_list.append(diffusion_loss)
        n_gradient_step += 1

        if n_gradient_step % args.log_interval == 0:
            tqdm.write(f"[step {n_gradient_step}] avg_diffusion_loss: {np.mean(diffusion_loss_list):.6f}")
            diffusion_loss_list = []

        if n_gradient_step % args.save_interval == 0:
            agent.save(ckpt_dir / f'diffusion_{n_gradient_step}.ckpt')

        if n_gradient_step >= args.gradient_steps:
            break
    agent.save(ckpt_dir / 'diffusion_final.ckpt')
    print('done')


if __name__ == '__main__':
    pipeline()
