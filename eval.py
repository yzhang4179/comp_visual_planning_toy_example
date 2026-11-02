from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from cvp.dataset import ToyDataset
from cvp.diffusion import CVP
from cvp.nn_diffusion import DiT1d
from cvp.utils import seed_everything
from cvp.utils.hub import resolve_file
from cvp.utils.plot import plot_plan, save_mp4


@hydra.main(config_path="configs", config_name="toy", version_base=None)
def pipeline(args):
    seed_everything(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"

    in_dim = 2 * args.points_per_token
    inst_horizon = args.horizon // args.points_per_token
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- Create Dataset ----------------------#
    data_path = resolve_file(args.data_path, args.hf_repo)
    dataset = ToyDataset(
        dataset_path=data_path, horizon=args.horizon, pad_before=0, pad_after=0
    )
    normalizer = dataset.normalizers["states"]

    # ---------------------- Set Start and Goal ----------------------#
    # the first and the last token of one episode
    data = np.load(data_path)[args.episode]  # (600, 2)
    normalized_data = normalizer.normalize(data).astype(np.float32)
    start = torch.from_numpy(
        normalized_data[: args.points_per_token].reshape(1, in_dim)
    ).to(device)
    goal = torch.from_numpy(
        normalized_data[-args.points_per_token :].reshape(1, in_dim)
    ).to(device)
    prior = torch.zeros((1, args.canon_horizon, in_dim), device=device)
    prior[:, 0, :] = start
    prior[:, -1, :] = goal

    # ---------------------- Create Diffusion Model ----------------------#
    nn_diffusion = DiT1d(
        in_dim=in_dim,
        emb_dim=args.emb_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.depth,
        dropout=args.dropout,
    )

    agent = CVP(
        inst_horizon=inst_horizon,
        nn_diffusion=nn_diffusion,
        nn_condition=None,
        diffusion_steps=args.diffusion_steps,
        predict_noise=args.predict_noise,
        beta_schedule=args.beta_schedule,
        device=device,
    )
    agent.load(resolve_file(args.ckpt_path, args.hf_repo))
    agent.eval()

    # ---------------------- Compositional Sampling ----------------------#
    if args.method == "cvp":
        _, _, history = agent.cvp_sample(
            priors=prior,
            # ----------------- sampling ----------------- #
            temperature=args.temperature,
            sample_steps=args.sample_steps,
            # ------------------ chunks ------------------ #
            canon_horizon=args.canon_horizon,
            stride=args.stride,
            horizon_ctx=OmegaConf.to_container(args.horizon_ctx),
            # ------------------ guidance ----------------- #
            wg=args.wg,
            discount=args.discount,
            w_sync=args.w_sync,
            # ------------------ others ------------------ #
            preserve_history=True,
        )
    elif args.method == "gsc":
        _, _, history = agent.gsc_sample(
            priors=prior,
            # ----------------- sampling ----------------- #
            temperature=args.temperature,
            sample_steps=args.sample_steps,
            # ------------------ chunks ------------------ #
            canon_horizon=args.canon_horizon,
            stride=args.stride,
            horizon_ctx=OmegaConf.to_container(args.horizon_ctx),
            # ------------------ others ------------------ #
            preserve_history=True,
        )
    else:
        raise ValueError(f"Unknown method {args.method}, expected cvp or gsc")

    # ---------------------- Visualization ----------------------#
    num_chunks = len(agent.segments)
    n_steps = len(history["xts"])
    frame_ids = (
        list(range(0, n_steps, args.video_stride)) + [n_steps - 1] * args.video_hold
    )
    for key in ("xts", "x0s"):
        frames = []
        for i in frame_ids:
            data = normalizer.unnormalize(
                history[key][i].reshape(-1, 2)
            )  # (num_chunks * horizon, 2)
            frames.append(plot_plan(data, num_chunks))
        save_mp4(frames, output_dir / f"{key}.mp4", fps=args.video_fps)

    data = normalizer.unnormalize(history["xts"][-1].reshape(-1, 2))
    plot_plan(data, num_chunks, fname=output_dir / "plan.png")
    print(f"saved xts.mp4, x0s.mp4 and plan.png to {output_dir}")


if __name__ == "__main__":
    pipeline()
