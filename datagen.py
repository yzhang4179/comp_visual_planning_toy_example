from pathlib import Path

import hydra
import numpy as np

from cvp.utils.plot import plot_dataset


def flower(n_points: int, noise: float) -> np.ndarray:
    """
    One flower as a closed curve of `n_points` points, (n_points, 2).
    Petal k is the 120 degree arc, facing the origin, of the unit circle centered at (cos phi_k, sin phi_k) with
    phi_k = 150, 270, 390 degrees. The petals are traced one after another, so consecutive petals share an end
    point. Gaussian noise of std `noise` is added to every point.
    """
    assert n_points % 3 == 0, "n_points should be divisible by 3"
    arcs = []
    for phi in np.deg2rad([150.0, 270.0, 390.0]):
        theta = np.linspace(phi + 4 * np.pi / 3, phi + 2 * np.pi / 3, n_points // 3)
        x = np.cos(theta) + noise * np.random.randn(len(theta)) + np.cos(phi)
        y = np.sin(theta) + noise * np.random.randn(len(theta)) + np.sin(phi)
        arcs.append(np.stack((x, y), axis=1))
    return np.concatenate(arcs, axis=0)


def generate_dataset(n_episodes: int, n_points: int, noise: float) -> np.ndarray:
    """
    The flower rotated by `n_episodes` angles evenly spaced in [0, 2 pi], (n_episodes, n_points, 2).
    """
    episodes = []
    for angle in np.linspace(0, 2 * np.pi, n_episodes):
        matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
        episodes.append(flower(n_points, noise) @ matrix.T)
    return np.stack(episodes)


@hydra.main(config_path="configs", config_name="toy", version_base=None)
def pipeline(args):
    data_path = Path(args.data_path)
    if data_path.exists() and not args.overwrite:
        raise FileExistsError(f"{data_path} already exists, run with overwrite=true to replace it")
    np.random.seed(args.seed)

    # ---------------------- Generate Data ----------------------#
    episodes = generate_dataset(args.n_episodes, args.points_per_episode, args.data_noise)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(data_path, episodes)

    # ---------------------- Plot Data ----------------------#
    Path(args.dataset_figure).parent.mkdir(parents=True, exist_ok=True)
    plot_dataset(episodes, args.dataset_figure, max_episodes=args.dataset_figure_episodes)
    print(f"saved {episodes.shape} points to {data_path} and a preview to {args.dataset_figure}")


if __name__ == '__main__':
    pipeline()
