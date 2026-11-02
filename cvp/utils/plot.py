from typing import List, Optional

import imageio.v2 as imageio
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

FACTOR_COLORS = ['#00c743', '#ED784A', '#1984C5']  # factor 1, 2, 3 as in the paper
DATASET_COLOR = [84 / 255, 130 / 255, 168 / 255]


def square_canvas(size: float = 6.4):
    """
    Square figure without axes that covers the unit disc, `size` inches at 100 dpi.
    """
    fig = Figure(figsize=(size, size), layout="constrained")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot()
    ax.axis("off")
    # Symmetric limits
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    # Make the data aspect exactly equal and the axes box a square
    ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)
    return fig, ax


def fig_to_image(fig: Figure) -> np.ndarray:
    """
    Render the figure into a uint8 image of shape (H, W, 3).
    """
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def plot_dataset(episodes: np.ndarray, fname: str, max_episodes: Optional[int] = None):
    """
    Scatter the points of the episodes, (N, L, 2), keeping at most `max_episodes` evenly spaced episodes.
    """
    if max_episodes is not None and max_episodes < len(episodes):
        episodes = episodes[np.linspace(0, len(episodes) - 1, max_episodes).round().astype(int)]
    fig, ax = square_canvas()
    for episode in episodes:
        ax.scatter(episode[:, 0], episode[:, 1], s=32, alpha=1, color=DATASET_COLOR)
    fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")


def plot_plan(points: np.ndarray, num_chunks: int, fname: Optional[str] = None) -> np.ndarray:
    """
    Scatter the points of a plan with one color per chunk and return the figure as an image.

    Args:
        points: (num_chunks * chunk_length, 2) array of points, the chunks one after the other
        num_chunks: number of chunks
        fname: if given, the figure is also saved to this file
    """
    fig, ax = square_canvas()
    for j, chunk in enumerate(np.split(points, num_chunks)):
        ax.scatter(chunk[:, 0], chunk[:, 1], s=36, alpha=1, color=FACTOR_COLORS[j % len(FACTOR_COLORS)])
    if fname is not None:
        fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
    return fig_to_image(fig)


def save_mp4(images: List[np.ndarray], filename, fps: int = 20):
    writer = imageio.get_writer(
                    str(filename),
                    fps=fps,
                    codec="libx264",
                    quality=10,
                    pixelformat="yuv420p"
                )

    for image in images:
        writer.append_data(image)

    writer.close()
