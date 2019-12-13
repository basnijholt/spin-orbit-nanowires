import os
import os.path
import sys
from glob import glob
from typing import Optional

import matplotlib
import matplotlib.cm
import matplotlib.tri as mtri
import numpy as np
from matplotlib import pyplot as plt

import adaptive


def learner_till(till, learner, data):
    new_learner = adaptive.Learner2D(None, bounds=learner.bounds)
    new_learner.data = {k: v for k, v in data[:till]}
    for x, y in learner._bounds_points:
        # always include the bounds
        new_learner.tell((x, y), learner.data[x, y])
    return new_learner


def plot_tri(learner, ax, xy_size):
    ip = learner.ip()
    tri = ip.tri
    xs, ys = tri.points.T
    x_size, y_size = xy_size
    triang = mtri.Triangulation(x_size * xs, y_size * ys, triangles=tri.vertices)
    return ax.triplot(triang, c="k", lw=0.3, alpha=1, zorder=2), (ip.values, triang)


def to_gradient(data, horizontal, spread=20, mid=0.5):
    n, m = data.shape if horizontal else data.shape[::-1]
    x = np.linspace(1, 0, n)
    x = 1 / (np.exp((x - mid) * spread) + 1)  # Fermi-Dirac like
    gradient = x.reshape(1, -1).repeat(m, 0)
    if not horizontal:
        gradient = gradient.T
    gradient_rgb = matplotlib.cm.inferno(data)
    gradient_rgb[:, :, -1] = gradient
    return gradient_rgb


def get_new_artists(npoints, learner, data, ax, xy_size):
    new_learner = learner_till(npoints, learner, data)
    (line1, line2), (zs, triang) = plot_tri(new_learner, ax, xy_size)

    data = learner.interpolated_on_grid()[-1]  # This uses the original learner!
    x_size, y_size = xy_size
    im = ax.imshow(
        to_gradient(np.rot90(data), horizontal=False),
        extent=(-0.5 * x_size, 0.5 * x_size, -0.5 * y_size, 0.5 * y_size),
        zorder=3,
    )
    ax.tripcolor(triang, zs.flatten(), zorder=0, cmap="inferno")
    return im, line1, line2


def generate_cover(learner, save_fname: Optional[str] = "thesis-cover.pdf"):
    data = list(learner.data.items())

    # measured from the guides in Tomas's thesis: `thesis_cover.pdf`
    x_right = 14.335
    x_left = 0.591
    y_top = 0.591
    y_bottom = 10.039

    inch_per_cm = 2.54
    margin = 0.5 / inch_per_cm  # add 5 mm margin on each side

    x_size = x_right - x_left + 2 * margin
    y_size = y_bottom - y_top + 2 * margin
    xy_size = x_size, y_size

    spine_size = 0.8 / inch_per_cm

    fig, ax = plt.subplots(figsize=(x_size, y_size))
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

    ax.set_xticks([])
    ax.set_yticks([])

    im, line1, line2 = get_new_artists(len(data) // 5, learner, data, ax, xy_size)

    title = "Towards realistic numerical simulations \n of Majorana devices"
    title2 = "Towards realistic numerical simulations of Majorana devices"
    author = "Bas Nijholt"

    text_color = "white"

    ax.axis("off")
    text_zorder = 4
    for pos, text in zip([-0.8, 0.7], [author, title]):
        ax.text(
            x_size / 4,
            pos * (y_size - margin) / 2,
            text,
            color=text_color,
            weight="bold",
            verticalalignment="center",
            horizontalalignment="center",
            fontsize=18,
            zorder=text_zorder,
        )

    ax.text(
        -0.09,
        y_size / 4 - 0.9,
        title2,
        color=text_color,
        weight="bold",
        fontsize=10,
        zorder=text_zorder,
        rotation=-90,
        verticalalignment="center",
        horizontalalignment="left",
    )
    ax.text(
        -0.09,
        -y_size / 4 - 1,
        author,
        fontsize=10,
        zorder=text_zorder,
        color=text_color,
        weight="bold",
        rotation=-90,
        verticalalignment="center",
        horizontalalignment="left",
    )

    for i in [-1, +1]:
        ax.vlines(
            i * spine_size / 2, -y_size / 2, y_size / 2, linestyles=":", color="cyan"
        )
        ax.vlines(
            -i * x_size / 2 + i * margin,
            -y_size / 2,
            y_size / 2,
            linestyles=":",
            color="cyan",
        )
        ax.hlines(
            -i * y_size / 2 + i * margin,
            -x_size / 2,
            x_size / 2,
            linestyles=":",
            color="cyan",
        )

    ax.set_xlim(-x_size / 2, x_size / 2)
    ax.set_ylim(-y_size / 2, y_size / 2)
    print(f"Saving {save_fname}")
    if save_fname is not None:
        fig.savefig(
            save_fname, format="pdf", bbox_inches="tight", pad_inches=0.001, dpi=500
        )


def bounds_from_saved_learner(fname):
    learner = adaptive.Learner2D(None, [(-1, 1), (-1, 1)])
    learner.load(fname)
    xs, ys = np.array(list(learner.data.keys())).T
    bounds = [(xs.min(), xs.max()), (ys.min(), ys.max())]
    return bounds


def load_learner(fname="data/mu-sweep2/data_learner_0246.pickle"):
    learner = adaptive.Learner2D(None, bounds_from_saved_learner(fname))
    learner.load(fname)
    return learner


def save(fname):
    print(f"Opening {fname}")
    f = fname.replace("/", "__")[:-7]
    pdf_fname = f"covers/{f}.pdf"
    print(pdf_fname)
    if os.path.exists(pdf_fname):
        print("exists, exit!")
        sys.exit(0)

    learner = load_learner(fname)
    generate_cover(learner, pdf_fname)


if __name__ == "__main__":
    fnames = glob("data/*/*pickle")
    i = int(sys.argv[1])
    fname = fnames[i]
    save(fname)
