import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from cnn_vs_ffnn import load_fashion, CLASSES

OUT = "results/report_figs"
os.makedirs(OUT, exist_ok=True)

# 1. The receptive-field table (same numbers as the report), recomputed here
LAYERS = [
    ("input",  None, None),
    ("conv1 3x3", 3, 1),
    ("pool1 2x2", 2, 2),
    ("conv2 3x3", 3, 1),
    ("pool2 2x2", 2, 2),
]

def receptive_field_table():
    r, j = 1, 1
    rows = [("input", "-", "-", r, j)]
    for name, k, s in LAYERS[1:]:
        r = r + (k - 1) * j
        j = j * s
        rows.append((name, k, s, r, j))
    return rows

RF = receptive_field_table()
# stage sizes we actually visualize (matches the table given in the prompt)
STAGES = [
    ("input",          28, 1,  1),             # name, feature-map size, dummy, receptive field r (1 input pixel)
    ("conv1 3x3",       26, None, RF[1][3]),    # r = 3
    ("pool1 2x2",       13, None, RF[2][3]),    # r = 4
    ("conv2 3x3",       11, None, RF[3][3]),    # r = 8
    ("pool2 2x2",        5, None, RF[4][3]),    # r = 10
    ("dense (flatten)",  1, None, 28),          # whole image
]
print("Receptive field table used:")
for name, fmap, _, r in STAGES:
    print(f"  {name:<18} feature map {fmap:>3} x {fmap:<3}   receptive field {r} x {r}")


# 2. A tiny random conv-pool-conv-pool forward pass, only to draw plausible
#    (not trained) feature maps for the top row of the static figure. The
#    receptive-field boxes themselves come from the formula above, not from
#    these random weights, so they are exact regardless.
def relu(x): return np.maximum(x, 0)

def conv2d(x, w):
    k = w.shape[0]; H, W = x.shape
    out = np.zeros((H - k + 1, W - k + 1))
    for i in range(k):
        for j in range(k):
            out += w[i, j] * x[i:i + H - k + 1, j:j + W - k + 1]
    return out

def maxpool2(x):
    H, W = x.shape
    return x[:H // 2 * 2, :W // 2 * 2].reshape(H // 2, 2, W // 2, 2).max(axis=(1, 3))

def forward_maps(img, rng):
    w1 = rng.standard_normal((3, 3)) * 0.5
    w2 = rng.standard_normal((3, 3)) * 0.5
    c1 = relu(conv2d(img, w1))
    p1 = maxpool2(c1)
    c2 = relu(conv2d(p1, w2))
    p2 = maxpool2(c2)
    return [img, c1, p1, c2, p2]


# 3. Geometry: map a receptive-field size r, centred on one output unit, back
#    onto a box of pixel coordinates in the ORIGINAL 28x28 image.
def rf_box(center, r, image_size=28):
    """Return (row0, col0, height, width) of an r x r box centred at `center`,
    clipped to the image."""
    cy, cx = center
    half = r / 2.0
    r0 = max(0, cy - half); c0 = max(0, cx - half)
    r1 = min(image_size, cy + half); c1 = min(image_size, cx + half)
    return r0, c0, (r1 - r0), (c1 - c0)


# 4. Static figure: feature-map shrinking (top) + receptive field growing
#    (bottom), side by side, five stages.
def make_static_figure(img, center, cls_name, path):
    maps = forward_maps(img, np.random.default_rng(9))   # 5 maps: input, conv1, pool1, conv2, pool2
    stage_names = [s[0] for s in STAGES]                  # 6 names: input .. dense
    rf_sizes = [s[3] for s in STAGES]                     # 6 sizes: 1, 3, 4, 8, 10, 28
    n = len(STAGES)
    fig, ax = plt.subplots(2, n, figsize=(3.0 * n, 6))

    for i in range(n):
        if i < len(maps):
            m = maps[i]
            vmax = m.max() if m.max() > 0 else 1.0   # per-panel contrast stretch so later, fainter maps stay visible
            ax[0, i].imshow(m, cmap="gray", vmin=0, vmax=vmax)
            ax[0, i].set_title(f"{stage_names[i]}\nfeature map {m.shape[0]}x{m.shape[1]}", fontsize=10)
        else:
            ax[0, i].axis("off")
            ax[0, i].text(0.5, 0.5, "dense layer:\nreads all 400\npooled numbers\nat once", ha="center", va="center",
                          fontsize=10, transform=ax[0, i].transAxes)
        ax[0, i].set_xticks([]); ax[0, i].set_yticks([])

    for i, (name, r) in enumerate(zip(stage_names, rf_sizes)):
        ax[1, i].imshow(img, cmap="gray")
        r0, c0, h, w = rf_box(center, r)
        rect = patches.Rectangle((c0, r0), w, h, linewidth=2.5, edgecolor="red", facecolor="none")
        ax[1, i].add_patch(rect)
        ax[1, i].set_title(f"receptive field {r}x{r}", fontsize=10)
        ax[1, i].set_xticks([]); ax[1, i].set_yticks([])

    fig.suptitle(f"Scale by scale: feature maps shrink, receptive field grows  "
                 f"(test image: {cls_name})", fontsize=13)
    ax[0, 0].set_ylabel("feature map\n(shrinks)", fontsize=10)
    ax[1, 0].set_ylabel("receptive field\n(grows)", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(path, dpi=140)
    plt.close()
    print("saved", path)


# 5. Animated GIF: smoothly grow the red box from 3x3 to 28x28, so the
#    small-scale -> large-scale change is visible as motion.
def make_gif(img, center, cls_name, path, hold_frames=6, transition_frames=10):
    """Ease from one stage's box size to the next; hold briefly at each exact
    table value (3, 4, 8, 10, 28) so the numbers are readable."""
    sizes = [s[3] for s in STAGES[1:]]         # 3, 4, 8, 10, 28  (skip "input", which has no box)
    names = [s[0] for s in STAGES[1:]]
    frames = []

    def render(r, label):
        fig, ax = plt.subplots(figsize=(4.6, 4.6))
        ax.imshow(img, cmap="gray")
        r0, c0, h, w = rf_box(center, r)
        rect = patches.Rectangle((c0, r0), w, h, linewidth=3, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.set_title(f"{label}\nreceptive field: {r:.0f} x {r:.0f} pixels", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frame = Image.fromarray(buf).convert("RGB")
        plt.close(fig)
        return frame

    prev = 1  # start from a single input pixel
    prev_name = "input pixel"
    for size, name in zip(sizes, names):
        for _ in range(hold_frames):
            frames.append(render(prev, prev_name))
        for t in np.linspace(0, 1, transition_frames)[1:]:
            r = prev + t * (size - prev)
            frames.append(render(r, f"{prev_name} -> {name}"))
        prev, prev_name = size, name
    for _ in range(hold_frames * 2):
        frames.append(render(prev, prev_name))

    frames[0].save(path, save_all=True, append_images=frames[1:], duration=90, loop=0)
    print(f"saved {path} ({len(frames)} frames)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None, help="test-image index; default picks a sneaker")
    ap.add_argument("--center", type=float, nargs=2, default=None, metavar=("ROW", "COL"),
                     help="pixel to centre the receptive field on; default is the image centre")
    a = ap.parse_args()

    _, _, xte, yte = load_fashion()
    idx = a.index if a.index is not None else int(np.where(yte == 7)[0][0])  # class 7 = Sneaker
    img = xte[idx, 0]
    cls_name = CLASSES[yte[idx]]
    center = tuple(a.center) if a.center else (14.0, 14.0)

    make_static_figure(img, center, cls_name, f"{OUT}/receptive_field_scales.png")
    make_gif(img, center, cls_name, f"{OUT}/receptive_field_growth.gif")


if __name__ == "__main__":
    main()
