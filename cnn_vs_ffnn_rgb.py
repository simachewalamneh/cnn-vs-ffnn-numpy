import argparse, os, pickle, socket, tarfile, time, urllib.request
import numpy as np
import pandas as pd

from cnn_vs_ffnn import (Dense, ReLU, Flatten, Conv2D, MaxPool2, Model,
                          train, accuracy, per_class_prf, topk_acc, complexity, memory_footprint,
                          rotate as rotate_gray, translate as translate_gray, scale as scale_gray,
                          load_fashion)

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
IMG = 32     # CIFAR-10 image size (28 for Fashion-MNIST)
C_IN = 3     # RGB channels (1 for Fashion-MNIST)
IN_SHAPE = (C_IN, IMG, IMG)

# 1. DATA
def load_cifar10(data_dir="data"):
    socket.setdefaulttimeout(30)
    os.makedirs(data_dir, exist_ok=True)
    tgz = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extract_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extract_dir):
        if not os.path.exists(tgz):
            print("downloading CIFAR-10 (~163 MB)...")
            def hook(count, block_size, total_size):
                done = count * block_size
                pct = min(100, done * 100 / total_size) if total_size > 0 else 0
                print(f"\r{done/1e6:.1f} / {total_size/1e6:.1f} MB ({pct:.0f}%)", end="")
            urllib.request.urlretrieve(CIFAR_URL, tgz, reporthook=hook)
            print()
        with tarfile.open(tgz) as t:
            t.extractall(data_dir)

    def load_batch(fname):
        with open(os.path.join(extract_dir, fname), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y = np.array(d[b"labels"], dtype=np.int64)
        return x, y

    xtr_parts, ytr_parts = [], []
    for i in range(1, 6):
        x, y = load_batch(f"data_batch_{i}")
        xtr_parts.append(x); ytr_parts.append(y)
    xtr, ytr = np.concatenate(xtr_parts), np.concatenate(ytr_parts)
    xte, yte = load_batch("test_batch")
    return xtr, ytr, xte, yte


# 2. MODELS -- only the input size / channel count changes vs cnn_vs_ffnn.py
def build_ffnn_rgb(rng):
    return Model([Flatten(),
                  Dense(C_IN * IMG * IMG, 256, rng), ReLU(),   # 3*32*32 = 3072
                  Dense(256, 64, rng), ReLU(),
                  Dense(64, 10, rng)], "FFNN-RGB")


def build_cnn_rgb(rng):
    return Model([Conv2D(C_IN, 8, 3, rng), ReLU(), MaxPool2(),   # 32->30->15
                  Conv2D(8, 16, 3, rng), ReLU(), MaxPool2(),     # 15->13->6
                  Flatten(), Dense(16 * 6 * 6, 64, rng), ReLU(), Dense(64, 10, rng)], "CNN-RGB")

# 3. RGB-CAPABLE IMAGE TRANSFORMS
#    Generalised versions of cnn_vs_ffnn.py's rotate/translate/scale: same
#    bilinear-interpolation math, but operating on any (N, C, H, W), not just
#    (N, 1, 28, 28). Verified below to match the originals exactly.
def _warp_rgb(x, src_y, src_x):
    """x: (N,C,H,W); src_y/src_x: (H,W) source coordinates for each output pixel."""
    N, C, H, W = x.shape
    y0 = np.floor(src_y).astype(int); x0 = np.floor(src_x).astype(int); y1, x1 = y0 + 1, x0 + 1
    wy, wx = src_y - y0, src_x - x0

    def g(yy, xx):
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        yy_c, xx_c = np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)
        return x[:, :, yy_c, xx_c] * ok        # (N, C, H, W)
    out = (g(y0, x0) * (1 - wy) * (1 - wx) + g(y0, x1) * (1 - wy) * wx +
           g(y1, x0) * wy * (1 - wx) + g(y1, x1) * wy * wx)
    return out.astype(np.float32)


def _grid_rgb(H, W):
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    return yy - (H - 1) / 2, xx - (W - 1) / 2


def rotate_rgb(x, deg):
    H, W = x.shape[2:]; cy, cx = (H - 1) / 2, (W - 1) / 2
    a = np.deg2rad(deg); yy, xx = _grid_rgb(H, W); c, s = np.cos(a), np.sin(a)
    return _warp_rgb(x, c * yy - s * xx + cy, s * yy + c * xx + cx)


def translate_rgb(x, dx, dy=0):
    H, W = x.shape[2:]; cy, cx = (H - 1) / 2, (W - 1) / 2
    yy, xx = _grid_rgb(H, W); return _warp_rgb(x, yy + cy - dy, xx + cx - dx)


def scale_rgb(x, f):
    H, W = x.shape[2:]; cy, cx = (H - 1) / 2, (W - 1) / 2
    yy, xx = _grid_rgb(H, W); return _warp_rgb(x, yy / f + cy, xx / f + cx)


def verify_transforms_match_original():
    """Sanity check, run once at start-up: on 1x28x28 input the *_rgb functions
    must reproduce cnn_vs_ffnn.py's rotate/translate/scale exactly."""
    _, _, xte, _ = load_fashion()
    x = xte[:8]
    checks = [("rotate", rotate_gray(x, 15), rotate_rgb(x, 15)),
              ("translate", translate_gray(x, 2, 3), translate_rgb(x, 2, 3)),
              ("scale", scale_gray(x, 0.8), scale_rgb(x, 0.8))]
    for name, a, b in checks:
        d = np.abs(a - b).max()
        assert d == 0.0, f"{name}_rgb does not match the original {name}(): max diff {d}"
    print("transform check: rotate_rgb/translate_rgb/scale_rgb match cnn_vs_ffnn.py exactly (max diff 0.0)")

# 4. MAIN -- the same (A)-(F) report as Task 1, adapted for CIFAR-10

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ntrain", type=int, default=6000)
    ap.add_argument("--ntest", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--target_loss", type=float, default=1.8)
    ap.add_argument("--out", default="results_rgb")
    a = ap.parse_args()
    if a.quick: a.ntrain, a.ntest, a.epochs = 1000, 500, 3
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

    verify_transforms_match_original()

    Xtr, Ytr, Xte, Yte = load_cifar10()
    rs = np.random.default_rng(123)
    itr = rs.permutation(len(Xtr))[:a.ntrain]; ite = rs.permutation(len(Xte))[:a.ntest]
    xtr, ytr, xte, yte = Xtr[itr], Ytr[itr], Xte[ite], Yte[ite]
    print(f"CIFAR-10 subset: train={len(xtr)} test={len(xte)}  shape={xtr.shape}")

    builders = {"FFNN": build_ffnn_rgb, "CNN": build_cnn_rgb}
    models, hists = {}, {}
    for name, b in builders.items():
        print(f"\n=== Training {name} (RGB) ===")
        m = b(np.random.default_rng(0)); hists[name] = train(m, xtr, ytr, xte, yte, a.epochs); models[name] = m
        hists[name].to_csv(f"{a.out}/history_{name}_rgb.csv", index=False)

    # ---- (A) headline table ----
    rows = []
    for n, m in models.items():
        h = hists[n]; lg = m.predict_logits(xte); pred = lg.argmax(1); prf = per_class_prf(pred, yte)
        hit = h[h.train_loss <= a.target_loss]
        fl, macs, _ = complexity(m, in_shape=IN_SHAPE)                       # <-- fixed: explicit RGB shape
        t0 = time.perf_counter(); m.predict_logits(xte); inf_ms = (time.perf_counter() - t0) / len(xte) * 1000
        rows.append({"model": n, "params": m.n_params(),
                     "train_time_s": h.time_s.iloc[-1], "time/epoch_s": h.time_s.iloc[-1] / a.epochs,
                     "infer_ms/img": inf_ms,
                     "train_acc": h.train_acc.iloc[-1], "test_acc(top1)": h.test_acc.iloc[-1],
                     "top5_acc": topk_acc(lg, yte, 5),
                     "macro_precision": prf.precision.mean(), "macro_recall": prf.recall.mean(), "macro_F1": prf.f1.mean(),
                     "overfit_gap(train-test)": h.train_acc.iloc[-1] - h.test_acc.iloc[-1],
                     "test-acc peak-to-final drop": h.test_acc.max() - h.test_acc.iloc[-1],
                     f"epochs_to_loss<={a.target_loss}": int(hit.epoch.iloc[0]) if len(hit) else np.nan,
                     f"seconds_to_loss<={a.target_loss}": float(hit.time_s.iloc[0]) if len(hit) else np.nan,
                     "fwd_FLOPs/img": fl, "train_FLOPs/img(~3x fwd)": 3 * fl,
                     "connections(MACs)/img": macs, "connections_per_param": macs / m.n_params()})
        per_class_prf(pred, yte).to_csv(f"{a.out}/per_class_{n}_rgb.csv", index=False)
    head = pd.DataFrame(rows).set_index("model").T
    print("\n=== (A) HEADLINE COMPARISON (RGB) ==="); print(head.to_string(float_format=lambda v: f"{v:,.4f}"))
    head.to_csv(f"{a.out}/A_headline_rgb.csv")

    # ---- (B) architecture + memory ----
    print("\n=== (B) ARCHITECTURE (layer depth / types / params / FLOPs) ===")
    for n, m in models.items():
        _, _, df = complexity(m, in_shape=IN_SHAPE)                          # <-- fixed
        print(f"\n{n}:"); print(df.to_string(index=False)); df.to_csv(f"{a.out}/B_arch_{n}_rgb.csv", index=False)
    mem = pd.DataFrame({n: memory_footprint(m, in_shape=IN_SHAPE) for n, m in models.items()})  # <-- fixed
    print("\n=== (B2) MEMORY FOOTPRINT (float32, batch=64) ==="); print(mem.to_string(float_format=lambda v: f"{v:,.1f}"))
    mem.to_csv(f"{a.out}/B_memory_rgb.csv")

    # ---- (C) invariance under transformations ----
    tests = [("original", lambda x: x)]
    tests += [(f"rotate {d}°", (lambda d: lambda x: rotate_rgb(x, d))(d)) for d in (15, 30, 45, 60, 90, 180)]
    tests += [(f"shift {s}px", (lambda s: lambda x: translate_rgb(x, s, s))(s)) for s in (1, 2, 3, 4)]
    tests += [(f"scale x{f}", (lambda f: lambda x: scale_rgb(x, f))(f)) for f in (0.7, 0.85, 1.15, 1.3)]
    inv = pd.DataFrame({n: {t: accuracy(m, fn(xte), yte) for t, fn in tests} for n, m in models.items()})
    inv["CNN - FFNN"] = inv.CNN - inv.FFNN
    inv["FFNN retained %"] = 100 * inv.FFNN / inv.FFNN.iloc[0]; inv["CNN retained %"] = 100 * inv.CNN / inv.CNN.iloc[0]
    print("\n=== (C) SPATIAL INVARIANCE: test accuracy after transforming the test set ==="); print(inv.to_string(float_format=lambda v: f"{v:,.3f}"))
    inv.to_csv(f"{a.out}/C_invariance_rgb.csv")

    # ---- (D) data-scaling efficiency ----
    sizes = [250, 500, 1000, 2000, 4000, a.ntrain] if not a.quick else [250, 500]
    sizes = sorted(set(s for s in sizes if s <= a.ntrain)); ds_ep = max(3, a.epochs // 2) if not a.quick else 2
    ds = {}
    print(f"\n=== (D) DATA SCALING ({ds_ep} epochs each) ===")
    for n, b in builders.items():
        ds[n] = {}
        for s in sizes:
            m = b(np.random.default_rng(0)); h = train(m, xtr[:s], ytr[:s], xte, yte, ds_ep, verbose=False)
            ds[n][s] = h.test_acc.iloc[-1]; ds[n][f"gap@{s}"] = h.train_acc.iloc[-1] - h.test_acc.iloc[-1]
            print(f"  {n:4s} n={s:5d}  test={ds[n][s]:.3f}  train-test gap={ds[n][f'gap@{s}']:.3f}", flush=True)
    dsdf = pd.DataFrame({n: {s: ds[n][s] for s in sizes} for n in builders}); dsdf.index.name = "train_size"
    dsdf["CNN - FFNN"] = dsdf.CNN - dsdf.FFNN
    gap = pd.DataFrame({f"{n}_gap": {s: ds[n][f"gap@{s}"] for s in sizes} for n in builders})
    dsdf = dsdf.join(gap); print("\n", dsdf.to_string(float_format=lambda v: f"{v:,.3f}")); dsdf.to_csv(f"{a.out}/D_data_scaling_rgb.csv")

    # ---- (F) objects in RANDOM positions: train AND test on randomly shifted images (+-3 px) ----
    def random_shift(x, mx, seed):
        r = np.random.default_rng(seed); out = np.empty_like(x)
        for i in range(len(x)):
            out[i] = translate_rgb(x[i:i + 1], int(r.integers(-mx, mx + 1)), int(r.integers(-mx, mx + 1)))[0]
        return out
    ep_f = max(3, a.epochs // 2) if not a.quick else 2
    sx_tr, sx_te = random_shift(xtr, 3, 1), random_shift(xte, 3, 2)
    rowsF = {}
    print(f"\n=== (F) RANDOMLY SHIFTED DATA (+-3px), train & test both shifted, {ep_f} epochs ===")
    for n, b in builders.items():
        m = b(np.random.default_rng(0)); h = train(m, sx_tr, ytr, sx_te, yte, ep_f, verbose=False)
        rowsF[n] = dict(train_acc=h.train_acc.iloc[-1], test_acc=h.test_acc.iloc[-1], gap=h.train_acc.iloc[-1] - h.test_acc.iloc[-1],
                        params=m.n_params(), acc_on_centered_test=accuracy(m, xte, yte))
    dfF = pd.DataFrame(rowsF); print(dfF.to_string(float_format=lambda v: f"{v:,.3f}")); dfF.to_csv(f"{a.out}/F_shifted_data_rgb.csv")

    # ---- (E) feature hierarchy: RGB kernels + feature maps ----
    print("\n=== (E) FEATURE HIERARCHY: receptive field of one unit ===")
    print("  FFNN: layer-1 unit sees 32x32x3 = whole image (global, no locality); every deeper layer mixes global features.")
    print("  CNN : conv1 unit 3x3 | pool1 4x4 | conv2 unit 8x8 | pool2 10x10 | dense sees whole image (built from local parts)")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cnn, ff = models["CNN"], models["FFNN"]; x0 = xte[:1]
        c1 = cnn.layers[0].forward(x0, False); r1 = np.maximum(c1, 0); p1 = cnn.layers[2].forward(r1, False)
        c2 = np.maximum(cnn.layers[3].forward(p1, False), 0)

        def rgb_thumb(w):     # w: (3, k, k) -> normalised (k, k, 3) for imshow
            w = w.transpose(1, 2, 0)
            return (w - w.min()) / (w.max() - w.min() + 1e-8)

        fig, ax = plt.subplots(4, 8, figsize=(12, 6.5))
        for i in range(8):
            ax[0, i].imshow(rgb_thumb(cnn.layers[0].W[i]))            # RGB kernel, not grayscale: 3 input channels
            ax[1, i].imshow(r1[0, i], cmap="gray")
            ax[2, i].imshow(c2[0, i], cmap="gray")
            ax[3, i].imshow(rgb_thumb(ff.layers[1].W[:, i].reshape(C_IN, IMG, IMG)))
        for r, t in enumerate(["CNN conv1 kernels (3x3, RGB): edges/colour", "CNN conv1 feature maps (of 1 image)",
                               "CNN conv2 feature maps: parts/textures", "FFNN layer-1 weights as 32x32 RGB (unstructured)"]):
            ax[r, 0].set_ylabel(t, fontsize=7, rotation=0, ha="right", va="center")
        for a_ in ax.ravel(): a_.set_xticks([]); a_.set_yticks([])
        plt.suptitle(f"Feature hierarchy (RGB)  (input class: {CLASSES[yte[0]]})"); plt.tight_layout()
        plt.savefig(f"{a.out}/feature_hierarchy_rgb.png", dpi=130); print("  saved feature_hierarchy_rgb.png")

        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for n in models: ax[0].plot(hists[n].epoch, hists[n].train_loss, label=n); ax[1].plot(hists[n].epoch, hists[n].test_acc, label=n)
        ax[0].set_title("Training loss"); ax[1].set_title("Test accuracy"); ax[0].set_xlabel("epoch"); ax[1].set_xlabel("epoch")
        for n in ("FFNN", "CNN"): ax[2].plot(inv.index, inv[n], marker="o", label=n)
        ax[2].set_title("Accuracy under transformations"); ax[2].tick_params(axis="x", rotation=80)
        for a_ in ax: a_.legend(); a_.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{a.out}/curves_rgb.png", dpi=130); print("  saved curves_rgb.png")
    except ImportError:
        print("  (matplotlib not installed - figures skipped; tables are still saved as CSV)")
    print("\nDone. All tables saved in ./" + a.out)


if __name__ == "__main__":
    main()