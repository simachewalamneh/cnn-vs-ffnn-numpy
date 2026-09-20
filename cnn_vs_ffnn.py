import argparse, gzip, os, time, tracemalloc, urllib.request
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# 1. DATA  (Fashion-MNIST: 28x28 grayscale, 10 clothing classes)
URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_fashion(data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    for f in FILES:
        p = os.path.join(data_dir, f)
        if not os.path.exists(p):
            print("downloading", f)
            urllib.request.urlretrieve(URL + f, p)

    def imgs(f):
        with gzip.open(os.path.join(data_dir, f), "rb") as fh:
            return np.frombuffer(fh.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)

    def labs(f):
        with gzip.open(os.path.join(data_dir, f), "rb") as fh:
            return np.frombuffer(fh.read(), np.uint8, offset=8)

    xtr, ytr, xte, yte = imgs(FILES[0]), labs(FILES[1]), imgs(FILES[2]), labs(FILES[3])
    return (xtr / 255.0).astype(np.float32), ytr, (xte / 255.0).astype(np.float32), yte


# 2. LAYERS  (each has forward / backward and a list of (param, grad) pairs)
class Dense:
    def __init__(self, n_in, n_out, rng):
        self.W = (rng.standard_normal((n_in, n_out)) * np.sqrt(2.0 / n_in)).astype(np.float32)  # He init
        self.b = np.zeros(n_out, np.float32)
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)
        self.n_in, self.n_out = n_in, n_out

    def forward(self, x, train=True):
        self.x = x if train else None
        return x @ self.W + self.b

    def backward(self, d):
        self.dW[:] = self.x.T @ d
        self.db[:] = d.sum(0)
        return d @ self.W.T

    def params(self): return [(self.W, self.dW), (self.b, self.db)]
    def flops(self, in_shape): return 2 * self.n_in * self.n_out, (self.n_out,), self.n_in * self.n_out


class ReLU:
    def forward(self, x, train=True):
        self.mask = x > 0 if train else None
        return np.maximum(x, 0)

    def backward(self, d): return d * self.mask
    def params(self): return []
    def flops(self, s): return int(np.prod(s)), s, 0


class Flatten:
    def forward(self, x, train=True):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d): return d.reshape(self.shape)
    def params(self): return []
    def flops(self, s): return 0, (int(np.prod(s)),), 0


class Conv2D:
    """Valid convolution, stride 1.  Implemented as im2col + one big matrix multiply."""
    def __init__(self, c_in, c_out, k, rng):
        self.W = (rng.standard_normal((c_out, c_in, k, k)) * np.sqrt(2.0 / (c_in * k * k))).astype(np.float32)
        self.b = np.zeros(c_out, np.float32)
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)
        self.c_in, self.c_out, self.k = c_in, c_out, k

    def forward(self, x, train=True):
        N, C, H, W = x.shape; k = self.k
        Ho, Wo = H - k + 1, W - k + 1
        win = sliding_window_view(x, (k, k), axis=(2, 3))                    # N,C,Ho,Wo,k,k  (view)
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N * Ho * Wo, C * k * k)  # im2col
        out = cols @ self.W.reshape(self.c_out, -1).T + self.b               # (N*Ho*Wo, F)
        self.cols = cols if train else None
        self.xshape = x.shape
        return out.reshape(N, Ho, Wo, self.c_out).transpose(0, 3, 1, 2)

    def backward(self, d):
        N, F, Ho, Wo = d.shape; k = self.k; C = self.c_in
        d2 = d.transpose(0, 2, 3, 1).reshape(-1, F)
        self.dW[:] = (d2.T @ self.cols).reshape(self.W.shape)
        self.db[:] = d2.sum(0)
        dcols = (d2 @ self.W.reshape(F, -1)).reshape(N, Ho, Wo, C, k, k)
        dx = np.zeros(self.xshape, np.float32)
        for i in range(k):                     # "col2im": scatter-add each kernel offset
            for j in range(k):
                dx[:, :, i:i + Ho, j:j + Wo] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dx

    def params(self): return [(self.W, self.dW), (self.b, self.db)]

    def flops(self, s):
        C, H, W = s; Ho, Wo = H - self.k + 1, W - self.k + 1
        macs = self.c_out * Ho * Wo * C * self.k * self.k
        return 2 * macs, (self.c_out, Ho, Wo), macs


class MaxPool2:
    def forward(self, x, train=True):
        N, C, H, W = x.shape; H2, W2 = H // 2, W // 2
        xr = x[:, :, :H2 * 2, :W2 * 2].reshape(N, C, H2, 2, W2, 2)
        out = xr.max(axis=(3, 5))
        if train:
            self.mask = (xr == out[:, :, :, None, :, None]).astype(np.float32)
            self.mask /= self.mask.sum(axis=(3, 5), keepdims=True)           # split ties equally
            self.shape = x.shape
        return out

    def backward(self, d):
        N, C, H2, W2 = d.shape
        dx = np.zeros(self.shape, np.float32)
        dx[:, :, :H2 * 2, :W2 * 2] = (self.mask * d[:, :, :, None, :, None]).reshape(N, C, H2 * 2, W2 * 2)
        return dx

    def params(self): return []
    def flops(self, s): C, H, W = s; o = (C, H // 2, W // 2); return int(np.prod(s)), o, 0


class Model:
    def __init__(self, layers, name):
        self.layers, self.name = layers, name

    def forward(self, x, train=True):
        for l in self.layers: x = l.forward(x, train)
        return x

    def backward(self, d):
        for l in reversed(self.layers): d = l.backward(d)

    def params(self): return [pg for l in self.layers for pg in l.params()]
    def n_params(self): return int(sum(p.size for p, _ in self.params()))

    def predict_logits(self, x, bs=500):
        return np.vstack([self.forward(x[i:i + bs], train=False) for i in range(0, len(x), bs)])


def build_ffnn(rng):
    return Model([Flatten(), Dense(784, 128, rng), ReLU(), Dense(128, 64, rng), ReLU(), Dense(64, 10, rng)], "FFNN")


def build_cnn(rng):
    return Model([Conv2D(1, 8, 3, rng), ReLU(), MaxPool2(),      # 28->26->13
                  Conv2D(8, 16, 3, rng), ReLU(), MaxPool2(),     # 13->11->5
                  Flatten(), Dense(16 * 5 * 5, 64, rng), ReLU(), Dense(64, 10, rng)], "CNN")


# 3. LOSS + OPTIMISER + TRAINING LOOP
def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def cross_entropy(logits, y):
    p = softmax(logits)
    loss = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    g = p.copy(); g[np.arange(len(y)), y] -= 1
    return loss, g / len(y)


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p, self.lr, self.b1, self.b2, self.eps, self.t = params, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p, _ in params]; self.v = [np.zeros_like(p) for p, _ in params]

    def step(self):
        self.t += 1
        for i, (p, g) in enumerate(self.p):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t); vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


def accuracy(model, x, y): return float((model.predict_logits(x).argmax(1) == y).mean())


def train(model, xtr, ytr, xte, yte, epochs, bs=64, lr=1e-3, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    opt = Adam(model.params(), lr)
    hist, t_cum = [], 0.0
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        idx = rng.permutation(len(xtr)); losses = []
        for i in range(0, len(xtr), bs):
            b = idx[i:i + bs]
            loss, g = cross_entropy(model.forward(xtr[b]), ytr[b])
            model.backward(g); opt.step(); losses.append(loss)
        t_cum += time.perf_counter() - t0                       # only training time is counted
        rec = dict(epoch=ep, train_loss=float(np.mean(losses)), train_acc=accuracy(model, xtr, ytr),
                   test_acc=accuracy(model, xte, yte), time_s=t_cum)
        hist.append(rec)
        if verbose:
            print(f"  [{model.name}] ep {ep:2d}  loss {rec['train_loss']:.3f}  train {rec['train_acc']:.3f}  "
                  f"test {rec['test_acc']:.3f}  t={t_cum:6.1f}s", flush=True)
    return pd.DataFrame(hist)


# 4. METRICS
def topk_acc(logits, y, k):
    return float((np.argsort(-logits, 1)[:, :k] == y[:, None]).any(1).mean())


def per_class_prf(pred, y, n=10):
    rows = []
    for c in range(n):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        rows.append(dict(cls=CLASSES[c], precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0.0))
    return pd.DataFrame(rows)


def complexity(model, in_shape=(1, 28, 28)):
    """Analytic FLOPs (2*MACs per forward pass / sample), #connections (MACs), params, layer summary."""
    s, fl, macs, rows = in_shape, 0, 0, []
    for l in model.layers:
        f, s_out, m = l.flops(s)
        fl += f; macs += m
        rows.append((type(l).__name__, str(s), str(s_out), sum(p.size for p, _ in l.params()), f))
        s = s_out
    return fl, macs, pd.DataFrame(rows, columns=["layer", "in_shape", "out_shape", "params", "fwd_flops"])


def memory_footprint(model, batch=64, in_shape=(1, 28, 28)):
    """Bytes (float32). weights + grads + Adam state (2x) + stored activations for one training batch."""
    P = model.n_params() * 4
    x = np.zeros((batch, *in_shape), np.float32)
    act = x.nbytes; peak_infer = x.nbytes
    for l in model.layers:
        x = l.forward(x, train=False)
        act += x.nbytes; peak_infer = max(peak_infer, x.nbytes)
    tracemalloc.start()
    xb = np.random.rand(batch, *in_shape).astype(np.float32); yb = np.zeros(batch, int)
    _, g = cross_entropy(model.forward(xb), yb); model.backward(g)
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return dict(weights_KB=P / 1024, grads_KB=P / 1024, adam_state_KB=2 * P / 1024,
                activations_train_KB=act / 1024, peak_train_step_measured_KB=peak / 1024,
                inference_peak_activation_KB=peak_infer / 1024)


# 5. IMAGE TRANSFORMS (hand-written inverse mapping + bilinear interpolation)
def _warp(x, src_y, src_x):
    """x: (N,1,H,W); src_y/src_x: (H,W) float coordinates in the source image for each output pixel."""
    H, W = x.shape[2:]
    y0 = np.floor(src_y).astype(int); x0 = np.floor(src_x).astype(int); y1, x1 = y0 + 1, x0 + 1
    wy, wx = src_y - y0, src_x - x0

    def g(yy, xx):
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        return x[:, 0][:, np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)] * ok
    out = (g(y0, x0) * (1 - wy) * (1 - wx) + g(y0, x1) * (1 - wy) * wx +
           g(y1, x0) * wy * (1 - wx) + g(y1, x1) * wy * wx)
    return out[:, None].astype(np.float32)


def _grid(H=28, W=28):
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    return yy - (H - 1) / 2, xx - (W - 1) / 2


def rotate(x, deg):
    a = np.deg2rad(deg); yy, xx = _grid(); c, s = np.cos(a), np.sin(a)
    return _warp(x, c * yy - s * xx + 13.5, s * yy + c * xx + 13.5)


def translate(x, dx, dy=0):
    yy, xx = _grid(); return _warp(x, yy + 13.5 - dy, xx + 13.5 - dx)


def scale(x, f):
    yy, xx = _grid(); return _warp(x, yy / f + 13.5, xx / f + 13.5)


# 6. EXPERIMENTS
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ntrain", type=int, default=6000)
    ap.add_argument("--ntest", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--target_loss", type=float, default=0.45)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.quick: a.ntrain, a.ntest, a.epochs = 1000, 500, 3
    os.makedirs(a.out, exist_ok=True)
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

    Xtr, Ytr, Xte, Yte = load_fashion()
    rs = np.random.default_rng(123)
    itr = rs.permutation(len(Xtr))[:a.ntrain]; ite = rs.permutation(len(Xte))[:a.ntest]
    xtr, ytr, xte, yte = Xtr[itr], Ytr[itr], Xte[ite], Yte[ite]
    print(f"Fashion-MNIST subset: train={len(xtr)} test={len(xte)}")

    builders = {"FFNN": build_ffnn, "CNN": build_cnn}
    models, hists = {}, {}
    for name, b in builders.items():
        print(f"\n=== Training {name} ===")
        m = b(np.random.default_rng(0)); hists[name] = train(m, xtr, ytr, xte, yte, a.epochs); models[name] = m
        hists[name].to_csv(f"{a.out}/history_{name}.csv", index=False)

    # ---- (A) headline table: accuracy, time, params, overfitting, top-k, P/R/F1 ----
    rows = []
    for n, m in models.items():
        h = hists[n]; lg = m.predict_logits(xte); pred = lg.argmax(1); prf = per_class_prf(pred, yte)
        hit = h[h.train_loss <= a.target_loss]
        fl, macs, _ = complexity(m)
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
        per_class = per_class_prf(pred, yte); per_class.to_csv(f"{a.out}/per_class_{n}.csv", index=False)
    head = pd.DataFrame(rows).set_index("model").T
    print("\n=== (A) HEADLINE COMPARISON ==="); print(head.to_string(float_format=lambda v: f"{v:,.4f}"))
    head.to_csv(f"{a.out}/A_headline.csv")

    # ---- (B) architecture + memory ----
    print("\n=== (B) ARCHITECTURE (layer depth / types / params / FLOPs) ===")
    for n, m in models.items():
        _, _, df = complexity(m); print(f"\n{n}:"); print(df.to_string(index=False)); df.to_csv(f"{a.out}/B_arch_{n}.csv", index=False)
    mem = pd.DataFrame({n: memory_footprint(m) for n, m in models.items()})
    print("\n=== (B2) MEMORY FOOTPRINT (float32, batch=64) ==="); print(mem.to_string(float_format=lambda v: f"{v:,.1f}"))
    mem.to_csv(f"{a.out}/B_memory.csv")

    # ---- (C) invariance under transformations (trained on ORIGINAL images only) ----
    tests = [("original", lambda x: x)]
    tests += [(f"rotate {d}°", (lambda d: lambda x: rotate(x, d))(d)) for d in (15, 30, 45, 60, 90, 180)]
    tests += [(f"shift {s}px", (lambda s: lambda x: translate(x, s, s))(s)) for s in (1, 2, 3, 4)]
    tests += [(f"scale x{f}", (lambda f: lambda x: scale(x, f))(f)) for f in (0.7, 0.85, 1.15, 1.3)]
    inv = pd.DataFrame({n: {t: accuracy(m, fn(xte), yte) for t, fn in tests} for n, m in models.items()})
    inv["CNN - FFNN"] = inv.CNN - inv.FFNN
    inv["FFNN retained %"] = 100 * inv.FFNN / inv.FFNN.iloc[0]; inv["CNN retained %"] = 100 * inv.CNN / inv.CNN.iloc[0]
    print("\n=== (C) SPATIAL INVARIANCE: test accuracy after transforming the test set ==="); print(inv.to_string(float_format=lambda v: f"{v:,.3f}"))
    inv.to_csv(f"{a.out}/C_invariance.csv")

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
    dsdf = dsdf.join(gap); print("\n", dsdf.to_string(float_format=lambda v: f"{v:,.3f}")); dsdf.to_csv(f"{a.out}/D_data_scaling.csv")

    # ---- (F) objects in RANDOM positions: train AND test on randomly shifted images (+-3 px) ----
    def random_shift(x, mx, seed):
        r = np.random.default_rng(seed); out = np.empty_like(x)
        for i in range(len(x)):
            out[i] = translate(x[i:i + 1], int(r.integers(-mx, mx + 1)), int(r.integers(-mx, mx + 1)))[0]
        return out
    ep_f = max(3, a.epochs // 2) if not a.quick else 2
    sx_tr, sx_te = random_shift(xtr, 3, 1), random_shift(xte, 3, 2)
    rowsF = {}
    print(f"\n=== (F) RANDOMLY SHIFTED DATA (+-3px), train & test both shifted, {ep_f} epochs ===")
    for n, b in builders.items():
        m = b(np.random.default_rng(0)); h = train(m, sx_tr, ytr, sx_te, yte, ep_f, verbose=False)
        rowsF[n] = dict(train_acc=h.train_acc.iloc[-1], test_acc=h.test_acc.iloc[-1], gap=h.train_acc.iloc[-1] - h.test_acc.iloc[-1],
                        params=m.n_params(), acc_on_centered_test=accuracy(m, xte, yte))
    dfF = pd.DataFrame(rowsF); print(dfF.to_string(float_format=lambda v: f"{v:,.3f}")); dfF.to_csv(f"{a.out}/F_shifted_data.csv")

    # ---- (E) feature hierarchy: receptive fields + figure ----
    print("\n=== (E) FEATURE HIERARCHY: receptive field of one unit ===")
    print("  FFNN: layer-1 unit sees 28x28 = whole image (global, no locality); every deeper layer is a mix of global features.")
    print("  CNN : conv1 unit 3x3 | pool1 4x4 | conv2 unit 8x8 | pool2 10x10 | dense sees whole image (built from local parts)")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cnn, ff = models["CNN"], models["FFNN"]; x0 = xte[:1]
        c1 = cnn.layers[0].forward(x0, False); r1 = np.maximum(c1, 0); p1 = cnn.layers[2].forward(r1, False)
        c2 = np.maximum(cnn.layers[3].forward(p1, False), 0)
        fig, ax = plt.subplots(4, 8, figsize=(12, 6.5))
        for i in range(8):
            ax[0, i].imshow(cnn.layers[0].W[i, 0], cmap="RdBu"); ax[1, i].imshow(r1[0, i], cmap="gray"); ax[2, i].imshow(c2[0, i], cmap="gray")
            ax[3, i].imshow(ff.layers[1].W[:, i].reshape(28, 28), cmap="RdBu")
        for r, t in enumerate(["CNN conv1 kernels (3x3): edges/orientations", "CNN conv1 feature maps (of 1 image)",
                               "CNN conv2 feature maps: parts/textures", "FFNN layer-1 weights as 28x28 (unstructured)"]):
            ax[r, 0].set_ylabel(t, fontsize=7, rotation=0, ha="right", va="center")
        for a_ in ax.ravel(): a_.set_xticks([]); a_.set_yticks([])
        plt.suptitle(f"Feature hierarchy  (input class: {CLASSES[yte[0]]})"); plt.tight_layout()
        plt.savefig(f"{a.out}/feature_hierarchy.png", dpi=130); print("  saved feature_hierarchy.png")
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for n in models: ax[0].plot(hists[n].epoch, hists[n].train_loss, label=n); ax[1].plot(hists[n].epoch, hists[n].test_acc, label=n)
        ax[0].set_title("Training loss"); ax[1].set_title("Test accuracy"); ax[0].set_xlabel("epoch"); ax[1].set_xlabel("epoch")
        for n in ("FFNN", "CNN"): ax[2].plot(inv.index, inv[n], marker="o", label=n)
        ax[2].set_title("Accuracy under transformations"); ax[2].tick_params(axis="x", rotation=80)
        for a_ in ax: a_.legend(); a_.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{a.out}/curves.png", dpi=130); print("  saved curves.png")
    except ImportError:
        print("  (matplotlib not installed - figures skipped; tables are still saved as CSV)")
    print("\nDone. All tables saved in ./" + a.out)


if __name__ == "__main__":
    main()
