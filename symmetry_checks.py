import json, os
import numpy as np
from cnn_vs_ffnn import load_fashion, scale, CLASSES

OUT = "results/report_figs"; os.makedirs(OUT, exist_ok=True)
R = {}                                            # everything we measure goes here (saved as JSON)
rng = np.random.default_rng(0)


# ---------- basic operators (float64, written for clarity not speed) -------------------------------
def shift(x, dy, dx):            # (T_s x)(u) = x(u - s), circular
    return np.roll(x, (dy, dx), axis=(0, 1))

def corr_circ(x, w):             # (x * w)(u) = sum_v x(u+v) w(v),  circular boundary, w is k x k on v in [0,k)
    k = w.shape[0]; out = np.zeros_like(x)
    for i in range(k):
        for j in range(k):
            out += w[i, j] * np.roll(x, (-i, -j), axis=(0, 1))
    return out

def corr_valid(x, w):            # same but 'valid' (what our CNN layer does): output (H-k+1, W-k+1)
    k = w.shape[0]; H, W = x.shape; out = np.zeros((H - k + 1, W - k + 1))
    for i in range(k):
        for j in range(k):
            out += w[i, j] * x[i:i + H - k + 1, j:j + W - k + 1]
    return out

def maxpool2(x):                 # non-overlapping 2x2, stride 2
    H, W = x.shape; return x[:H // 2 * 2, :W // 2 * 2].reshape(H // 2, 2, W // 2, 2).max(axis=(1, 3))

relu = lambda z: np.maximum(z, 0)
rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


# ---------- test image: a real Fashion-MNIST item, shrunk so it has empty margin (no border effects) ----
Xtr, Ytr, _, _ = load_fashion()
img = Xtr[np.where(Ytr == 9)[0][0], 0].astype(np.float64)            # an ankle boot
x = scale(img[None, None].astype(np.float32), 0.4)[0, 0].astype(np.float64)
w_edge = np.array([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]])       # Sobel: vertical-edge detector
cls = CLASSES[Ytr[np.where(Ytr == 9)[0][0]]]

# ---------- 1. TRANSLATION EQUIVARIANCE  (Theorem 1) -----------------------------------------------
s = (3, 5)
R["1_circular_equiv_maxerr"] = float(np.abs(corr_circ(shift(x, *s), w_edge) - shift(corr_circ(x, w_edge), *s)).max())
# 'valid' convolution (our layer): equivariant wherever the object stays inside the frame
a = corr_valid(shift(x, *s), w_edge); b = shift(corr_valid(x, w_edge), *s)
R["1_valid_equiv_maxerr_object_inside"] = float(np.abs(a - b).max())
# a fully-connected layer with the same input/output size has NO such property
Wfc = rng.standard_normal((784, 784)) / 28
f = lambda z: (z.reshape(-1) @ Wfc).reshape(28, 28)
R["1_dense_equiv_relerr"] = rel(f(shift(x, *s)), shift(f(x), *s))
# ... and Theorem 2: build the 'matrix' of a convolution and check it is circulant/Toeplitz-structured (weight sharing)
n = 8; e = np.zeros(n); e[0] = 1; hk = np.array([1., 2., 3.] + [0.] * (n - 3))
C = np.stack([np.roll(hk, j) for j in range(n)], 1)                     # circulant matrix of a length-3 kernel
Sh = np.roll(np.eye(n), 1, axis=0)                                       # shift matrix
R["2_circulant_commutes_with_shift_err"] = float(np.abs(C @ Sh - Sh @ C).max())
R["2_random_dense_commutes_with_shift_err"] = float(np.abs(rng.standard_normal((n, n)) @ Sh - Sh @ rng.standard_normal((n, n))).max())
# number of free parameters: dense map 784->784 vs 3x3 convolution
R["params_dense_784x784"] = 784 * 784; R["params_conv_3x3"] = 9

# ---------- 2. POOLING: approximate invariance, aliasing -------------------------------------------
Wk = np.abs(rng.standard_normal((8, 3, 3)))                               # 8 random positive kernels
def feats(xx, mode):
    maps = [maxpool2(relu(corr_valid(xx, k))) for k in Wk]                  # conv -> relu -> maxpool(2)
    return np.concatenate([m.ravel() for m in maps]) if mode == "flatten" else np.array([m.sum() for m in maps])
rows = []
for sh in range(0, 7):
    xs = shift(x, sh, sh)                                                    # diagonal shift of sh pixels
    rows.append(dict(shift_px=sh,
                     flatten_relchange=rel(feats(xs, "flatten"), feats(x, "flatten")),
                     globalsum_relchange=rel(feats(xs, "gsum"), feats(x, "gsum")),
                     raw_pixels_relchange=rel(xs, x)))
R["3_pool_table"] = rows
# exact equivariance of pooling for shifts that are multiples of the stride
z = relu(corr_valid(x, w_edge))[:26, :26]
R["3_pool_equiv_even_shift_err"] = float(np.abs(maxpool2(shift(z, 2, 2)) - shift(maxpool2(z), 1, 1)).max())
d = [np.abs(maxpool2(shift(z, 1, 1)) - shift(maxpool2(z), c, c)).max() for c in (0, 1)]
R["3_pool_odd_shift_min_err_over_coarse_shifts"] = float(min(d))          # >0  => aliasing, no coarse shift explains it

# ---------- 3. ROTATION: not equivariant; rot90 is, if the KERNEL rotates too ----------------------
xr = np.rot90(x)
target = np.rot90(corr_valid(x, w_edge))
R["4_rot90_plain_conv_relerr"] = rel(corr_valid(xr, w_edge), target)                       # same kernel: mismatch
R["4_rot90_rotated_kernel_relerr"] = rel(corr_valid(xr, np.rot90(w_edge)), target)          # rotate kernel too: exact
# p4 'lifting' convolution: one output map per kernel rotation r = 0..3
lift = lambda xx, w: np.stack([corr_valid(xx, np.rot90(w, r)) for r in range(4)])
L0, L1 = lift(x, w_edge), lift(xr, w_edge)
errs = {sgn: max(rel(L1[r], np.rot90(L0[(r + sgn) % 4])) for r in range(4)) for sgn in (-1, 1)}
sgn = min(errs, key=errs.get)
R["4_p4_lifting_equiv_err"] = float(errs[sgn]); R["4_p4_lifting_wrong_direction_err"] = float(errs[-sgn])
inv_plain = lambda xx: np.abs(corr_valid(xx, w_edge)).sum()
inv_p4 = lambda xx: lift(xx, w_edge).max(axis=0).sum()                                      # group-max, then global sum
R["4_plain_conv_globalsum_rot90_relchange"] = float(abs(inv_plain(xr) - inv_plain(x)) / inv_plain(x))
R["4_p4_groupmax_globalsum_rot90_relchange"] = float(abs(inv_p4(xr) - inv_p4(x)) / inv_p4(x))
# a 45-degree rotation is NOT a symmetry of the square grid -> no exact equivariance even for p4
from cnn_vs_ffnn import rotate
x45 = rotate(x[None, None].astype(np.float32), 45)[0, 0].astype(np.float64)
R["4_p4_groupmax_globalsum_rot45_relchange"] = float(abs(inv_p4(x45) - inv_p4(x)) / inv_p4(x))

# ---------- 4. SCALE SEPARATION: receptive fields --------------------------------------------------
def receptive_field(layers):                     # layers = [(name,k,stride)] ; r_l = r_{l-1} + (k-1) j_{l-1},  j_l = j_{l-1} s
    r, j, rows = 1, 1, []
    for name, k, st in layers:
        r = r + (k - 1) * j; j = j * st; rows.append((name, k, st, r, j))
    return rows
cnn_layers = [("conv1 3x3", 3, 1), ("pool1 2x2", 2, 2), ("conv2 3x3", 3, 1), ("pool2 2x2", 2, 2)]
R["5_receptive_field_cnn"] = receptive_field(cnn_layers)
R["5_receptive_field_no_pooling_5_convs"] = receptive_field([(f"conv{i}", 3, 1) for i in range(1, 6)])
# Empirical check: which input pixels influence ONE unit after pool2? (positive weights + large bump => monotone)
Wa, Wb = np.abs(rng.standard_normal((1, 3, 3))) + .1, np.abs(rng.standard_normal((1, 3, 3))) + .1
net = lambda xx: maxpool2(relu(corr_valid(maxpool2(relu(corr_valid(xx, Wa[0]))), Wb[0])))
base = net(np.abs(rng.standard_normal((28, 28))) * 0.1); ref = np.abs(rng.standard_normal((28, 28))) * 0.1
base = net(ref); unit = (2, 2); mask = np.zeros((28, 28), bool)
for i in range(28):
    for j in range(28):
        xx = ref.copy(); xx[i, j] += 100.0
        mask[i, j] = abs(net(xx)[unit] - base[unit]) > 1e-9
ii, jj = np.where(mask)
R["5_empirical_rf_of_pool2_unit"] = dict(n_pixels=int(mask.sum()), height=int(ii.max() - ii.min() + 1), width=int(jj.max() - jj.min() + 1))

json.dump(R, open(f"{OUT}/checks.json", "w"), indent=2, default=float)

# ---------- print ----------------------------------------------------------------------------------
print(f"test image: {cls}, shrunk to 0.4x so it has empty margin\n")
for k, v in R.items():
    if k == "3_pool_table":
        print("3_pool_table  (relative change of the feature vector when the image is shifted diagonally by s pixels)")
        print(f"  {'s':>2} {'raw pixels':>11} {'flatten(pool)':>14} {'global-sum(pool)':>17}")
        for r_ in v: print(f"  {r_['shift_px']:>2} {r_['raw_pixels_relchange']:>11.3f} {r_['flatten_relchange']:>14.3f} {r_['globalsum_relchange']:>17.4f}")
    else: print(f"{k}: {v}")

# ---------- figures --------------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})
def show(ax, im, t, cmap="gray", lim=None):
    kw = dict(vmin=-lim, vmax=lim) if lim else {}
    ax.imshow(im, cmap=cmap, **kw); ax.set_title(t); ax.set_xticks([]); ax.set_yticks([])

# Fig 1: translation equivariance
fig, ax = plt.subplots(2, 4, figsize=(10.5, 5.4))
xs = shift(x, *s); A = corr_circ(x, w_edge); B = corr_circ(xs, w_edge)
show(ax[0, 0], x, "x"); show(ax[0, 1], xs, f"T_s x   (shift s = {s})"); show(ax[0, 2], A, "x * w", "RdBu"); show(ax[0, 3], shift(A, *s), "T_s (x * w)", "RdBu")
ax[1, 0].axis("off"); show(ax[1, 1], B, "(T_s x) * w", "RdBu")
lim = float(np.abs(A).max())
show(ax[1, 2], B - shift(A, *s), f"difference  (max|.| = {R['1_circular_equiv_maxerr']:.1e})", "RdBu", lim)
D_ = f(shift(x, *s)) - shift(f(x), *s)
show(ax[1, 3], D_, f"same test, DENSE layer\n(relative error {R['1_dense_equiv_relerr']:.2f})", "RdBu", float(np.abs(D_).max()))
ax[1, 0].text(0.5, 0.5, "Convolution:\n(T_s x) * w  =  T_s (x * w)", ha="center", va="center", fontsize=11)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_translation_equivariance.png", dpi=150); plt.close()

# Fig 2: rotation
fig, ax = plt.subplots(1, 5, figsize=(12, 2.9))
show(ax[0], x, "x"); show(ax[1], xr, "rot90(x)")
show(ax[2], target, "TARGET: rot90( x * w )", "RdBu")
show(ax[3], corr_valid(xr, w_edge), f"rot90(x) * w   (same kernel)\nrel. error {R['4_rot90_plain_conv_relerr']:.2f}", "RdBu")
show(ax[4], corr_valid(xr, np.rot90(w_edge)), f"rot90(x) * rot90(w)\nrel. error {R['4_rot90_rotated_kernel_relerr']:.0e}", "RdBu")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_rotation.png", dpi=150); plt.close()

# Fig 3: receptive fields
ext_pool = receptive_field([("conv", 3, 1), ("pool", 2, 2)] * 4)                 # hypothetical 8-layer conv/pool stack
ext_plain = receptive_field([("conv", 3, 1)] * 8)                                 # 8 convs, no pooling
R["5_rf_8layers_with_pooling"] = [r[3] for r in ext_pool]; R["5_rf_8layers_no_pooling"] = [r[3] for r in ext_plain]
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
ax[0].plot(range(1, 9), [r[3] for r in ext_plain], "s--", color="C1", label="8 conv layers, no pooling (linear growth)")
ax[0].plot(range(1, 9), [r[3] for r in ext_pool], "o:", color="C0", label="conv-pool x4 (compounding growth)")
ax[0].plot(range(1, 5), [r[3] for r in R["5_receptive_field_cnn"]], "o-", color="C0", lw=2.5, label="our CNN (first 4 layers)")
ax[0].axhline(28, color="gray", ls=":"); ax[0].text(1.0, 29, "whole image (28 px)", color="gray", fontsize=8)
ax[0].set_xlabel("layer index"); ax[0].set_ylabel("receptive field (pixels)"); ax[0].set_title("Receptive field growth: scale separation via pooling")
ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(alpha=.3)
ax[1].imshow(mask, cmap="Blues"); ax[1].set_title(f"Input pixels that influence ONE unit after pool2\n(measured: {R['5_empirical_rf_of_pool2_unit']['height']}x{R['5_empirical_rf_of_pool2_unit']['width']} block, {R['5_empirical_rf_of_pool2_unit']['n_pixels']} pixels)")
ax[1].set_xticks([]); ax[1].set_yticks([])
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_receptive_field.png", dpi=150); plt.close()

# Fig 4: pooling invariance
fig, ax = plt.subplots(figsize=(6.2, 3.8))
sx = [r_["shift_px"] for r_ in R["3_pool_table"]]
ax.plot(sx, [r_["raw_pixels_relchange"] for r_ in R["3_pool_table"]], "k:o", label="raw pixels")
ax.plot(sx, [r_["flatten_relchange"] for r_ in R["3_pool_table"]], "r-s", label="conv+pool, flattened (like our CNN)")
ax.plot(sx, [r_["globalsum_relchange"] for r_ in R["3_pool_table"]], "g-^", label="conv+pool, then global sum")
ax.set_xlabel("diagonal shift (pixels)"); ax.set_ylabel("relative change of representation"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("How invariant is each representation to shifts?")
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_pooling_invariance.png", dpi=150); plt.close()

# Fig 5: measured robustness from the Task-1 experiment (full run)
import pandas as pd
if os.path.exists("results/C_invariance.csv"):
    C_ = pd.read_csv("results/C_invariance.csv", index_col=0)
    groups = [("Rotation", ["original"] + [f"rotate {d}°" for d in (15, 30, 45, 60, 90)], [0, 15, 30, 45, 60, 90], "angle (degrees)"),
              ("Shift", ["original"] + [f"shift {p}px" for p in (1, 2, 3, 4)], [0, 1, 2, 3, 4], "shift (pixels, both axes)"),
              ("Scale (zoom)", ["scale x0.7", "scale x0.85", "original", "scale x1.15", "scale x1.3"], [0.7, 0.85, 1.0, 1.15, 1.3], "zoom factor")]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for a_, (t, idx, xv, xl) in zip(ax, groups):
        a_.plot(xv, C_.loc[idx, "FFNN retained %"], "o-", label="FFNN"); a_.plot(xv, C_.loc[idx, "CNN retained %"], "s-", label="CNN")
        a_.set_title(t); a_.set_xlabel(xl); a_.grid(alpha=.3)
    ax[0].set_ylabel("accuracy retained (% of untransformed)"); ax[0].legend()
    plt.tight_layout(); plt.savefig(f"{OUT}/fig5_measured_robustness.png", dpi=150); plt.close()
print("\nfigures + checks.json written to", OUT)
json.dump(R, open(f"{OUT}/checks.json", "w"), indent=2, default=float)
