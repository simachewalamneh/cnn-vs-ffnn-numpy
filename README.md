# CNN vs FFNN from scratch (NumPy + pandas only)

Why are CNNs good at image classification? This project implements a **CNN** and a **feed-forward network (FFNN)**
using only NumPy and pandas (no PyTorch/TensorFlow) and compares them on **Fashion-MNIST**.

Everything is hand-written: dense and convolution layers (im2col), max-pooling, ReLU, softmax cross-entropy,
backprop, Adam, image transforms (rotate / shift / zoom with bilinear interpolation), metrics, FLOP and memory accounting.

## Run
```bash
pip install -r requirements.txt
python gradcheck.py                 # verifies backprop against numerical gradients
python cnn_vs_ffnn.py --quick       # ~5 s smoke test
python cnn_vs_ffnn.py --ntrain 10000 --ntest 3000 --epochs 15 --target_loss 0.40   # full run (~2.5 min)
```
The dataset downloads automatically into `data/` on first run. Tables are written to `results/`.

## Models
| | FFNN | CNN |
|---|---|---|
| Architecture | 784 -> 128 -> 64 -> 10 | Conv3x3(8) -> Pool -> Conv3x3(16) -> Pool -> 400 -> 64 -> 10 |
| Parameters | 109,386 | 27,562 |

## Results (10k train / 3k test, 15 epochs, one seed)
| Metric | FFNN | CNN |
|---|---|---|
| Test accuracy (top-1) | 85.5% | 85.1% |
| Top-5 accuracy | 99.6% | 99.7% |
| Macro F1 | 0.856 | 0.852 |
| Overfitting gap (train - test) | 7.1 pts | 5.2 pts |
| Training time | 4.1 s | 38.6 s |
| Forward FLOPs / image | 218k | 443k |
| Training activation memory (batch 64) | 490 KB | 4,440 KB |
| Connections per parameter | 1.0 | 7.8 |
| Accuracy after 15 deg rotation | 53% | 65% |
| Accuracy after 2 px shift | 41% | 52% |
| Accuracy after 0.7x zoom | 36% | 57% |
| Accuracy after 60 deg rotation | 6.7% | 7.0% |
| Trained and tested on random +-3 px shifts | 75.0% | 77.8% |

## Takeaways
- On centered images, accuracy is a tie (differences < 1 pt are within noise for 3,000 test images).
- The CNN reaches it with **4x fewer parameters**, overfits less, and degrades more gracefully under small shifts, zooms and rotations.
- It wins clearly when object position varies (translation equivariance from weight sharing + local invariance from pooling).
- The CNN costs more compute: ~9x slower to train here, ~2x FLOPs, ~9x activation memory. Weight sharing saves parameters, not operations.
- Neither model is rotation-invariant: both collapse near chance at 60 deg.

Limitations: single seed, small subset, small models, CPU only. Average several seeds before drawing fine-grained conclusions.

## Task 2: symmetry and scale separation (theory + numerical verification)
`Doc/CNN_Symmetry_and_Scale_Separation_Report.tex` (compiled PDF alongside it) explains, with proofs, why CNNs work: translation equivariance of convolution
(and why convolution is the *only* such linear layer), pooling and aliasing, why rotation and zoom are not covered, receptive-field growth
(scale separation), and stability to deformations. `symmetry_checks.py` verifies every claim numerically and regenerates the figures.
```bash
python symmetry_checks.py    # writes results/report_figs/*.png and checks.json (needs results/C_invariance.csv for fig 5)
```

## Extension: CIFAR-10 (RGB) experiment
The grayscale study above uses centered, single-channel Fashion-MNIST. `cnn_vs_ffnn_rgb.py` reruns the same
comparison on 3-channel, uncentered **CIFAR-10** images to test whether the CNN's advantage grows on harder,
real-world data — it does, going from a near-tie to a large, decisive gap. Full results, tables and discussion
are in **`README_rgb.md`**; this file and the Task 2 report above are unchanged by that experiment.
```bash
python cnn_vs_ffnn_rgb.py --ntrain 6000 --ntest 2000 --epochs 12
```

## File structure
```
cnn-vs-ffnn-numpy/
├── cnn_vs_ffnn.py                    # Task 1: CNN/FFNN models, training, experiments (Fashion-MNIST)
├── cnn_vs_ffnn_rgb.py                # CIFAR-10 (RGB) extension of Task 1 — reuses layers from cnn_vs_ffnn.py
├── gradcheck.py                      # numerical gradient check (grayscale models)
├── symmetry_checks.py                # numerical verification of the Task 2 theory + figure generation
├── receptive_field_visualization.py  # receptive-field measurement/visualization used in the Task 2 report
├── requirements.txt
├── README.md                         # this file — Task 1 + Task 2 overview
├── README_rgb.md                     # CIFAR-10/RGB experiment write-up and results
├── LICENSE
├── .gitignore
├── Doc/                               # Task 2 report (LaTeX source, compiled PDF, figures)
├── data/                              # downloaded datasets (Fashion-MNIST, CIFAR-10) — gitignored
├── results/                           # Task 1 grayscale CSV tables and figures
└── results_rgb/                       # CIFAR-10/RGB CSV tables and figures
```

## What's implemented so far
- **Task 1 (grayscale):** CNN and FFNN built from scratch in NumPy, trained and compared on Fashion-MNIST across
  accuracy, F1, training time, FLOPs, memory, overfitting, data-scaling efficiency, robustness to
  rotation/shift/zoom, and feature hierarchy. See Results above.
- **Task 2:** full mathematical report on symmetry and scale separation in CNNs — translation equivariance,
  the convolution uniqueness theorem, pooling/aliasing, rotation/scale limits, and the general group-theoretic
  (geometric deep learning) treatment of scale separation, each theorem paired with a numerical check against
  the Task 1 models. See `Doc/`.
- **RGB extension (in progress):** the same Task 1 comparison rerun on CIFAR-10 to test the theory on harder,
  uncentered, 3-channel data. See `README_rgb.md`. Not yet gradient-checked with `gradcheck.py`.

## Files
- `cnn_vs_ffnn.py` - all models, training and experiments (Task 1)
- `gradcheck.py` - numerical gradient check
- `symmetry_checks.py` - numerical verification of the Task 2 theory + figures
- `Doc/` - Task 2 report (LaTeX + PDF)
- `results/` - CSV tables (A headline, B architecture and memory, C invariance, D data scaling, F shifted data) and figures

## References
- LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.
- Krizhevsky, A., Sutskever, I., and Hinton, G. E. (2012). ImageNet Classification with Deep Convolutional Neural Networks. *Advances in Neural Information Processing Systems (NeurIPS)*, 25.
- Bronstein, M. M., Bruna, J., Cohen, T., and Veličković, P. (2021). Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. arXiv:2104.13478.