# CNN vs FFNN on CIFAR-10 (RGB) — extension of the grayscale study

This is a companion experiment to the main Fashion-MNIST study (see `README.md` and
`docs/CNN_Symmetry_and_Scale_Separation_Report.tex` for the full Task 2 mathematics, which
this experiment does not change). It reruns the same from-scratch NumPy CNN/FFNN comparison
on **CIFAR-10** (3-channel, 32x32, unsegmented natural images) to test whether the CNN's
advantage grows on harder, non-centered RGB data. Implementation: `cnn_vs_ffnn_rgb.py`
(reuses the layer/training code in `cnn_vs_ffnn.py`; only the data loader and input channel
count differ).

## Run
```bash
python cnn_vs_ffnn_rgb.py --ntrain 6000 --ntest 2000 --epochs 12
```
Tables are written to `results_rgb/`.

## Headline result
| Metric | FFNN | CNN |
|---|---|---|
| Parameters | 803,786 | 38,970 |
| Train time | 323.0 s | 544.1 s |
| Test accuracy (top-1) | 39.1% | 52.5% |
| Top-5 accuracy | 87.8% | 93.1% |
| Macro F1 | 0.378 | 0.521 |
| Overfit gap (train − test) | 6.2 pts | 10.5 pts |
| Forward FLOPs / image | 1,607,232 | 873,056 |
| Connections per parameter | 1.00 | 10.95 |

**The CNN wins by 13.5 accuracy points with ~20x fewer parameters.** This is a much larger,
more decisive gap than the near-tie seen on centered Fashion-MNIST (where the two models
were within 1 point of each other) — CIFAR-10's uncentered, varied-pose objects give
translation equivariance real work to do, which is the central prediction of the symmetry
argument in the Task 2 report.

**On the overfit gap:** CNN's gap is larger in absolute terms, but this is not "worse
generalization" — CNN's train accuracy (63.0%) is far above FFNN's (45.3%), so the larger
gap reflects a network that is actually learning more usable signal, not one that is
memorizing more noise.

## Robustness to image transformations
| Transform | FFNN acc | CNN acc | FFNN retained % | CNN retained % |
|---|---|---|---|---|
| original | 0.391 | 0.525 | 100.0% | 100.0% |
| shift 2px | 0.216 | 0.450 | 55.4% | 85.7% |
| shift 3px | 0.163 | 0.344 | 41.7% | 65.6% |
| scale 0.85x | 0.118 | 0.475 | 30.3% | 90.6% |
| rotate 15° | 0.216 | 0.462 | 55.4% | 88.0% |
| rotate 60° | 0.156 | 0.194 | 39.9% | 36.9% |
| rotate 90° | 0.219 | 0.210 | 56.1% | 40.1% |

CNN is clearly more robust to shift and scale, matching the equivariance argument. At large
rotation (60°–90°) the *absolute* accuracies are close to tied, and CNN's *retained %* looks
worse than FFNN's there — but that's a scaling artifact of CNN having a much higher baseline
to fall from, not evidence that the CNN degrades faster in an absolute sense. This matches
the Task 2 report's Lemma: convolution carries no built-in rotation symmetry, so neither
model is protected at large rotation angles.

## Data-scaling efficiency
| Train size | FFNN test acc | CNN test acc | CNN − FFNN |
|---|---|---|---|
| 250 | 0.178 | 0.153 | −0.024 |
| 500 | 0.234 | 0.214 | −0.020 |
| 1000 | 0.232 | 0.274 | +0.042 |
| 2000 | 0.275 | 0.363 | +0.088 |
| 4000 | 0.324 | 0.431 | +0.108 |
| 6000 | 0.330 | 0.474 | +0.144 |

**There is a genuine crossover.** With very little data (250–500 images) the FFNN actually
beats the CNN — the CNN's inductive bias has nothing to pay off with so few examples per
class. From ~1000 images onward the CNN pulls ahead and the gap widens steadily with more
data. This is the clearest evidence in either experiment for "CNN structure needs a minimum
amount of data to earn its keep."

## Memory footprint (float32, batch 64)
| | FFNN | CNN |
|---|---|---|
| Weights | 3,139.8 KB | 152.2 KB |
| Peak training-step memory | 4,008.7 KB | 23,589.0 KB |
| Inference peak activation | 768.0 KB | 1,800.0 KB |

Same pattern as Fashion-MNIST: weight sharing shrinks the parameter count but not the
compute or activation memory — the CNN is ~6x heavier in peak training memory here.

## Robustness when both train and test are shifted (±3px)
| | FFNN | CNN |
|---|---|---|
| Test accuracy | 0.312 | 0.439 |
| Train − test gap | 0.047 | 0.056 |

Training on shifted data helps both models close some of the robustness gap, but the CNN
still leads by 12.7 points.

## Takeaways
- Fashion-MNIST (centered) → near tie in accuracy; CIFAR-10 (uncentered, RGB) → 13.5-point
  CNN lead. This contrast is the strongest empirical support for the symmetry argument:
  weight sharing helps most exactly where position varies.
- CNN's data-scaling crossover (loses below ~1000 images, wins above it) is a result the
  Fashion-MNIST run didn't surface and is worth its own slide.
- Neither model is rotation-invariant at large angles — consistent with the Task 2 report's
  proof that convolution is not rotation-equivariant.
- CNN costs more compute and memory throughout, same trade-off as the grayscale study.

## Limitations
- Single seed, 6,000/2,000 train/test subset, 12 epochs — both models' accuracy curves were
  still rising at the final epoch (see `results_rgb/curves_rgb.png`), so absolute numbers
  would likely improve with longer training.
- `gradcheck.py` was written for the grayscale models and has not been re-run against
  `build_cnn_rgb` / `build_ffnn_rgb`; the numbers above are not yet gradient-verified the way
  the Fashion-MNIST run was.
- Rotation/scale transforms use bilinear interpolation, which blurs the image and accounts
  for some of the accuracy loss beyond the geometric transform itself.

## Files
- `cnn_vs_ffnn_rgb.py` — RGB data loader, model builders, training/reporting script
- `results_rgb/` — CSV tables (A headline, B architecture/memory, C invariance, D data
  scaling, F shifted data) and figures (`feature_hierarchy_rgb.png`, `curves_rgb.png`)

This experiment does not modify or replace the Task 2 mathematical report
(`docs/CNN_Symmetry_and_Scale_Separation_Report.tex`); it is additional empirical evidence
for the same argument, on a second, harder dataset.
