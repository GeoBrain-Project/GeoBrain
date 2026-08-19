# Part 2: build the earth

Part 3 puts physics on an earth model. This part builds one. Seven
scripts in workflow order, each a complete step with a measured claim,
and each answering a question a geologist actually asks rather than
demonstrating an algorithm for its own sake.

Three things every script inherits.

**Truths are synthetic, so every claim is checkable.** Six of the seven
build a field with a known variogram, sample it, and score the recovery
against the answer. That is the only way to say "this method is 24%
better" and mean it. The exception is the case study, which runs on a
built-in dataset, and says plainly what that dataset is.

**Conclusions are computed, not written down.** Where a script compares
methods, the sentence that names the winner is generated from the
numbers. A model change cannot leave a stale claim behind.

**Negative results are kept.** Declustering that changes nothing, a
neighbourhood sweep that is not monotonic, a proxy that helps at every
correlation when the textbook expects a threshold; these are printed as
measured, with the reason, because a gallery that only shows the method
working teaches you nothing about when it does not.

| # | Script | The step, and the one thing it teaches |
|---|---|---|
| 01 | `01_estimation_vs_simulation.py` | Kriging is the better single map (RMSE 10.3 against a realisation's 14.8); simulation is the better model of the earth (variance 202 against the kriged map's 100, truth 216). The kriged map carries 44% of the truth's semivariance, a realisation 83%: and kriging variance maps where the data are, while the ensemble spread maps what the outcome could be |
| 02 | `02_variogram.py` | Omnidirectional averages the anisotropy away; binning by azimuth recovers a 3.0 ratio against a true 3.1. The angular tolerance does not simply improve as it narrows, and refitting from 60, 120 and 400 samples shows the variogram's own uncertainty: which every downstream weight inherits |
| 03 | `03_kriging_estimators.py` | The estimators differ in what they assume about the MEAN. A trend inflates the fitted range past the domain, so detrend first; then simple/ordinary/universal separate properly. Block support cuts the variance 42%, and the block's own average semivariance says why |
| 04 | `04_categorical_and_cutoffs.py` | Estimate the probability, do not threshold the estimate: indicator kriging scores Brier 0.170 against 0.278 and lands within a few points of the true proportion above cutoff, where the shortcut over-flags the map. Facies go through categorical simulation, which reproduces proportions and returns a rock name everywhere |
| 05 | `05_multivariate.py` | A densely sampled proxy is data. Collocated cokriging turns 40 primary samples plus a known secondary into a 24% lower error, and the sweep shows the return growing with the correlation rather than crossing a threshold: the estimator believes the number you hand it, so measure it |
| 06 | `06_implicit_modelling.py` | Geometry from contacts and dips, and the block model is differentiable with respect to them. The sigmoid temperature is not cosmetic: at 1 the adjoint matches a central difference to 1%, at 5 they differ by a factor of two, at 50 they differ in sign. Crisp pictures and usable gradients are different settings |
| 07 | `07_case_study.py` | The whole sequence on one table: decluster, transform, fit, simulate, post-process: ending in a grade-tonnage curve computed per realisation, because reading one off the averaged map understates the area above cutoff by 2.6 points |

The order is the reading order: the distinction everything rests on (01),
the input everything inherits (02), estimation (03), the two questions
that are not about a value (04), using more than one variable (05),
building the structure rather than the property (06), and all of it at
once (07).

Every script is seeded, runs on CPU, and writes its figure to `out/`.
Nothing is read from disk; script 07's table comes from
`geobrain.datasets`, which generates it deterministically. Run any of
them from the repository root:

```bash
python examples/02_geomodel/01_estimation_vs_simulation.py
```

The slowest are `01`, `04` and `07` at two to four minutes, all of it
sequential simulation; the rest finish inside one.

## A note on `geobrain.datasets`

The loaders (`walker_lake`, `jura`, `meuse`, `mining_3d`) are
**synthetic approximations**. Each builds a table whose columns, count
and general character mirror a published benchmark, from seeded random
processes rather than digitised source data. They are deterministic and
they exercise a workflow fairly. They are not the published data, and
script 07 says so where a reader would otherwise assume otherwise,
including where the stand-in fails to reproduce the benchmark's
character, which for Walker Lake is its skew.
