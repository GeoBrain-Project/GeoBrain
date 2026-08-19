# Inversion (`geobrain.inverse`, `geobrain.optim`)

`geobrain.inverse` states the problem; `geobrain.optim` searches it.

## Stating the problem

```python
# sketch: runnable version in the quick start
InverseProblem(forward=..., observed=..., likelihood=..., prior=...)
```

| Piece | Options |
|---|---|
| Likelihood | `GaussianLikelihood`, `StudentTLikelihood` via `ChannelLikelihood` for multi-channel data |
| Prior | `GaussianPrior`, or any object with `log_prob` and `sample` |
| Waveform misfits | `L2Waveform`, `L1Waveform`, `HuberWaveform`, `EnvelopeWaveform`, `GlobalCorrelationWaveform`, `NormalizedIntegrationWaveform`, `TravelTimeWaveform`, `WassersteinWaveform` |
| Filtering | `FrequencyFilteredMisfit`, `butterworth_lowpass` |
| Joint | `JointProblem`, `JointForward`, `JointModelSpace` |

The waveform misfits are not interchangeable. Least squares is the one that
falls into a local minimum when the starting model is a cycle out; envelope,
travel-time and Wasserstein misfits are the ones that do not, and a multi-scale
schedule usually starts with one of those.

## Searching it

`Inverter` is the loop; `AdamConfig` and `LBFGSConfig` choose the optimizer.

| Piece | Options |
|---|---|
| Regularizers | `smallness`, `smoothness`, `smoothness_second_order`, `total_variation`, `total_variation_second_order`, `tikhonov`, `l1`, `l2` |
| Structural coupling | `cross_gradient`, `gmm_prior` |
| Weighting | `depth_weighting`, `DepthWeight`, `Weight`, `Mask`, `Freeze` |
| Gradient processing | `GaussianSmooth`, `NormClip`, `NaNGuard`, `BoundsClamp`, `StepProjection` |
| Bookkeeping | `InversionResult`, `IterationRecord`, `StopReason`, `CancellationToken`, `log_every` |

```{admonition} A regularizer is a prior seen from the other side
:class: tip

A smoothness term in the objective *is* a Gaussian prior on the model's
roughness. Writing it as a `prior=` on the problem rather than as a term in the
loss means the Bayesian door sees it too, automatically, and without a second
copy that can drift out of step.
```

## Where your geology goes

Regularization is not a numerical detail; it is where you say what you believe
about the earth before the data arrive. Smoothness says gradational contacts.
Total variation says blocky units with sharp boundaries. `cross_gradient` says
two properties change *where* together but not *how much* together, which is
what you want when the physics are independent but the geology is not.
`gmm_prior` says the cells belong to a known set of rock types, which is the
strongest statement of the four and the one that turns two property images into
a rock-unit map.

```{figure} /_figures/05_inversion_toolbox.png
:class: full-width
:alt: Regularizers, bounds and an IRLS pass compared

The knobs this page describes, and what each one buys: regularizers, bounds
and an IRLS sparsity pass. From
`examples/01_architecture/05_inversion_toolbox.py`.
```

## Stopping

Data carry noise. Driving the misfit to zero fits the noise and grows structure
to do it, so the honest stopping rule is chi-squared, the misfit measured in
units of the noise, reaching 1. The physics examples stop there and show what
carrying on looks like.

## See also

- `examples/01_architecture/05_inversion_toolbox.py`: regularizers, bounds and
  IRLS, compared on one problem.
- `examples/03_physics/02_dc_resistivity.py`: the stopping rule, and what
  happens past it.
- `examples/03_physics/04_petrophysical_joint_inversion.py`: `gmm_prior` as
  the coupling that gives a rock-unit map.
