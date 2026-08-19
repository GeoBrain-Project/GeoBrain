# Geological modeling (`geobrain.geomodel`)

Everything that builds an earth before any physics touches it: geostatistics,
implicit geological modelling, and the frames that carry the result.

## The distinction the module is built around

Kriging gives you the **best single map**: minimum error variance, and
therefore smoother than the earth. Simulation gives you maps with the **right
variability**, each one wrong in a different way. Which you want depends on the
question: an average grade wants the estimate; a probability of exceeding a
cutoff wants the ensemble, because you cannot read a probability off a smooth
map.

## Geostatistics (`geobrain.geomodel.geostats`)

| Category | What is there |
|---|---|
| Kriging | `SimpleKriging`, `OrdinaryKriging`, `UniversalKriging`, `BlockKriging`, `IndicatorKriging`, `SoftIndicatorKriging`, `CollocatedCokriging` |
| Simulation | `SGSIM`, `DSSIM`, `SISIM`, `CoSGSIM`, `LUSIM`, `FFTMA`, `PlurigaussianSim` |
| Multiple-point | `SNESIM`, `FILTERSIM`, `DirectSampling`, `ImageQuilting` |
| Generative | `VAESimulator`, `GANSimulator`, `DiffusionSimulator` |
| Variograms | `VariogramCalculator`, `CrossVariogramCalculator`, `CovarianceModel`, `VariogramKernel`, automatic fitting with diagnostics |
| Transforms | `NormalScore`, `BoxCox`, `YeoJohnson`, `Detrend`, `Decluster`, `IndicatorEncode` |
| Validation | `KFold`, `LeaveOneOut`, `PostSimulation`, `PostIndicatorKriging` |

A field, start to finish:

```python
import numpy as np
from geobrain.geomodel import (FFTMA, CovarianceModel, GeoGrid,
                               PropertyMetadata, SimulationExecutionConfig,
                               VariogramKernel)

grid = GeoGrid(shape=(60, 40), origin=(0.0, 0.0), spacing=(10.0, 10.0))
model = CovarianceModel(nugget=0.0, structures=[
    VariogramKernel(kind=VariogramKernel.SPHERICAL, contribution=1.0,
                    ranges=(150.0, 60.0, 1.0e6), angles=(0.0, 0.0, 0.0))])

sim = FFTMA(model,
            property=PropertyMetadata(name="V", kind="continuous", unit="1"),
            execution=SimulationExecutionConfig(n_realizations=1, seed=7))
frame = sim(None, grid).realizations[0].frame
values = np.asarray(frame.to_numpy("simulation"), dtype=float).reshape(-1)
print(values.shape, f"mean {values.mean():+.3f}, sd {values.std():.3f}")
```

```text
(2400,) mean -0.115, sd 1.078
```

```{warning}
The variogram `ranges` bind to the grid axes **positionally**, and a flat field
unflattens with x varying fastest, so `values.reshape(NY, NX)` is `[y, x]`,
ready for `imshow`. Reshaping the other way round and transposing gives an
earth that still looks plausible and has its anisotropy mirrored. If your
sample locations stop sitting on the features they were drawn from, this is
why.
```

## Frames and models

| Object | Carries |
|---|---|
| `GeoGrid` | a regular grid domain: shape, origin, spacing |
| `GeoPoints` | scattered locations |
| `GeoFrame` | geometry plus named properties; what a kriging or simulation call takes and returns |
| `PropertyMetadata` | a property's name, kind (continuous / categorical) and unit |
| `EarthModel` / `Field` | several coupled properties on one mesh, which a joint inversion updates together |
| `Category` | a named class with a code, because a facies is a name, not a number |

## Implicit modelling (`geobrain.geomodel.implicit`)

`ImplicitModel` builds geometry from what a geologist actually records,
namely surface points, orientations, series and faults, as the level sets of a scalar
field. It is **differentiable with respect to the input data**, so the
derivative of a unit's volume with respect to a contact point's position is
one backward pass, and a block model can be an inversion parameter rather than
a fixed input.

`SurfacePointData`, `OrientationData`, `SeriesDefinition`, `FaultDefinition`
and `ImplicitModelConfig` are the inputs; `StackRelation` says how series stack.

```{figure} /_figures/07_case_study.png
:class: full-width
:alt: A geostatistical study from samples to a probability

A study end to end, from a sample table to the probability a decision needs.
From `examples/02_geomodel/07_case_study.py`.
```

## See also

- `examples/02_geomodel/01_estimation_vs_simulation.py`: why the two answer
  different questions, measured on the same data.
- `examples/02_geomodel/03_kriging_estimators.py`: what each estimator assumes
  about the mean.
- `examples/02_geomodel/06_implicit_modelling.py`: geometry that carries
  gradients.
- `examples/02_geomodel/07_case_study.py`: decluster, transform, fit,
  simulate, and finish with a probability rather than a map.
