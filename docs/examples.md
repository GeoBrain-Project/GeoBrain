# Examples gallery

Twenty-nine scripts in five parts, meant to be read in order. Each is seeded,
runs on CPU, prints its measurements as it goes, and writes one figure to its
part's `out/`. Three of them also write a GIF, where time is the thing being
shown.

They are the tutorial layer of this documentation. Rather than repeat them
here in prose, this page says what each part is for and what each script
proves, so you can pick an entry point.

```bash
python examples/00_showcase/01_gravity_inversion.py       # see it work
python examples/01_architecture/01_operator_contract.py   # learn the API
python examples/03_physics/01_seismic_fwi.py              # put it to work
```

```{figure} /_figures/01_seismic_fwi.png
:alt: Multi-scale full-waveform inversion on Marmousi II

Multi-scale full-waveform inversion on Marmousi II, the longest workflow in
the gallery. From `examples/03_physics/01_seismic_fwi.py`.
```

## Part 0: see it work

Six self-contained scripts, each proving one architectural claim with a
running figure.

| # | Script | What it proves |
|---|---|---|
| 01 | `01_gravity_inversion.py` | The whole platform in three objects: physics, problem, solver |
| 02 | `02_operator_composition.py` | Serial chains with `@`, parallel channels with `OperatorBundle`, one backward |
| 03 | `03_mesh_projection_joint_inversion.py` | Joint inversion across an unstructured and a structured mesh |
| 04 | `04_deterministic_bayes_unified.py` | Deterministic and Bayesian from one problem, one keyword apart |
| 05 | `05_differentiability_modes.py` | Three gradient mechanisms, all checked against finite differences |
| 06 | `06_neural_network_integration.py` | The unknown as an image, a decoder's weights, or its latent code |

## Part 1: understand it

Eight scripts that open one layer each, in the order you meet them.

| # | Script | What it opens |
|---|---|---|
| 01 | `01_operator_contract.py` | `ModelState` in, `ForwardOutput` out, and the contract between |
| 02 | `02_mesh_taxonomy.py` | Four meshes, one domain, and the capabilities that decide what may run |
| 03 | `03_composition_rules.py` | How a chain's contract is derived from its links |
| 04 | `04_differentiability_levels.py` | The ladder from full autograd to a declared custom VJP |
| 05 | `05_inversion_toolbox.py` | Regularizers, bounds and IRLS: where your geology goes |
| 06 | `06_bayesian_workflow.py` | Chains, R-hat, ESS, and the posterior predictive check |
| 07 | `07_custom_operator.py` | Two hooks and two declarations buy you the rest of the platform |
| 08 | `08_data_io_and_figures.py` | SEG-Y, VTK, HDF5 in; `geobrain.vis` out |

## Part 2: build the earth

Geostatistics and implicit geological modelling.

| # | Script | The step |
|---|---|---|
| 01 | `01_estimation_vs_simulation.py` | Kriging for the best map, simulation for the right variability |
| 02 | `02_variogram.py` | The input everything downstream inherits, and its own uncertainty |
| 03 | `03_kriging_estimators.py` | Simple, ordinary, universal: what each assumes about the mean |
| 04 | `04_categorical_and_cutoffs.py` | Estimate the probability; do not threshold the estimate |
| 05 | `05_multivariate.py` | A densely sampled proxy is data: collocated cokriging |
| 06 | `06_implicit_modelling.py` | Geometry from contacts and dips, differentiable with respect to them |
| 07 | `07_case_study.py` | The whole sequence, ending in a probability rather than a map |

## Part 3: tour the physics

Seven full workflows on real earth models: the Marmousi II benchmark for
seismic, correlated geostatistical fields elsewhere, and inversions that stop
on chi-squared rather than on an iteration count.

| # | Script | The family |
|---|---|---|
| 01 | `01_seismic_fwi.py` | Multi-scale full-waveform inversion on Marmousi II |
| 02 | `02_dc_resistivity.py` | Topography as part of the model; chi-squared as the stopping rule |
| 03 | `03_induced_polarization.py` | Chargeability as an image DC is blind to |
| 04 | `04_petrophysical_joint_inversion.py` | Gravity and magnetics answering *which rock is where* |
| 05 | `05_em_induction.py` | An airborne sounding inverted for a layered earth |
| 06 | `06_potential_fields.py` | Gravity answers once; magnetics differently at every latitude |
| 07 | `07_reservoir_flow.py` | A five-spot, and one adjoint that prices every cell |

## Part 4: decide what to measure next

An inversion says what is down there; a survey plan has to say which
measurement would change the decision, before it is paid for.

| # | Script | The question |
|---|---|---|
| 01 | `01_borehole_efficacy_of_information.py` | Where should the next hole go, and how deep: costed before drilling |

## How the figures work

Twenty-nine scripts could easily be twenty-nine styles. They are not: a shared
toolkit, `_style.py`, sits in each part's directory and every figure is built
through it. The rules it enforces are written down in
[`examples/README.md`](https://github.com/GeoBrain-Project/GeoBrain/blob/main/examples/README.md),
and the short version is:

- **Colour has roles, not ramps.** One ramp for a value, one for a signed
  quantity, one for a magnitude, discrete colours for classes, plus three
  domain ramps where the convention is older than the argument: seismic
  amplitude, velocity, resistivity.
- **One colour bar per scale, not per panel**, because panels drawn on
  different limits cannot be compared by eye however firmly the caption asks.
- **A panel has to earn its place.** Every figure is two to six panels.
- **A title says what a panel is and what it measured**, never how to read
  it. If a panel needs that, the panel is wrong.
- **Animations only where time is the subject.**
