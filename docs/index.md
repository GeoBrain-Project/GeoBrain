# GeoBrain

**GeoBrain** is an open, modular platform for **Geo**scientific **B**ayesian
**R**easoning with **A**rtificial **In**telligence, built for integrated
subsurface modeling.

It combines differentiable physics, Bayesian inference and deep learning so
that a whole workflow, from a geostatistical earth model through rock
physics to a geophysical forward model and back out as an inversion, is one
computational graph. Every forward model is a composable operator, every
operator declares how it differentiates, and the same problem object serves a
deterministic optimiser and a posterior sampler alike.

```{admonition} Where to start
:class: tip

New here? [Installation](installation.md) then [Quick start](quickstart.md)
is ten minutes. If you would rather see it work first, every figure in the
[examples gallery](examples.md) is the unedited output of a script you can run.
```

```{figure} /_figures/01_gravity_inversion.png
:alt: Gravity inversion: truth, data and recovery

The whole platform on one screen: a density model, the gravity it produces,
and the model recovered back out of that data. From
`examples/00_showcase/01_gravity_inversion.py`.
```

## The four ideas

Everything else in GeoBrain, from the physics families to the samplers to
the neural parameterizations, is something you can add without touching these.

:::{card} An operator is a contract, not a function
Every forward model is a `ForwardOperator` that declares a
`DifferentiabilitySpec`: its trainable inputs, its outputs, and *how* its
gradient is obtained: full autograd, an implicit adjoint, or a hand-written
VJP. The declaration is checked against finite differences, so "this is
differentiable" is something the platform can be held to rather than a promise
in a docstring.
:::

:::{card} Physics composes with `@`
Chain operators in series, run them in parallel as an `OperatorBundle`, and the
composition's contract is *derived* from its links: trainable inputs from the
entry link, outputs from the terminal link, differentiability level the weakest
of the members. A chain that cannot be honoured is refused when you build it,
not when you run it.
:::

:::{card} A mesh declares capabilities; physics declares requirements
`TensorMesh`, `OctreeMesh` and `UnstructuredMesh` each
satisfy named protocols, and each physics says which it needs, so "this kernel
cannot run on that mesh" is a type error rather than a wrong answer. A
differentiable `MeshProjection` bridges them, which is what lets one joint
inversion run the wave equation on a structured grid and gravity straight on
triangles.
:::

:::{card} One problem, two doors
The same `InverseProblem` serves a deterministic optimiser
(`create_inverter().run()`) and a posterior sampler
(`as_posterior().sample("nuts")`). Uncertainty quantification is a method call
on the object you already built, not a second pipeline to keep in step with the
first.
:::

[Read the architecture in full](architecture.md), or see all four at once in
`examples/00_showcase/`.

## What is in the box

| | |
|---|---|
| **Geological modeling** | kriging family, sequential and indicator simulation, variograms with fitting diagnostics, differentiable implicit modelling |
| **Rock physics** | effective medium, granular media, fluid substitution, empirical relations, petrophysics |
| **Wave** | acoustic / elastic / visco time-domain, Helmholtz with an implicit adjoint, AVO reflectivity, wavelets |
| **Electromagnetics** | DC, IP, SIP, MT, FDEM, TEM, CSEM, airborne, SP |
| **Potential fields** | gravity 2-D/3-D, magnetics with remanence, Euler deconvolution |
| **Flow** | differentiable two-phase reservoir simulation with wells |
| **Inversion** | regularizers, bounds, IRLS, gradient processors, Adam and L-BFGS |
| **Bayesian** | HMC, NUTS, Langevin, SVGD behind one `Posterior`, with transforms and diagnostics |
| **Neural networks** | Deep Image Prior and latent-space reparameterizations |
| **Decision** | efficacy of information, scored cell by cell from a prior ensemble |

```{toctree}
:hidden:
:caption: Getting started

installation
quickstart
architecture
```

```{toctree}
:hidden:
:caption: Modules

modules/mesh
modules/geomodel
modules/rock_physics
modules/wave
modules/em
modules/potential
modules/flow
modules/inversion
modules/bayesian
modules/neural
modules/io_vis
modules/decision
```

```{toctree}
:hidden:
:caption: Reference

examples
```
