# Part 0: the platform at a glance

Six scripts, six architectural claims, each proven by a running figure.
Start here; read `01_architecture/` when you want to know *why* they work.

| # | Script | The claim it proves |
|---|---|---|
| 01 | `01_gravity_inversion.py` | The whole platform is three objects: a differentiable forward, an `InverseProblem`, an `Inverter`: shown on a depth-weighted, IRLS-compact 2-D gravity inversion |
| 02 | `02_operator_composition.py` | Physics composes two ways and only two ways: SERIAL with `@` (`Gravity2D @ GardnerOperator`, derived contract) and PARALLEL with `OperatorBundle` (named channels over one shared `ModelState`): the chain nests inside the bundle, and one `backward()` lands both channels in the same `vp.grad` |
| 03 | `03_mesh_projection_joint_inversion.py` | Differentiable `MeshProjection` is what makes joint inversion work across mesh kinds: the model lives on a triangular `UnstructuredMesh` refined along a dipping target, the wave equation gets its structured 25 m grid through the projection, gravity runs STRAIGHT on the triangles, and gradients from both meet on the same triangles |
| 04 | `04_deterministic_bayes_unified.py` | Deterministic and Bayesian share ONE API: porosity drives five rock-physics laws into a nine-angle AVO gather; `problem.create_inverter().run()` gives the MAP, `problem.as_posterior(transforms=...).sample("nuts")` gives the bounded posterior: a one-line pivot, the Inverter's loss IS `-log posterior` to the digit |

| 05 | `05_differentiability_modes.py` | Gradients come from several mechanisms and every operator DECLARES which: `Acoustic2D` unrolls 500 timesteps (`FULL_AUTOGRAD`), `Helmholtz2D` (the same wave equation) replaces the whole solve with one adjoint (`IMPLICIT_VJP`), `Gravity2D` ships a hand-written backward (`CUSTOM_VJP`). All three match central finite differences, and where they differ it is the finite difference that is wrong, proven by its O(h²) convergence |

| 06 | `06_neural_network_integration.py` | Parameterization is an operator, so "what are the unknowns?" is a one-line choice: the porosity image itself, a `ConvDecoder2d`'s WEIGHTS (`seismic @ WeightReparameterization(...)`: the deep image prior), or its CODE (`seismic @ LatentReparameterization(...)`). All three run through the same `InverseProblem.create_inverter().run()` on an unchanged rock-physics-to-AVO chain over an FFT-MA co-simulated section, and the network parameterizations recover porosity 46% and 33% better than the explicit one, an ordering that follows neither the unknown count nor the data misfit, because the prior is not a term in the objective but the set of images the decoder can draw |

Every script generates its own data, runs on CPU in minutes, and saves its
figure to `out/`.
