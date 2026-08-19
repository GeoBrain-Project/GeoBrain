# Architecture

Four contracts hold GeoBrain together. Everything else, from the physics
families to the samplers to the neural parameterizations, is something you can add without
touching them.

## An operator is a contract, not a function

Every forward model is an `Operator`. The ones that produce data are
`ForwardOperator`s; the ones that turn one property into another are
`PropertyTransform`s. Both declare a `DifferentiabilitySpec`:

```python
from geobrain.physics.rock.models import GardnerOperator

spec = GardnerOperator().differentiability
print(spec.trainable_inputs, "->", spec.output_keys, "via", spec.level.value)
```

```text
('vp',) -> ('rho',) via full_autograd
```

Three things are being promised there: what you may ask for a gradient with
respect to, what comes out, and *how* the gradient is obtained. That last one
is the interesting one, because it is not always autograd:

| level | what it means | typical case |
|---|---|---|
| `FULL_AUTOGRAD` | the whole forward pass is a differentiable graph | rock physics, potential fields on a grid |
| `IMPLICIT_VJP` | the forward pass solves an equation; the gradient comes from an implicit-function adjoint | Helmholtz, DC resistivity, reservoir flow |
| `CUSTOM_VJP` | a hand-written backward | kernels where autograd would build an unaffordable graph |

The declaration is not a comment. `examples/00_showcase/05_differentiability_modes.py`
checks all three against finite differences and prints the agreement, and the
same check is what a new operator has to pass.

```{admonition} Why a level at all
:class: note

Because the cost model differs. An implicit adjoint gets you the gradient for
the price of one extra solve, no matter how many parameters there are; a naive
autograd tape through the same solver would store every Newton iteration. When
a chain reports `implicit_vjp`, that is telling you which of those you are
paying for.
```

## Physics composes with `@`

```python
from geobrain.physics.potential import Gravity2D, PotentialSurvey2D
from geobrain.physics.rock.models import GardnerOperator
import torch

stations = torch.stack([torch.linspace(0.0, 600.0, 25, dtype=torch.float64),
                        torch.ones(25, dtype=torch.float64)], dim=1)
chain = Gravity2D(PotentialSurvey2D(stations)) @ GardnerOperator()
print(chain.differentiability.trainable_inputs,
      chain.differentiability.output_keys,
      chain.differentiability.level.value)
```

```text
('vp',) ('gz',) custom_vjp
```

The chain's contract was **derived**, by three rules:

1. Trainable inputs come from the **entry** link, the rightmost, which runs
   first. You give the chain `vp`, not `rho`, even though the gravity kernel
   asks for `rho`: that is produced upstream.
2. Output keys come from the **terminal** link.
3. The differentiability level is the **weakest** of the members. One
   implicit-VJP link makes the whole chain implicit-VJP, because that is what
   the reader is actually paying.

Parallel composition is an `OperatorBundle`: named channels that thread one
`ModelState` through every member and merge into one `ForwardOutput`. Its
trainable inputs are the union; its output keys are the channel names. Chains
nest inside bundles, so a joint problem is a bundle of chains, and one
`backward()` accumulates every channel's sensitivity into the same `.grad`.

Compositions that cannot be honoured are refused when you build them. Only the
last link may be a `ForwardOperator`; put one mid-chain and you get a
structured error naming the offending slot, not a wrong answer at run time.
`examples/01_architecture/03_composition_rules.py` is the rulebook, demonstrated.

```{figure} /_figures/02_operator_composition.png
:alt: Serial and parallel operator composition

Serial composition with `@` and parallel channels through `OperatorBundle`,
both differentiated by a single backward pass. From
`examples/00_showcase/02_operator_composition.py`.
```

## A mesh declares capabilities; physics declares requirements

A mesh is not an array shape. `TensorMesh` (uniform or graded), `OctreeMesh`,
`UnstructuredMesh` and `CylindricalMesh` each satisfy a set of named
protocols (`UniformMesh`, `StructuredMesh`, `GeometryMesh`,
`PrismGeometryMesh`, `ConnectivityMesh`), and each physics declares which of
them it needs.

So "this kernel cannot run on that mesh" surfaces as a type error rather than
as a plausible wrong answer. A finite-difference wave equation needs
structure; a Talwani gravity kernel needs only cell geometry, so it runs
straight on triangles.

When one problem needs both, a differentiable `MeshProjection` bridges them,
and being differentiable is the whole point: the gradient comes back through
the projection, so a joint inversion can run the wave equation on a structured
grid and gravity on the unstructured model mesh and still return one gradient
from one backward pass. That is
`examples/00_showcase/03_mesh_projection_joint_inversion.py`.

```{figure} /_figures/03_mesh_projection_joint_inversion.png
:class: gb-tall
:alt: Joint inversion across two meshes

Joint inversion across two discretizations, bridged by a differentiable
projection so one backward pass reaches both. From
`examples/00_showcase/03_mesh_projection_joint_inversion.py`.
```

## One problem, two doors

```python
# sketch: runnable version in the quick start
problem = InverseProblem(forward=..., observed=..., likelihood=..., prior=...)

result    = problem.create_inverter(params=..., optimizer=...).run(n_iters=60)
posterior = problem.as_posterior(transforms=...)
draws     = posterior.sample("nuts", params=result.best_params, n_iters=300)
```

The deterministic door descends `-log posterior`; the Bayesian door explores
the same function through the same object. `"hmc"`, `"langevin"` and `"svgd"`
are the same call with a different string.

This buys two things:

- **A regularizer and a prior are the same thing seen twice.** A smoothness
  term in the objective is a Gaussian prior on the model's roughness; writing
  it as a prior means the sampler sees it too, automatically.
- **Bounds become transforms.** `PositiveTransform`, `IntervalTransform` and
  friends are changes of variable, so the sampler works in an unbounded space
  while the model stays inside its physical range by construction, rather than
  being clipped, which would break detailed balance.

`examples/00_showcase/04_deterministic_bayes_unified.py` runs both doors on one
rock-physics AVO problem and checks that the inverter's loss at the MAP matches
`-problem.log_posterior` to the digit.

## The state objects

| object | carries | why separate |
|---|---|---|
| `ModelState` | the tensors the physics reads, such as `vp`, `rho`, `sigma` … | the unknown, and what derives from it |
| `ForwardContext` | what is not a model: the mesh, the device, solver options | so the same operator runs on another mesh without being rebuilt |
| `ForwardOutput` | the data, plus any diagnostic fields | data are what you invert; fields are what you look at |

`examples/01_architecture/01_operator_contract.py` opens all three.

## Where next

- [Examples gallery](examples.md): the architecture, demonstrated and measured.
- [Quick start](quickstart.md): the shortest path through all four contracts.
