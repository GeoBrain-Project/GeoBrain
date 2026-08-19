# Quick start

One problem, solved twice, deterministically and then Bayesianly, with the
physics assembled from two operators. Every output below is what the script
actually prints.

## 1. A mesh, and a chain of operators on it

```python
import torch
from geobrain.core import ForwardContext, ModelState
from geobrain.mesh import TensorMesh
from geobrain.physics.rock.models import GardnerOperator
from geobrain.physics.potential import Gravity2D, PotentialSurvey2D

torch.manual_seed(0)                        # so the numbers below reproduce

mesh = TensorMesh(shape=(12, 24), spacing=(25.0, 25.0))
ctx = ForwardContext.of(mesh=mesh)
stations = torch.stack([torch.linspace(0.0, 600.0, 25, dtype=torch.float64),
                        torch.full((25,), 1.0, dtype=torch.float64)], dim=1)

# vp -> rho (Gardner) -> gz (Talwani). '@' is operator composition.
forward = Gravity2D(PotentialSurvey2D(stations)) @ GardnerOperator()
print(forward.differentiability)
```

```text
DifferentiabilitySpec(level=<DifferentiabilityLevel.CUSTOM_VJP: 'custom_vjp'>,
                      trainable_inputs=('vp',), output_keys=('gz',),
                      structural_params=(), input_units={})
```

Read that line, because it is the whole idea. You composed a rock-physics
transform with a gravity kernel; the **chain** worked out that what you can
train is `vp`, that what comes out is `gz`, and that the gradient will arrive
through a hand-written VJP, the weakest mechanism of the two links. Nothing
asserted it. Ask a chain that cannot be honoured and it refuses at build time.

```{note}
`dtype=torch.float64` is not decoration. The potential-field operators are
strict about it, and a float32 survey is rejected rather than quietly losing
precision in a kernel that sums over every cell.
```

## 2. Synthetic data from a known earth

```python
vp_true = 1800.0 + 40.0 * torch.arange(12, dtype=torch.float64)[:, None]
vp_true = vp_true.expand(12, 24).clone()
vp_true[4:8, 8:16] += 400.0                 # a fast, dense block

with torch.no_grad():
    observed = forward(ModelState({"vp": vp_true}), ctx).data["gz"]
observed = observed + 2.0e-8 * torch.randn_like(observed)
```

A `ModelState` is the bag of tensors the physics reads; a `ForwardContext`
carries what is not a model: here, the mesh. Keeping them apart is what lets
the same operator run on a different mesh without being rebuilt.

## 3. One problem object

```python
from geobrain.inverse import GaussianLikelihood, InverseProblem

problem = InverseProblem(forward=forward, observed={"gz": observed},
                         likelihood=GaussianLikelihood(std=2.0e-8))
```

This is the object both solvers below take. It knows the forward model, the
data, and how the data are distributed around it, and nothing about how you
intend to search.

## 4a. Deterministic: the best-fitting model

```python
from geobrain.optim import LBFGSConfig

start = {"vp": (1800.0 + 40.0 * torch.arange(12, dtype=torch.float64)[:, None]
                ).expand(12, 24).clone()}
result = problem.create_inverter(
    params=start, optimizer=LBFGSConfig(lr=1.0, max_iter=30),
    bounds={"vp": (1500.0, 4000.0)}, ctx=ctx).run(n_iters=10)
print(f"misfit {result.best_loss:.4g} after {result.completed_iters} iters")
```

```text
misfit 8.232e-09 after 10 iters
```

## 4b. Bayesian: the models still standing

```python
from geobrain.bayes import PositiveTransform

posterior = problem.as_posterior(transforms={"vp": PositiveTransform()})
draws = posterior.sample("nuts", params=result.best_params, n_iters=100,
                         ctx=ctx, warmup=50, max_depth=5)
print("posterior sd [m/s]:", draws.samples["vp"].std(0).mean().item())
```

```text
posterior sd [m/s]: 29.630669631493827
```

The pivot is `as_posterior()` on the problem you already built, followed by one
`sample()` call. `"hmc"`, `"langevin"` and `"svgd"` are the same call with a
different string. The `PositiveTransform` is a change of variable, so the
sampler explores an unbounded space while `vp` stays positive by construction.

The number is the point of doing it at all. The deterministic run returned one
velocity model; the posterior says the data leave about 30 m/s of room around
it, which is the difference between an answer and an answer you can act on.
Twenty-five gravity stations over a 12x24 grid is a badly underdetermined
problem, so read that spread as what this survey pins down, not as a general
result.

```{figure} /_figures/04_deterministic_bayes_unified.png
:alt: One problem solved deterministically and sampled

The same pivot this page walks through, run on an AVO problem: one problem
object solved deterministically, then sampled. From
`examples/00_showcase/04_deterministic_bayes_unified.py`.
```

## Where next

- [Architecture](architecture.md): the contracts underneath all of this.
- [Examples gallery](examples.md): twenty-nine runnable scripts, from a
  three-object minimum to full multiphysics inversions.
