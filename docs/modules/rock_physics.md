# Rock physics (`geobrain.physics.rock`)

The layer between geology and geophysics: what a rock made of these minerals,
with this porosity, saturated with this fluid, does to a wave or a field.

Every model here is differentiable, and the composable ones are
`PropertyTransform`s, so a porosity parameter can drive five rock-physics laws
and a wave equation, and the gradient returns through all of them without a
hand-written derivative anywhere.

| Category | Models |
|---|---|
| Effective medium | Voigt-Reuss-Hill, Hashin-Shtrikman, `SelfConsistent`, `Hudson`, `SayersKachanov` |
| Granular media | `HertzMindlin`, soft sand, stiff sand, contact cement |
| Fluid substitution | `Gassmann`, `WoodFluidMix`, `BatzleWangBrine`, `BiotHighFrequency` |
| Empirical | `Gardner`, `RockPhysicsTemplate` |
| Petrophysics | `KozenyCarman` permeability, `ArchieResistivity` |
| Conversions | `VelocitiesFromModuli`, `ModuliFromVelocities` |

## Two ways in

The **functions** are for a one-off calculation:

```python
import torch
from geobrain.physics.rock import velocities_from_moduli

k = torch.tensor(2.0e10, dtype=torch.float64)     # bulk modulus [Pa]
mu = torch.tensor(1.0e10, dtype=torch.float64)    # shear modulus [Pa]
rho = torch.tensor(2300.0, dtype=torch.float64)   # density [kg/m3]
vp, vs = velocities_from_moduli(k, mu, rho)
print(f"vp {float(vp):.0f} m/s, vs {float(vs):.0f} m/s")
```

```text
vp 3807 m/s, vs 2085 m/s
```

The **operators** are for putting in a chain, where they carry a contract:

```python
from geobrain.physics.rock.models import GardnerOperator

spec = GardnerOperator().differentiability
print(spec.trainable_inputs, "->", spec.output_keys, "via", spec.level.value)
```

```text
('vp',) -> ('rho',) via full_autograd
```

`ROCK_OPERATOR_TYPES` and `get_rock_operator` are the registry, for code that
has to pick a model by name.

```{figure} /_figures/04_petrophysical_joint_inversion.png
:class: full-width
:alt: Two images resolved into rock units

Where rock physics earns its place in an inversion: a petrophysical prior
turns two independent images into one set of rock units. From
`examples/03_physics/04_petrophysical_joint_inversion.py`.
```

## See also

- `examples/01_architecture/03_composition_rules.py`: Gardner and Castagna as
  chain links feeding Aki-Richards.
- `examples/00_showcase/04_deterministic_bayes_unified.py`: porosity driving
  five rock-physics laws into an AVO gather, inverted both ways.
- `examples/03_physics/04_petrophysical_joint_inversion.py`: petrophysics as a
  coupling term rather than a forward step.
