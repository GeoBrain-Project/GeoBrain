# Reservoir flow (`geobrain.physics.flow`)

Differentiable two-phase flow with wells: the forward model behind history
matching, well placement and closed-loop reservoir management.

| Piece | What is there |
|---|---|
| Grid and rock | `CartGrid`, `Rock` |
| Operators | `TransientFlowOperator`, `FlowEvolutionOperator`, `ParametricFlowOperator` |
| Wells | `Well`, `WellGroup`, `Perforation`, `BHPControl`, `RateControl`, `WellObservationOperator` |
| Time stepping | `TimeStepScheduler`, `AdaptiveTimeStepper` |
| Configuration | `FlowExecutionConfig`, `FlowHistoryConfig`, `FlowModelSchema`, `FlowFieldSpec` |
| Diagnostics | `FlowConvergenceDiagnostics`, `discover_flow_capabilities` |

The fluid and rock property models live one level down:

`geobrain.physics.flow.models.OilWaterModel`,
`geobrain.physics.flow.properties.relperm.RelPermCorey`,
`...properties.pvt.PVTAnalytic`, `...properties.fluid.OilWaterFluid`, and
`...wells.compute_well_index` gives the Peaceman well index from the cell
geometry.

## Why differentiable matters here

The time march is an **implicit Newton solve at every step**. Taping it would
store every Newton iteration of every timestep; instead the operator carries an
implicit-function adjoint, so one backward pass returns

$$\frac{\partial\,\text{cumulative oil}}{\partial \log k}$$

for **every cell in the model**, at the price of one extra simulation rather
than one per cell.

Read that map as economics, not as physics. Permeability is worth having near
the injector, where it buys injectivity, and worth *not* having along a streak
that runs straight to one producer, because that spends the water on one well.
It is the first step of a history match or a well-placement study.

## The working rule it demonstrates

Sweep is decided by the permeability field, not by the well pattern. Four
producers arranged symmetrically around one injector produce four different
histories, because the high-permeability streaks between them are not
symmetric.

```{note}
Well rates follow the canonical source convention: injection positive,
production negative. Displays negate production, and the axes say so.
```

```{figure} /_figures/07_reservoir_flow.png
:alt: A five-spot waterflood and its adjoint sensitivity

A five-spot waterflood, the water cut it produces well by well, and the
adjoint sensitivity that prices every cell. From
`examples/03_physics/07_reservoir_flow.py`.
```

## See also

- `examples/03_physics/07_reservoir_flow.py`: a five-spot waterflood, the
  water cut per well, and the adjoint that prices every cell.
