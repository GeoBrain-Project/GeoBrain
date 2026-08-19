# Part 1: understand the machine

Part 0 shows that the platform works. Part 1 is how you learn to write
your own code with it: eight scripts, each opening one layer, in the
order you meet them.

| # | Script | What you learn |
|---|---|---|
| 01 | `01_operator_contract.py` | The two arrows, `ModelState` / `ForwardContext` / `ForwardOutput`, the `DifferentiabilitySpec`, and how to read a structured `GeoBrainError` |
| 02 | `02_mesh_taxonomy.py` | Uniform, graded, octree and unstructured meshes; the capability matrix that decides which physics may run on which; `MeshProjection` and what each discretization costs |
| 03 | `03_composition_rules.py` | The rules chains and bundles derive: entry inputs, terminal outputs, weakest level, and the compositions the platform refuses |
| 04 | `04_differentiability_levels.py` | The five levels; the implicit-VJP seam that turns a solver's backward pass into one adjoint solve; `gradient_check` as an audit tool |
| 05 | `05_inversion_toolbox.py` | Likelihood, prior, the regularizer library, bounds and optimizer choice, one problem inverted four ways at matched data fit |
| 06 | `06_bayesian_workflow.py` | Multiple chains, split R-hat, effective sample size, traces and posterior predictive checks: how to decide whether to believe a posterior |
| 07 | `07_custom_operator.py` | Writing both extension points, declaring capabilities, raising your own structured errors, and dropping the result into the shipped stack |
| 08 | `08_data_io_and_figures.py` | SEG-Y / VTK / HDF5 round-trips and the `geobrain.vis` helpers: how data gets in and results get out |

Every script is self-contained, seeded, CPU-friendly, and writes its
figure to `out/`. Run any of them from the repository root:

```bash
python examples/01_architecture/05_inversion_toolbox.py
```
