# Part 3: tour the physics families

Part 1 teaches the machine and Part 2 builds the earth. Part 3 puts
physics on it: seven scripts in family order, each a complete workflow on
a real earth model: build the model, design the survey, run the forward,
show what a specialist would ask to see, then invert it or price it with
the adjoint.

Three things every script inherits.

**Models come from `_models.py`** where they are shared: a window of the
**Marmousi II** benchmark, read from `../data/marmousi` and resampled onto
the working grid, or a correlated geostatistical field drawn with the
platform's own FFT-MA simulator. Nothing in this section is a rectangle
for its own sake, and where a script builds its own bodies it is because
the point being made needs a known answer.

**Figures come from the gallery's shared toolkit** in `_style.py`:
four colour roles rather than a ramp per script, one colour bar per
SCALE rather than per panel. `geobrain.vis` ships
domain ramps and one-call plotters too; they are demonstrated where that
module is the subject, in `01_architecture/08_data_io_and_figures.py`,
and the rest of the gallery stays on the roles so that a reader who has
learned one figure has learned them all.

**Inversions stop on chi-squared, and say what stopping cost them.**
Fitting noise is not convergence. Scripts 02 and 03 show the rule working;
04 makes the stop load-bearing, because a smooth stage run *past* the
noise leaves its petrophysical coupling nothing to do; 05 reports that it
settles just above 1 and that the base of its conductor is the price.

| # | Script | The family, and the one thing it teaches |
|---|---|---|
| 01 | `01_seismic_fwi.py` | Multi-scale FWI on Marmousi: L-BFGS closes three times the gap Adam does, and a start smoothed less than the half-wavelength would just be the starting model talking |
| 02 | `02_dc_resistivity.py` | A survey over a ridge: topography is part of the model, the air must be held fixed, and chi-squared says when to stop |
| 03 | `03_induced_polarization.py` | Chargeability as an image independent of resistivity: a body DC is blind to by construction |
| 04 | `04_petrophysical_joint_inversion.py` | Gravity and magnetics answering *which rock is where*: a three-unit Gaussian mixture as the coupling term snaps the recovery onto known petrophysical clusters: 25% → 75% of anomalous cells inside their own unit's 95% ellipse, at held data fit, and hands back a rock-unit map |
| 05 | `05_em_induction.py` | An airborne frequency-domain sounding inverted for a layered earth: twenty numbers, eleven layers, and the discipline of quoting a conductance and a depth range instead of a boundary |
| 06 | `06_potential_fields.py` | Gravity answers once; magnetics answers differently at every latitude and for every remanence: the case that breaks susceptibility inversion |
| 07 | `07_reservoir_flow.py` | A five-spot on a geostatistical permeability field: four identical wells, four different histories, and one adjoint that prices every cell |

The order is the reading order: the wave equation first (01), then the
electrical methods (02–03), the potential fields both jointly and alone
(04, 06) with electromagnetic induction between them (05), and reservoir
flow last (07).

Every script is seeded, runs on CPU, and writes its figure to `out/`.
Only 01 needs anything the repository does not already hold: the Marmousi
sections, fetched and verified by `python examples/data/fetch_marmousi.py`.
Run any of them from the repository root:

```bash
python examples/03_physics/01_seismic_fwi.py
```

The slowest are `02`, `05` and `07` at one to three minutes; the rest
finish well inside one. In `05` almost all of that is the 1-D Hankel
transform itself, at roughly 3 s per forward-and-adjoint evaluation.
