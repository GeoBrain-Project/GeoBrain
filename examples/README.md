# GeoBrain examples

Five parts, in reading order. Pick your entry point by what you want
first:

- **`00_showcase/`**: see it work. Six self-contained scripts that each
  prove one architectural claim with a running figure: the three-object
  platform pattern, serial and parallel operator composition, joint
  inversion bridged by a differentiable mesh projection, the single API
  that serves both deterministic and Bayesian inversion, the several
  gradient mechanisms behind one declared contract, and a neural network
  dropped into the middle of a physics chain without a wrapper.
- **`01_architecture/`**: understand it. Eight scripts that open one
  layer each, in the order you meet them: the operator contract, the mesh
  taxonomy, the composition rules, the differentiability levels, the
  inversion toolbox, the Bayesian workflow, writing your own operator,
  and getting data in and results out.
- **`02_geomodel/`**: build the earth. Geostatistics and implicit
  geological modelling: the estimation-versus-simulation distinction, the
  variogram everything depends on, categorical facies, multivariate
  conditioning, and geometry that carries gradients.
- **`03_physics/`**: use it. Seven scripts in family order, each a full
  workflow on a real earth model: the Marmousi II benchmark for seismic,
  correlated geostatistical fields elsewhere, and inversions that stop on
  chi-squared rather than on an iteration count. Seismic FWI,
  DC resistivity and induced polarization, petrophysically guided joint
  gravity-magnetics inversion, frequency-domain EM induction, gravity and
  magnetics with remanence, and reservoir flow.
- **`04_decision/`**: decide what to measure next. An inversion says
  what is down there; a survey plan has to say which measurement would
  change the decision, before it is paid for. The efficacy of
  information of a proposed borehole, scored cell by cell from a prior
  ensemble, scanned over every candidate location and depth, and
  repeated on an independent ensemble to separate the finding from the
  noise.

## How the figures work

Twenty-nine scripts could easily be twenty-nine styles. They are not: a
shared toolkit, `_style.py`, sits in each part's directory and every
figure is built through it, so a reader who has learned one figure has
learned them all.

- **Colour has four roles, not nine ramps.** `CMAP_MODEL` (plasma) for a
  property whose value matters; `CMAP_ANOMALY` (RdBu_r, always on limits
  symmetric about zero) for a signed quantity; `CMAP_MAGNITUDE` (YlGnBu,
  light at zero) for a non-negative magnitude; and discrete palette
  colours for classes, because a rock unit is a name rather than a point
  on a ramp. A quantity that is signed in principle but one-sided in the
  data (a chargeability that is never negative, a density deficit that is
  never positive) takes `CMAP_EXCESS` / `CMAP_DEFICIT`, which are the two
  *halves* of `CMAP_ANOMALY`: same colours, same white at zero, but the
  whole bar is values the earth can actually take.
- **Three quantities beat the roles, because the domain got there first.**
  A seismic amplitude is blue-white-red about zero, a velocity model is
  the cool-to-warm rainbow, and a resistivity section is the resistivity
  rainbow: `geo_seismic`, `geo_velocity` and `geo_resistivity`, all
  shipped by `geobrain.vis` and registered by `apply_style`. The last two
  are rainbows and rainbows do have a real defect (their lightness rises
  and then falls, so two values print as one grey and the bright band
  reads as an edge that is not there); they are used anyway, because a
  velocity model in anything else costs a reader more recognition than it
  buys them accuracy. `jet` is not used: same rainbow, sharper false
  edges, no provenance.
- **Line colour is a role or an index, never both.** Truth is black,
  starting models blue, recovered and fitted results orange, posteriors
  green, and a threshold to compare against (chi-squared = 1, the sill,
  the score of the true model) is neutral grey, so it never competes
  with a data series for a hue. `PALETTE[k]` is only for the k-th of n
  things whose only property is being different from each other, and it
  is ordered so the first four are as far apart as the gamut allows.
- **One colour bar per SCALE, not per panel.** Panels showing the same
  quantity share limits and a single bar. That is not tidiness: panels
  drawn on different limits cannot be compared by eye, however firmly the
  caption asks the reader to. Bars are the same thickness whether they
  serve one panel or four, and unless a bar serves the whole grid it sits
  *under* its panels rather than beside them. A bar beside two panels of
  a row of three takes its space out of those two, and the columns stop
  lining up from row to row.
- **Every grid is built from the same canonical panel size**, so a 1×3 and
  a 2×3 look like they came from the same project. The canonical panel is
  landscape, which suits a map; a well display, a depth profile or a
  sounding is read *down* the page and sets a narrow `panel_w` with a tall
  `panel_h`, because a landscape panel flattens the one axis those figures
  are about. Panels are *not* lettered: (a), (b), (c) exist so a caption
  can point at one, and these figures carry their claim in the panel title
  instead. A title says what the panel is and what it *measured*: "RMSE
  0.038", "chi-squared 0.71", "97% of anomalous cells correct". It never
  explains how to read the panel: if a panel needs that, the panel is
  wrong and the fix is in the panel, not in prose above it.
- **No figure-level title.** A figure that captions itself says the same
  thing twice. The claim is already the first line of the script's
  docstring and the "what it proves" column of the tables above, and it
  spends half an inch of every figure doing so. What the figure is comes
  from where it is: a filename, a table row, a gallery tile.
- **A panel has to earn its place.** Every figure is two to six panels.
  What gets cut is a map whose scalar summary is already printed in a
  title, a second panel that repeats the first in different units, or
  three maps that differ by less than the eye can read. Each of those is
  a panel the reader has to rule out before reaching the one that
  matters. Where three near-identical maps were the honest comparison,
  one transect through all three replaced them.
- **Animations only where time is the subject**: a flood front advancing,
  a model sharpening band by band, an ensemble flickering through what it
  does not know. Three of the twenty-nine scripts write a GIF; the rest
  are stills, because a reader cannot scrub back to the frame they wanted.

`geobrain.vis` ships one-call plotters as well as the ramps above; they
are demonstrated where that module is the subject, in
`01_architecture/08_data_io_and_figures.py`.

Every script is seeded, CPU-friendly and self-contained, and saves its
figure next to itself under `out/`. The one exception is the seismic
pair, which reads the Marmousi II sections that
`data/fetch_marmousi.py` downloads and verifies. Run any of them from
the repository root:

```bash
pip install -e ".[examples]"                              # from a checkout
python examples/00_showcase/01_gravity_inversion.py       # see it work
python examples/01_architecture/01_operator_contract.py   # learn the API
python examples/03_physics/01_seismic_fwi.py              # put it to work
python examples/04_decision/01_borehole_efficacy_of_information.py
```

More parts will land here release by release.
