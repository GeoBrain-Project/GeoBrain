# Part 4: decide what to measure next

Parts 1 to 3 build an earth and put physics on it. This part spends
money. The question is no longer "what is down there" but "which
measurement, taken next, would change what I decide", and that can be
answered before the measurement is taken, because it depends only on the
prior ensemble and on what the decision is.

| # | Script | The question, and the one thing it teaches |
|---|---|---|
| 01 | `01_borehole_efficacy_of_information.py` | Where should the next hole go, and how deep? A binary indicator ensemble on a 30x50 section, scored with `SpatialDecisionAccuracy`. One 14-cell hole buys 20% of the perfect-information ceiling. Depth outranks location 7 to 1, and repeating the whole scan on an independent prior ensemble says which part of that recommendation is a finding and which is Monte-Carlo noise. The deepest hole buys the most accuracy and the least per cell drilled: a budget question the same map answers |

Three things the script is careful about.

**Efficacy is not a variance map.** Variance says where you do not know;
efficacy says where not knowing costs you a decision. A cell already at
probability 0.99 is worth almost nothing to measure however expensive it
was to model.

**There are two currencies and one of them is a trap.**
`SpatialDecisionAccuracy(normalize=False)` gives the absolute gain in the
probability of deciding correctly, the map to drill on.
`normalize=True` gives each cell's gain as a fraction of that cell's own
ceiling, which goes bright exactly where the prior had already decided
(measured: 0.96 in settled cells against 0.006 for the absolute map, and
the two correlate with prior variance at −0.74 and +0.29 respectively).
Both are plotted, side by side, so the difference is visible rather than
described.

**A scan computed from 250 realisations is an estimate.** The whole
location-and-depth scan is run twice, on independent prior ensembles, and
the script compares their disagreement against the spread it is being
asked to interpret before recommending anything.

The method follows Caers, Scheidt, Yin, Wang, Mukerji and House,
"Efficacy of Information in Mineral Exploration Drilling", *Natural
Resources Research* 31(3):1157 (2022), the spatial example of their
Figures 1 and 2.

The script is seeded, runs on CPU in under two minutes, prints its
measurements as it goes, and writes one figure to `out/`. Nothing is read
from disk. Run it from the repository root:

```bash
python examples/04_decision/01_borehole_efficacy_of_information.py
```
