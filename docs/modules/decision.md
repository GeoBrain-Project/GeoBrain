# Decision under uncertainty (`geobrain.decision`)

An inversion says what is down there. A survey plan has to say which
measurement would **change the decision**, before it is paid for.

| Piece | What it answers |
|---|---|
| `SpatialDecisionAccuracy` | how often a decision made cell by cell would be right |
| `expected_accuracy_gain` | how much a proposed measurement improves that |
| `ValueOfInformation`, `VOIResult` | the same question in money |
| `expected_utility_gain` | when the payoff is not accuracy |
| `MutualInformationEstimator` | how much a measurement tells you about the model, in bits |
| `ClosedLoopManager` | measure, update, decide, repeat, with `EnsembleUpdater` and `HistoryPolicy` |

## Two currencies that disagree

`SpatialDecisionAccuracy` can report the gain in two currencies, and they do
not agree about where to drill.

The **absolute** gain is the increase in the probability of deciding correctly,
and it is the map to drill on.

The **normalised** gain is that increase as a fraction of the ceiling available
at each cell, and it **saturates to 1 wherever the prior had already decided**.
A cell the prior was certain about has almost no ceiling, so buying almost none
of it still scores near-perfect. Score a survey in that currency and it will
send you to drill where you already knew the answer.

```{admonition} Measured, not asserted
:class: note

On the worked example the normalised map averages 0.95 over cells whose prior
variance is below 0.01, the cells where nothing was left to learn. The
absolute map averages 0.044 across the whole section, which is 20% of the
theoretical ceiling. Those are the two numbers to compare.
```

Both are useful; they answer different questions. The normalised one is a
capture-fraction diagnostic, answering "of what was learnable here, how
much did we get?", not a drilling target.

```{figure} /_figures/01_borehole_efficacy_of_information.png
:class: gb-tall
:alt: One borehole scored cell by cell

The efficacy of one proposed borehole, scored cell by cell from a prior
ensemble. From `examples/04_decision/01_borehole_efficacy_of_information.py`.
```

## What a design scan buys

Scanning every candidate location and depth turns the question from "is this
hole worth drilling" into "which hole". Two findings from the worked example
are worth carrying into your own:

- **Depth outranked location by 7×.** Averaged over location, the gain spanned
  a range seven times wider than it did averaged over depth. If you can only
  optimise one thing, optimise the one that moves the number.
- **The lateral preference did not survive a fresh prior ensemble.** Two
  independent ensembles disagreed by 0.003 on average, against a
  location-to-location spread of 0.008, and picked favourite locations eight
  cells apart. A design that changes when you redraw the prior is not a design.

That second check is the one most easily skipped, and the one that decides
whether a recommendation is real.

## See also

- `examples/04_decision/01_borehole_efficacy_of_information.py`: the efficacy
  of one borehole in both currencies, scanned over every candidate location and
  depth, and repeated on an independent ensemble.
