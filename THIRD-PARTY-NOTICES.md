# Third-party notices

GeoBrain 0.2.0 contains or redistributes the following third-party scientific
assets. This notice supplements, and does not replace, the project Apache-2.0
licence.

## Key (2012) Hankel digital-linear filters

- Component: 201-point Hankel digital-linear-filter coefficient tables.
- Publication: Kerry Key, “Is the fast Hankel transform faster than direct
  quadrature?”, *Geophysics* 77(3), 2012, DOI
  `10.1190/geo2011-0237.1`.
- Upstream lineage: `empymod` / `emsig/libdlf` v0.3.0, commit
  `bb95c97b87836f4374e56e6be6541aa3df53aee8`.
- Licence: Creative Commons Attribution 4.0 International (`CC-BY-4.0`).
- GeoBrain transformation: whitespace-delimited float64 columns were
  repackaged as NPZ without numerical transformation. Per-asset provenance and
  SHA-256 digests are shipped beside the tables.

The attribution above must be retained when the coefficient tables are
redistributed. The licence text is available from
<https://creativecommons.org/licenses/by/4.0/>.

## Cephes J0 approximation

- Component: torch Cephes J0 float64 approximation.
- Source lineage: SciPy v1.17.0 / `scipy/xsf`, XSF commit
  `0d0a593fd31073af10062d0093144e13ae34f8f3`, derived from
  `include/xsf/cephes/j0.h`.
- Licence: BSD 3-Clause (`BSD-3-Clause`).
- Copyright: Copyright 1984, 1987, 1989, 2000 by Stephen L. Moshier;
  Copyright (c) 2024, SciPy.

The complete Cephes Math Library Release 2.8 / SciPy notice is distributed as
`geobrain/physics/em/numerics/hankel/CEPHES_J0_LICENSE.txt`; exact source and
transformation metadata are distributed as `cephes_j0_provenance.json`.

Author: Mingliang Liu (mingliangliu@sdu.edu.cn)  
Version: 0.2.0
