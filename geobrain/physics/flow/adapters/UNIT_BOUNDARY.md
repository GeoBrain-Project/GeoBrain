# Flow unit boundary: strict SI kernel, deck-side conversions

Every flow kernel solves the SI Darcy forms directly: `F = mob·T·Δp` in
m³/s, `acc = V·Δ(φ/B)/dt` with dt in seconds, gravity `ρ·g·z` with
`core.constants.STANDARD_GRAVITY`, across the TPFA family (SinglePhase /
OilWater / BlackOil / BlackOilVarSwitch) and the MPFA / NFVM /
compositional tier alike. No FIELD unit constant (ALPHA, BETA, M = 1/144,
ft/s² G/GC, day-based DT bounds) exists anywhere in the kernels, and a
governance test fails if one is reintroduced.

## The slot map is literal

| Slot | Carries |
|---|---|
| `Rock.permeability_m2` | m² |
| `Rock.reference_pressure_pa`, `PVT* *_pa` | Pa |
| `Rock/PVT* *_pa_inv` | 1/Pa |
| `PVTAnalytic.viscosity_ref_pa_s` | Pa·s |
| `PVTAnalytic.density_ref_kg_m3` | kg/m³ |
| `CartGrid dx_m / dy_m / dz_m` | m |
| operator `t_end` / scheduler `dt_list` | s |
| state / `final_pressure` | Pa |
| residual blocks | m³/s of surface volume (schema unit `m³/s`) |
| gas FVF slots | rm³/sm³ |

Anchored by the frame-arbiter test (a 1-D eigenmode decay matches the
constant-free SI prediction; rel ≤ 1e-6) and by frozen equivalence pins:
the SI kernel reproduces the retired FIELD-kernel physics for every TPFA
model to rel ≤ 5e-4, the deviation being entirely the old kernel's
4-significant-figure ALPHA rounding (exact mapping constants agree at
1.03e-4).

## Remaining crossings

1. **Wells currency**: the wells layer speaks phase MASS (kg/s); the
   TPFA models speak surface VOLUME (m³/s). `well_system` divides /
   multiplies by the explicit standard density
   (`_volume_rate_from_mass` / `_mass_from_volume_residual`); state
   pressure and dt pass through untouched, guarded by the SI-boundary
   contract tests.
2. **Deck ingest**: Eclipse FIELD / METRIC decks convert ONCE to SI in
   `adapters/eclipse.py::read_eclipse_deck_si`; `read_eclipse_case`
   accepts the result directly.
3. **Interchange**: the `field_units.py` conversion library is the
   published way to bring external field-unit data in/out; every pair is
   pinned (constant + round-trip) and the inventory below must name every
   export (completeness-guarded).

## Conversion library (`field_units.py`)

`pressure_psi_to_pa` / `pressure_pa_to_psi`,
`compressibility_psi_inv_to_pa_inv` / `compressibility_pa_inv_to_psi_inv`,
`density_lbm_ft3_to_kg_m3` / `density_kg_m3_to_lbm_ft3`,
`temperature_c_to_k` / `temperature_k_to_c`,
`length_ft_to_m` / `length_m_to_ft`,
`permeability_md_to_m2` / `permeability_m2_to_md`,
`viscosity_cp_to_pa_s` / `viscosity_pa_s_to_cp`,
`time_day_to_s` / `time_s_to_day`,
`liquid_rate_stb_day_to_m3_s` / `liquid_rate_m3_s_to_stb_day`,
`gas_rate_scf_day_to_m3_s` / `gas_rate_m3_s_to_scf_day`.
