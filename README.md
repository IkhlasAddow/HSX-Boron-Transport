# Standalone HSX Boron 1D Transport Model

This folder contains a revised 1D radial boron impurity transport model.

The code is **inspired by the STRAHL / pySTRAHL impurity transport workflow**,
but this is a simplified standalone model. It shows the modeling framework and
the sensitivity to uncertain transport inputs. **It should not be presented as
a validated boron transport prediction.**

---

## What the model solves

The model evolves six boron charge states — B0, B1+, B2+, B3+, B4+, B5+ — on
a 1D radial grid in the minor-radius coordinate. For each charge state it
solves:

```
dn_Z/dt = -(1/r) d/dr [ r(-D_Z dn_Z/dr + V_Z n_Z) ]
          + ionization/recombination coupling
          + source (B0 only)
```

Radial geometry: cylindrical approximation in minor radius r, which is standard
for 1D impurity transport. The 4π²R toroidal factor cancels identically in the
finite-volume operator. This is **not** a full 3D VMEC flux-surface calculation.

Solver: implicit BDF (L-stable) via `scipy.integrate.solve_ivp`. BDF is
required because the B0→B1+ ionization timescale (~2 µs at HSX core
conditions) creates stiffness ratios of ~10⁴ over a 50 ms simulation.

**Known numerical approximations:**
- First-order upwinding for convection (introduces small numerical diffusion).
- Zero-gradient (Neumann) outer boundary condition — standard but can produce
  mild artificial density accumulation near the LCFS at early times.
- No SOL physics, no wall recycling, no radiation transport.

---

## What is measured

The measured inputs in this version are:

| Input | Source |
|---|---|
| Te(r), ne(r) | HSX Thomson scattering, shot 37, t = 0.83–0.85 s (`DATA/2026-04-15_shot37_t0p83-0p85.mat`) |
| B ionization rates | OpenADAS SCD89 (`DATA/scd89_b.dat`) |
| B recombination rates | OpenADAS ACD89 (`DATA/acd89_b.dat`) |

Thomson coverage: ρ = 0.016–0.906 (10 chords). The LCFS anchor (ρ = 1.0) is
**assumed** (see below). Error bars are loaded from the `.mat` file and shown
on Plot 01.

The atomic rates include:
- Electron-impact ionization
- Radiative and dielectronic recombination

**Charge-exchange recombination (CX) is NOT included.** CX depends on Ti and
requires CX cross-sections not available in the current data files. For HSX
conditions with moderate Ti (~70 eV) and low neutral density, CX is secondary
to radiative recombination, but this should be revisited if Ti is large.

---

## What is assumed or scanned

The following quantities are **not measured** in this model:

| Input | Status | What to do |
|---|---|---|
| Ti(r) | Constant, assumed | Replace with processed CHERS Ti(r) |
| Er(r), pinch V(r) | Assumed shape, amplitude unconstrained | Replace with CHERS force balance or ambipolar Er |
| Anomalous D(r) | Three generic cases (low/medium/high) | Constrain from B spectroscopy or turbulence study |
| Neutral source rate | Order-of-magnitude placeholder (1×10²⁰ s⁻¹) | Replace with dropper calibration or B spectroscopy |
| LCFS Te, ne | Assumed edge anchor (Te=40 eV, ne=1×10¹⁸ m⁻³) | Replace with Langmuir probe or reflectometer data |

**Transport is not boron-specific.** The default D and V profiles are generic
sensitivity-study inputs. At runtime the code prints a clear warning:

```
WARNING: Transport is not boron-specific unless a future boron transport CSV is supplied.
```

The pinch amplitude convention (`charge_scaled` case):
```
V_Z = -0.08 × (Z / 5) × shape(r)   [m/s]
```
The denominator 5 normalizes to Z_max = 5 (fully stripped boron, B5+), so B5+
receives the full amplitude of −0.08 m/s and lower charge states receive
proportionally less. **This is NOT the same as Z/6 scaling** (which would be
used for a C6+ analogy). Neither the amplitude nor the shape is derived from an
Er measurement.

**Source rate caveat:** The default source rate of 10²⁰ s⁻¹ over 50 ms
deposits ~5×10¹⁸ boron atoms — comparable in magnitude to the electron
inventory. This is an order-of-magnitude placeholder only. Replace it with a
physical estimate from dropper calibration or boron emission spectroscopy before
any quantitative comparison.

---

## Important transport caveat

This model does **not** claim that the transport coefficients are boron-specific.

In HSX pySTRAHL carbon analyses, transport profiles were constrained by
impurity spectroscopy data. For boron in the current campaign, that workflow
has not been completed. Therefore this code uses generic anomalous diffusion
and an assumed pinch unless you provide a boron-specific transport file.

Only when a future CSV from SFINCS/PENTA or a validated spectroscopic fit is
supplied should transport be described as boron-specific.

---

## Limitations summary

The following limitations should be stated whenever results are presented:

1. 1D cylindrical geometry — not full 3D stellarator flux-surface transport.
2. No SOL physics — LCFS is a loss boundary, not a recycling boundary.
3. No neutral penetration model — the source is a prescribed Gaussian.
4. No charge-exchange recombination.
5. Ti is assumed constant — CHERS Ti(r) needed for quantitative accuracy.
6. Pinch is assumed — Er(r) measurement or neoclassical calculation needed.
7. D is scanned generically — not fitted to boron data.
8. Source rate is a placeholder — needs experimental calibration.
9. Outer boundary condition (Neumann) can cause mild edge density artifact.
10. First-order upwinding introduces small numerical diffusion at low D.

---

## Folder structure

```
hsx_boron_transport_revised/
│
├── run_model.py               Main single-run script
├── run_transport_scan.py      Scan over low/medium/high D cases
├── hsx_boron_inputs.py        Profile loader, ADAS reader, transport arrays
├── boron_transport_solver.py  Finite-volume solver
├── requirements.txt
├── README.md
│
├── DATA/
│   ├── 2026-04-15_shot37_t0p83-0p85.mat   Thomson Te/ne (measured)
│   ├── scd89_b.dat                          OpenADAS boron SCD
│   ├── acd89_b.dat                          OpenADAS boron ACD
│   └── wout_HSX_qhsExtend.nc               VMEC (used only for a, R)
│
└── OUTPUT/
```

---

## Installation

```
python -m pip install numpy scipy matplotlib
```

If pySTRAHL is installed, the code will use the VMEC helper to read HSX radii.
If not, it falls back to nominal values: a = 0.12 m, R = 1.20 m.

---

## How to run

### Single case

```
python run_model.py
```

Default: Ti = 70 eV, D = medium, pinch = charge_scaled, source = 1×10²⁰ s⁻¹.

```
python run_model.py --ti 70 --D-case medium --pinch charge_scaled
python run_model.py --ti 40 --D-case low --pinch off
python run_model.py --ti 100 --D-case high --pinch assumed_inward
```

Outputs are saved in `OUTPUT/Ti70_medium_charge_scaled/`.

### Transport scan

```
python run_transport_scan.py
```

Runs all three D cases and saves comparison plots in `OUTPUT/transport_scan/`.

### Future boron-specific transport

```
python run_model.py --future-boron-transport-csv path/to/boron_transport.csv
```

Expected CSV columns: `rho, D_B1, V_B1, D_B2, V_B2, D_B3, V_B3, D_B4, V_B4, D_B5, V_B5`.
Only with this file should transport be called boron-specific.

---

## Plots produced by `run_model.py`

### `01_profiles_used.png`

Te and ne profiles from Thomson scattering (shot 37) with raw data points and
error bars overlaid on the PCHIP fit. Ti is shown with a clear warning that it
is assumed and not measured. The grey band shows the region beyond Thomson
coverage (ρ > 0.906) where the profile is extrapolated to the assumed LCFS
anchor.

### `02_adas_rates_vs_radius.png`

Boron electron-impact ionization and recombination rates (s⁻¹) evaluated along
the HSX profile, for all five transitions. Derived from measured Te and ne
(ADAS SCD89/ACD89). CX recombination is not included.

### `03_transport_coefficients_used.png`

Assumed diffusivity (identical for all charge states in the generic case) and
assumed pinch velocity (scaled by Z/5 in the charge_scaled case). Both panels
carry explicit warnings that these are assumed inputs, not measurements or
boron-specific computations.

### `04_damkohler_timescale_map.png`

A 2D map in (ρ, D) space showing log₁₀(τ_D / τ_iz) for neutral boron.
τ_D = L² / D where L = source deposition width (a fixed global parameter,
**not** a local gradient scale length). The white line shows the D profile used
in this run. This is a diagnostic tool — it shows which transport–ionization
regime the model operates in for the chosen source geometry. It does not
validate the transport model.

### `05_charge_states_final.png`

Left panel: final radial profiles normalized to the overall maximum (shape
only), with charge-state fractions at ρ = 0.5 annotated in the corner.
Right panel: fractional populations compared to local coronal equilibrium
fractions (dashed). Agreement in the core indicates the atomic physics has
converged self-consistently; deviations reveal transport-driven departures
from equilibrium.

### `06_core_fraction_vs_time.png`

Time evolution of two core penetration metrics:
- Core fraction: r-weighted particle inventory inside ρ < 0.5.
- Peaking factor: n(0) / ⟨n⟩_vol — values > 1 indicate on-axis concentration.

The case parameters are annotated on each panel.

### `07_summary_single_run.png`

Compact summary for presentations: profiles, assumed D, final total boron
profile, and core fraction time trace. Includes a B5+ core fraction and
qualitative ΔZ_eff estimate. Title explicitly labels transport as assumed.

---

## Plots produced by `run_transport_scan.py`

### `01_diffusivity_cases.png`

The three assumed D profiles (low/medium/high) with an annotation that these
are generic, not boron-specific.

### `02_core_fraction_scan.png`

Core fraction vs time for all three D cases, annotated with Ti and pinch
assumptions.

### `03_charge_state_scan.png`

Final charge-state profiles (normalized shapes) for all three cases, with
fractional population annotations at ρ = 0.5.

### `04_peaking_factor_scan.png`

On-axis peaking factor vs time for all three D cases.

### `05_scan_summary_bar.png`

Bar chart of final core fraction for the three D cases. Annotated with all
assumed inputs (Ti, pinch, source rate).

---

## What would make the model predictive

To make this model quantitatively predictive, the assumed transport inputs
should be replaced with:

1. **CHERS Ti(r):** Measured ion temperature profile for the relevant shot.
   Use the processed CHERS chord geometry and inversion for the HSX geometry.

2. **Er(r) from CHERS force balance or ambipolar calculation:** Replace the
   assumed pinch with an experimental or neoclassical radial electric field.
   The ambipolar Er for HSX can be computed with SFINCS using the measured
   profiles and VMEC equilibrium.

3. **Boron-specific neoclassical D and V from SFINCS/PENTA:** Run SFINCS with
   the HSX QHS VMEC equilibrium, measured Te/ne profiles, and boron as the
   impurity species. HSX's quasi-helical symmetry suppresses neoclassical
   transport relative to non-optimized stellarators — this is a key physics
   prediction the framework is designed to test.

4. **Anomalous D constraint from boron spectroscopy:** Use boron line emission
   time evolution (BII, BIII, BIV, BV) or MIMS/CHERS line evolution to fit
   the anomalous diffusivity and distinguish it from the neoclassical
   contribution.

Until those are available, this model should be presented as a transport
framework and sensitivity study — not a final prediction.
