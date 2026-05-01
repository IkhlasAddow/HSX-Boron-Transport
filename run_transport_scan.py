"""
run_transport_scan.py

Run low / medium / high generic anomalous transport cases and produce
comparison plots.

This script exists to make the sensitivity to assumed diffusivity explicit.
None of the three cases should be presented as the correct answer — they are
all sensitivity inputs.  The spread in core penetration is the result.

CAVEATS:
  - Transport is NOT boron-specific.  D and V are assumed/scanned.
  - Ti is assumed constant.  Replace with CHERS Ti(r) when available.
  - Source rate 1e20 s^-1 is an order-of-magnitude placeholder.
  - n_points=80 gives a reasonable balance of speed and spatial resolution.
    For publication-quality results use n_points >= 100.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from hsx_boron_inputs import load_profiles, load_boron_adas_rates, build_transport_arrays
from boron_transport_solver import make_edge_neutral_source, solve_boron_transport


ROOT     = Path(__file__).resolve().parent
DATA_DIR = ROOT / "DATA"
OUT_DIR  = ROOT / "OUTPUT" / "transport_scan"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHARGE_LABELS = ["B0", "B1+", "B2+", "B3+", "B4+", "B5+"]
CHARGE_COLORS = plt.cm.tab10(np.linspace(0, 0.6, 6))

# Source rate: order-of-magnitude placeholder.  See README for caveats.
SOURCE_RATE_S = 1.0e20


def core_fraction(result, rho_cut=0.5):
    """Particle-number fraction inside rho < rho_cut (r-weighted)."""
    mask    = result.rho < rho_cut
    weights = np.maximum(result.r_m, result.r_m[1])
    total   = np.sum(result.nZ * weights[None, None, :], axis=(1, 2))
    core    = np.sum(result.nZ[:, :, mask] * weights[mask][None, None, :], axis=(1, 2))
    return np.divide(core, total, out=np.zeros_like(core), where=total > 0)


def peaking_factor(result):
    """n_total(rho=0) / <n_total>_vol at each time step."""
    weights = np.maximum(result.r_m, result.r_m[1])
    n_total = np.sum(result.nZ, axis=1)          # (nt, nr)
    n_axis  = n_total[:, 0]
    n_avg   = np.sum(n_total * weights[None, :], axis=1) / np.sum(weights)
    return np.divide(n_axis, n_avg, out=np.ones_like(n_axis), where=n_avg > 0)


def run_case(D_case, ti_eV=70.0, pinch_case="charge_scaled", n_points=80):
    """Run one transport case and return (profiles, result, metadata)."""
    profiles = load_profiles(DATA_DIR, n_points=n_points, ti_case_eV=ti_eV)
    ionization_s, recombination_s, rates_meta = load_boron_adas_rates(profiles, DATA_DIR)
    transport = build_transport_arrays(profiles.rho, D_case=D_case, pinch_case=pinch_case)

    source = make_edge_neutral_source(
        profiles.rho,
        total_particles_s=SOURCE_RATE_S,
        minor_radius_m=profiles.minor_radius_m,
        major_radius_m=profiles.major_radius_m,
        width=0.035,
        center=0.98,
    )

    metadata = {
        "description": "Generic anomalous transport scan for HSX boron class-project model.",
        "D_case": D_case,
        "Ti_eV": ti_eV,
        "Ti_note": "ASSUMED constant — not measured; replace with CHERS Ti(r)",
        "pinch_case": pinch_case,
        "pinch_note": (
            "Assumed shape and amplitude — not from Er measurement. "
            "charge_scaled: amplitude = -0.08 * Z/5 m/s (Z_max=5 for boron)."
        ),
        "source_rate_particles_s": SOURCE_RATE_S,
        "source_rate_note": (
            "Order-of-magnitude placeholder. Replace with dropper calibration "
            "or boron spectroscopy constraint."
        ),
        "transport_is_boron_specific": False,
        "important_note": (
            "D and V are assumed/scanned; boron-specific ADAS rates and "
            "measured Thomson Te/ne are used."
        ),
    }

    result = solve_boron_transport(
        profiles.rho,
        profiles.r_m,
        transport.D_by_Z_m2_s,
        transport.V_by_Z_m_s,
        ionization_s,
        recombination_s,
        source,
        t_end=0.05,
        n_save=150,
        metadata=metadata,
    )
    return profiles, result, metadata, ionization_s, recombination_s




def main():
    print("=" * 72)
    print("HSX boron generic transport scan")
    print("WARNING: D, Ti, pinch are all assumed/scanned — not boron-specific.")
    print(f"WARNING: Source rate = {SOURCE_RATE_S:.0e} s^-1 is a placeholder.")
    print("=" * 72)

    cases      = ["low", "medium", "high"]
    results    = {}
    iz_all     = {}
    rec_all    = {}
    profiles_ref = None

    for case in cases:
        print(f"  Running D case: {case} ...", flush=True)
        profiles, result, metadata, iz, rec = run_case(case)
        results[case]  = result
        iz_all[case]   = iz
        rec_all[case]  = rec
        profiles_ref   = profiles
        with open(OUT_DIR / f"metadata_{case}.json", "w") as f:
            json.dump(metadata, f, indent=2)

    # ------------------------------------------------------------------
    # 01 — Diffusivity profiles
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for case in cases:
        ax.semilogy(results[case].rho, results[case].D_m2_s[0], label=f"D = {case}")
    ax.set_xlabel("r/a")
    ax.set_ylabel("D [m$^2$/s]")
    ax.set_title("Generic anomalous diffusivity scan\n(same D for all charge states in each case)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(title="D case")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_diffusivity_cases.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 02 — Core fraction versus time
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    final_fracs = []
    for case in cases:
        frac = core_fraction(results[case])
        final_fracs.append(frac[-1])
        ax.plot(results[case].t, frac, label=f"{case}: final = {frac[-1]:.3f}")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("fraction inside r/a < 0.5")
    ax.set_title("Core penetration sensitivity to assumed diffusivity\n"
                 "(r-weighted particle fraction; Ti = 70 eV assumed; charge_scaled pinch)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_core_fraction_scan.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 03 — Charge-state profiles (normalized shapes)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True, dpi=150)
    for j, case in enumerate(cases):
        result = results[case]
        final  = result.nZ[-1]
        maxval = np.max(final) if np.max(final) > 0 else 1.0

        # Fraction annotation at rho ~ 0.5
        i_half  = np.argmin(np.abs(result.rho - 0.5))
        n_half  = final[:, i_half]
        tot_h   = n_half.sum()
        fracs   = n_half / tot_h if tot_h > 0 else np.zeros(6)
        frac_str = "\n".join([f"{CHARGE_LABELS[z]}: {100*fracs[z]:.1f}%" for z in range(6)])

        for Z in range(6):
            axes[j].plot(result.rho, final[Z] / maxval,
                         color=CHARGE_COLORS[Z], label=CHARGE_LABELS[Z])
        axes[j].set_title(f"D = {case}")
        axes[j].set_xlabel("r/a")
        axes[j].grid(alpha=0.3)
        axes[j].text(0.97, 0.97, f"Fractions at r/a=0.5:\n{frac_str}",
                     transform=axes[j].transAxes, fontsize=6.5,
                     va="top", ha="right",
                     bbox=dict(facecolor="white", edgecolor="0.5", alpha=0.88))

    axes[0].set_ylabel("density (normalized to overall max\nfor each case — SHAPE only)")
    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle(
        "Final charge-state profiles: low / medium / high assumed D\n"
        "Normalized to per-case maximum — see inset for fractional populations at r/a=0.5",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_charge_state_scan.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 04 — Peaking factor scan
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
    for case in cases:
        pkf = peaking_factor(results[case])
        ax.plot(results[case].t, pkf, label=case)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("peaking factor  n(0) / <n>$_\\mathrm{vol}$")
    ax.set_title("On-axis peaking of total boron vs assumed D\n"
                 "(> 1 indicates core accumulation)")
    ax.grid(alpha=0.3)
    ax.legend(title="D case")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_peaking_factor_scan.png", dpi=200)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 05 — Summary bar chart
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.5), dpi=150)
    colors = ["C0", "C1", "C2"]
    bars = ax.bar(cases, final_fracs, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
    ax.set_ylabel("final fraction inside r/a < 0.5")
    ax.set_title("Summary: core penetration vs assumed D\n"
                 "All cases use assumed Ti=70 eV, charge_scaled pinch, source=1e20 s⁻¹")
    ax.set_ylim(0, max(final_fracs) * 1.25)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_scan_summary_bar.png", dpi=200)
    plt.close(fig)

    # Save CSV summary.
    np.savetxt(
        OUT_DIR / "scan_summary.csv",
        np.column_stack([np.arange(len(cases)), final_fracs]),
        delimiter=",",
        header="case_index,final_core_fraction\n# 0=low  1=medium  2=high\n"
               "# Ti=70eV assumed, pinch=charge_scaled, source=1e20 s^-1 (placeholder)",
        comments="",
    )

    print("\nSaved scan outputs to:", OUT_DIR)
    for case, frac in zip(cases, final_fracs):
        print(f"  {case:>6s}: final core fraction = {frac:.4f}")


if __name__ == "__main__":
    main()