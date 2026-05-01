"""
run_model.py

Run the revised standalone HSX boron 1D transport model.

This is a presentation/class-project model inspired by STRAHL/pySTRAHL.  It
uses measured HSX Thomson Te/ne and boron OpenADAS rates, but it treats Ti,
pinch, and anomalous diffusivity as assumed/scanned inputs unless a future
boron SFINCS/PENTA transport CSV is provided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from hsx_boron_inputs import (
    load_profiles,
    load_boron_adas_rates,
    build_transport_arrays,
    generic_diffusivity_profile,
)
from boron_transport_solver import make_edge_neutral_source, solve_boron_transport


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "DATA"
OUT_DIR = ROOT / "OUTPUT"
OUT_DIR.mkdir(exist_ok=True)


CHARGE_LABELS = ["B0", "B1+", "B2+", "B3+", "B4+", "B5+"]
# Color cycle consistent across all plots.
CHARGE_COLORS = plt.cm.tab10(np.linspace(0, 0.6, 6))


def parse_args():
    p = argparse.ArgumentParser(description="Run the HSX boron 1D transport model.")
    p.add_argument("--ti", type=float, default=70.0, choices=[40.0, 70.0, 100.0],
                   help="Assumed ion temperature in eV: 40, 70, or 100.")
    p.add_argument("--D-case", default="medium", choices=["low", "medium", "high"],
                   help="Generic anomalous diffusivity case.")
    p.add_argument("--pinch", default="charge_scaled",
                   choices=["off", "assumed_inward", "charge_scaled"],
                   help="Pinch model.")
    p.add_argument("--source-rate", type=float, default=1e20,
                   help="Total neutral boron source in particles/s.  "
                        "Default 1e20 s^-1 is an order-of-magnitude placeholder; "
                        "replace with a dropper calibration or spectroscopy estimate.")
    p.add_argument("--source-width", type=float, default=0.035,
                   help="Gaussian source width in normalized radius.")
    p.add_argument("--source-center", type=float, default=0.98,
                   help="Gaussian source center in normalized radius.")
    p.add_argument("--t-end", type=float, default=0.05,
                   help="Simulation end time [s].")
    p.add_argument("--n-points", type=int, default=100,
                   help="Number of radial grid points.")
    p.add_argument("--future-boron-transport-csv", default=None,
                   help="Optional boron-specific transport CSV from SFINCS/PENTA or validated fitting.")
    return p.parse_args()


def core_fraction(result, rho_cut=0.5):
    """
    Fraction of total boron inventory inside rho < rho_cut.

    Weighted by r (proportional to cylindrical volume element dV ~ r dr),
    so this is a particle-number fraction, not a line-averaged fraction.
    """
    mask = result.rho < rho_cut
    weights = np.maximum(result.r_m, result.r_m[1])
    total_vs_t = np.sum(result.nZ * weights[None, None, :], axis=(1, 2))
    core_vs_t  = np.sum(result.nZ[:, :, mask] * weights[mask][None, None, :], axis=(1, 2))
    return np.divide(core_vs_t, total_vs_t, out=np.zeros_like(core_vs_t), where=total_vs_t > 0)


def peaking_factor(result):
    """
    n_total(rho=0) / <n_total>_vol  at each time step.

    A peaking factor > 1 indicates on-axis accumulation.
    """
    weights = np.maximum(result.r_m, result.r_m[1])
    n_total = np.sum(result.nZ, axis=1)          # shape (nt, nr)
    n_axis  = n_total[:, 0]
    n_avg   = (np.sum(n_total * weights[None, :], axis=1) /
               np.sum(weights))
    return np.divide(n_axis, n_avg, out=np.ones_like(n_axis), where=n_avg > 0)


def coronal_equilibrium_fractions(ionization_s, recombination_s):
    """
    Local coronal equilibrium charge-state fractions f_Z(r).

    Coronal equilibrium assumes ionization and recombination balance locally,
    ignoring transport.  If the time-dependent solution converges to these
    fractions in the core, it validates that the atomic physics is self-
    consistent.  Deviations reveal transport-driven departures from equilibrium.

    Returns f_Z: shape (nr, 6), with sum(f_Z, axis=1) = 1.
    """
    nr = ionization_s.shape[0]
    f = np.zeros((nr, 6))
    f[:, 0] = 1.0  # start with all neutral

    for Z in range(5):
        iz  = np.maximum(ionization_s[:, Z], 1e-40)
        rec = np.maximum(recombination_s[:, Z], 1e-40)
        f[:, Z + 1] = f[:, Z] * iz / rec

    # Normalize to unit sum.
    total = f.sum(axis=1, keepdims=True)
    f /= np.maximum(total, 1e-40)
    return f


def make_case_label(args):
    """Short one-line label summarizing run parameters for figure annotations."""
    return (f"D={args.D_case}, pinch={args.pinch}, "
            f"Ti={int(args.ti)} eV, source={args.source_rate:.0e} s\u207b\xb9")


def make_metadata(args, profiles, transport_inputs, rates_meta):
    import scipy, numpy
    measured = {
        "Te_ne": profiles.Te_source + "; " + profiles.ne_source,
        "atomic_rates": rates_meta,
    }
    assumed = {
        "Ti": profiles.Ti_source,
        "source_rate_particles_s": args.source_rate,
        "source_rate_note": (
            "Order-of-magnitude placeholder.  Replace with dropper calibration "
            "or boron spectroscopy constraint before any quantitative comparison."
        ),
        "source_width_rho": args.source_width,
        "source_center_rho": args.source_center,
        "D_case": args.D_case,
        "pinch_case": args.pinch,
        "pinch_note": (
            "Assumed shape and amplitude; not derived from Er measurement. "
            "charge_scaled uses amplitude = -0.08 * Z/5 m/s, normalized so "
            "B5+ (Z_max) receives the full -0.08 m/s."
        ),
    }
    future_work = [
        "Replace constant Ti with processed CHERS Ti(r).",
        "Use measured or ambipolar Er(r) instead of an assumed pinch.",
        "Replace generic D and V with boron-specific SFINCS/PENTA transport when available.",
        "Use boron injection data / spectroscopy to constrain anomalous diffusivity and source rate.",
    ]
    return {
        "model_description": "Standalone 1D radial boron impurity transport model inspired by STRAHL/pySTRAHL.",
        "transport_warning": (
            "Transport is not boron-specific unless future_boron_transport_csv was supplied. "
            "Default D and V are assumed/scanned inputs for a sensitivity study."
        ),
        "transport_is_boron_specific": transport_inputs.is_boron_specific,
        "measured_inputs": measured,
        "assumed_or_scanned_inputs": assumed,
        "future_work_to_make_predictive": future_work,
        "software_versions": {
            "python": sys.version,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
        },
    }


CHARGE_LABELS = ["B0", "B1+", "B2+", "B3+", "B4+", "B5+"]
CHARGE_COLORS = plt.cm.tab10(np.linspace(0, 0.6, 6))


# ---------------------------------------------------------------------------
# Plot 01 — profiles
# ---------------------------------------------------------------------------
def plot_profiles(profiles, out_dir):
    """Te, ne, Ti profiles with Thomson scatter points and error bars."""
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7.5, 7.8), dpi=150)
    lcfs_rho = profiles.rho_ts[-1] if profiles.rho_ts is not None else 0.906

    # --- Te ---
    ax[0].plot(profiles.rho, profiles.Te_eV, color="C0", lw=1.8,
               label="Te — PCHIP fit")
    if profiles.rho_ts is not None:
        ax[0].scatter(profiles.rho_ts, profiles.Te_ts, color="C0", s=28, zorder=5,
                      label="Te — Thomson data (shot 37)")
        if profiles.Te_ts_err_hi is not None and profiles.Te_ts_err_lo is not None:
            ax[0].errorbar(profiles.rho_ts, profiles.Te_ts,
                           yerr=[profiles.Te_ts_err_lo, profiles.Te_ts_err_hi],
                           fmt="none", color="C0", alpha=0.5, capsize=3)
    ax[0].axvline(lcfs_rho, color="0.5", ls=":", lw=0.9, label="edge of Thomson coverage")
    ax[0].set_ylabel("Te [eV]")
    ax[0].legend(fontsize=7.5)
    ax[0].grid(alpha=0.3)
    ax[0].set_title("HSX profiles used in transport model — shot 37", fontsize=10)

    # --- ne ---
    ax[1].plot(profiles.rho, profiles.ne_m3, color="C1", lw=1.8,
               label="ne — PCHIP fit")
    if profiles.rho_ts is not None:
        ax[1].scatter(profiles.rho_ts, profiles.ne_ts, color="C1", s=28, zorder=5,
                      label="ne — Thomson data (shot 37)")
        if profiles.ne_ts_err_hi is not None and profiles.ne_ts_err_lo is not None:
            ax[1].errorbar(profiles.rho_ts, profiles.ne_ts,
                           yerr=[profiles.ne_ts_err_lo, profiles.ne_ts_err_hi],
                           fmt="none", color="C1", alpha=0.5, capsize=3)
    ax[1].axvline(lcfs_rho, color="0.5", ls=":", lw=0.9)
    ax[1].set_ylabel("ne [m$^{-3}$]")
    ax[1].legend(fontsize=7.5)
    ax[1].grid(alpha=0.3)

    # --- Ti (assumed constant — note carried in legend label) ---
    ax[2].plot(profiles.rho, profiles.Ti_eV, color="C2", lw=1.8,
               label=f"Ti = {profiles.Ti_eV[0]:.0f} eV — assumed constant (no CHERS)")
    ax[2].set_ylabel("Ti [eV]")
    ax[2].set_xlabel("r/a")
    ax[2].legend(fontsize=7.5)
    ax[2].grid(alpha=0.3)

    # Grey band beyond Thomson coverage.
    for a in ax:
        a.axvspan(lcfs_rho, 1.0, alpha=0.07, color="grey")

    fig.tight_layout()
    fig.savefig(out_dir / "01_profiles_used.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 02 — ADAS rates
# ---------------------------------------------------------------------------
def plot_rates(profiles, ionization_s, recombination_s, out_dir):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7.5, 6.2), dpi=150)

    for Z in range(5):
        ax[0].semilogy(profiles.rho, np.maximum(ionization_s[:, Z], 1e-40),
                       color=CHARGE_COLORS[Z],
                       label=f"{CHARGE_LABELS[Z]} \u2192 {CHARGE_LABELS[Z+1]}")
    ax[0].set_ylabel("ionization rate [s$^{-1}$]")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].set_title(
        "Boron OpenADAS rates (scd89_b / acd89_b) along HSX shot-37 profile\n"
        "(electron-impact only; CX recombination not included)",
        fontsize=9,
    )

    for Z in range(5):
        ax[1].semilogy(profiles.rho, np.maximum(recombination_s[:, Z], 1e-40),
                       color=CHARGE_COLORS[Z + 1],
                       label=f"{CHARGE_LABELS[Z+1]} \u2192 {CHARGE_LABELS[Z]}")
    ax[1].set_ylabel("recombination rate [s$^{-1}$]")
    ax[1].set_xlabel("r/a")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend(fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / "02_adas_rates_vs_radius.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 03 — transport coefficients
# ---------------------------------------------------------------------------
def plot_transport(result, out_dir, is_boron_specific: bool):
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7.5, 6.2), dpi=150)

    transport_note = ("boron-specific (from CSV)" if is_boron_specific
                      else "generic assumed — sensitivity study only, not boron-specific")
    ax[0].semilogy(result.rho, result.D_m2_s[0], color="C0", lw=2,
                   label="D [m$^2$/s] — same for all charge states")
    ax[0].set_ylabel("D [m$^2$/s]")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"Transport coefficients used ({transport_note})", fontsize=9)

    for Z in range(1, 6):
        ax[1].plot(result.rho, result.V_m_s[Z], color=CHARGE_COLORS[Z],
                   label=f"B{Z}+")
    ax[1].axhline(0, color="k", lw=0.8, ls=":")
    ax[1].set_ylabel("pinch velocity [m/s]  (V < 0 = inward)")
    ax[1].set_xlabel("r/a")
    ax[1].grid(alpha=0.3)
    ax[1].legend(ncol=3, fontsize=8,
                 title="assumed pinch — not from Er measurement" if not is_boron_specific else None)

    fig.tight_layout()
    fig.savefig(out_dir / "03_transport_coefficients_used.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 04 — Damköhler / timescale map
# ---------------------------------------------------------------------------
def plot_damkohler(profiles, result, out_dir, source_width):
    """
    log10(tau_D / tau_iz) for B0 as a function of (rho, D).
    L = source deposition width — a fixed global scale, not a local gradient length.
    """
    rho    = profiles.rho
    D_grid = np.logspace(-4, 0.5, 180)
    L      = max(source_width * profiles.minor_radius_m, 1e-4)
    tau_iz = 1.0 / np.maximum(result.ionization_s[:, 0], 1e-40)

    logDa = np.zeros((len(D_grid), len(rho)))
    for i, D in enumerate(D_grid):
        logDa[i, :] = np.log10(np.maximum((L**2 / D) / tau_iz, 1e-40))

    fig, ax = plt.subplots(figsize=(8.0, 5.4), dpi=150)
    pcm = ax.pcolormesh(rho, D_grid, logDa, shading="auto", cmap="coolwarm",
                        vmin=-3, vmax=3)
    cbar = plt.colorbar(pcm, ax=ax)
    cbar.set_label(r"$\log_{10}(\tau_D/\tau_\mathrm{iz})$ for B0")
    ax.contour(rho, D_grid, logDa, levels=[0], colors="k", linewidths=1.5)
    ax.semilogy(rho, result.D_m2_s[0], "w-", lw=2.4, label="D used in this run")
    ax.set_yscale("log")
    ax.set_xlabel("r/a")
    ax.set_ylabel("D [m$^2$/s]")
    L_mm = L * 1e3
    ax.set_title(
        f"B0: ionization vs diffusion timescale\n"
        f"L = source deposition width = {L_mm:.1f} mm (fixed global scale)",
        fontsize=9,
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "04_damkohler_timescale_map.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 05 — charge-state profiles + coronal equilibrium
# ---------------------------------------------------------------------------
def plot_charge_states(result, ionization_s, recombination_s, out_dir, case_label):
    coronal_f = coronal_equilibrium_fractions(ionization_s, recombination_s)

    final  = result.nZ[-1]
    maxval = np.max(final) if np.max(final) > 0 else 1.0

    # Fractional populations at rho ~ 0.5.
    i_half    = np.argmin(np.abs(result.rho - 0.5))
    n_half    = final[:, i_half]
    total_h   = n_half.sum()
    fracs_sim = n_half / total_h if total_h > 0 else np.zeros(6)

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 5.3), dpi=150)

    # Left: normalized shapes with fraction annotation.
    for Z in range(6):
        ax[0].plot(result.rho, final[Z] / maxval, color=CHARGE_COLORS[Z],
                   label=CHARGE_LABELS[Z])
    frac_str = "\n".join([f"{CHARGE_LABELS[z]}: {100*fracs_sim[z]:.1f}%"
                          for z in range(6)])
    ax[0].text(0.97, 0.97, f"Fractions at r/a=0.5:\n{frac_str}",
               transform=ax[0].transAxes, fontsize=7, va="top", ha="right",
               bbox=dict(facecolor="white", edgecolor="0.5", alpha=0.88))
    ax[0].set_xlabel("r/a")
    ax[0].set_ylabel("density (normalized to overall max)")
    ax[0].set_title(f"Charge-state profiles at t={result.t[-1]:.3f} s\n"
                    f"Normalized — shape only (see inset for fractions at r/a=0.5)",
                    fontsize=9)
    ax[0].legend(ncol=2, fontsize=8)
    ax[0].grid(alpha=0.3)

    # Right: fractional populations vs coronal equilibrium.
    import matplotlib.patches as mpatches
    for Z in range(6):
        n_tot_r = final.sum(axis=0)
        f_sim   = np.divide(final[Z], n_tot_r, out=np.zeros_like(final[Z]),
                            where=n_tot_r > 0)
        ax[1].plot(result.rho, f_sim,      color=CHARGE_COLORS[Z], lw=1.8,
                   label=CHARGE_LABELS[Z])
        ax[1].plot(result.rho, coronal_f[:, Z], color=CHARGE_COLORS[Z],
                   lw=1.0, ls="--", alpha=0.7)
    ax[1].set_xlabel("r/a")
    ax[1].set_ylabel("charge-state fraction  (sum = 1)")
    ax[1].set_title("Fractional populations vs coronal equilibrium\n"
                    "(solid = simulation, dashed = coronal equilibrium)",
                    fontsize=9)
    ax[1].grid(alpha=0.3)
    solid_p = mpatches.Patch(color="grey", label="simulation (solid)")
    dash_p  = mpatches.Patch(color="grey", fill=False, label="coronal equil. (dashed)")
    ax[1].legend(handles=[solid_p, dash_p], fontsize=7.5, loc="upper left")

    fig.suptitle(case_label, fontsize=8, y=0.01)
    fig.tight_layout()
    fig.savefig(out_dir / "05_charge_states_final.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 06 — core fraction and peaking factor
# ---------------------------------------------------------------------------
def plot_core_fraction(result, out_dir, case_label):
    frac = core_fraction(result, rho_cut=0.5)
    pkf  = peaking_factor(result)

    fig, ax = plt.subplots(2, 1, figsize=(7.5, 6.5), dpi=150, sharex=True)

    ax[0].plot(result.t, frac, color="C0")
    ax[0].set_ylabel("fraction inside r/a < 0.5")
    ax[0].set_title(f"Boron core penetration metrics\n({case_label})", fontsize=9)
    ax[0].grid(alpha=0.3)

    ax[1].plot(result.t, pkf, color="C3")
    ax[1].axhline(1.0, color="k", lw=0.8, ls=":")
    ax[1].set_ylabel("peaking factor  n(0) / <n>$_\\mathrm{vol}$")
    ax[1].set_xlabel("time [s]")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "06_core_fraction_vs_time.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 07 — summary
# ---------------------------------------------------------------------------
def plot_summary(profiles, result, out_dir, case_label):
    frac  = core_fraction(result, rho_cut=0.5)
    final = result.nZ[-1]
    total = np.sum(final, axis=0)
    total_norm = total / (np.max(total) if np.max(total) > 0 else 1)

    n_B5  = final[5]
    B5_frac_core = (n_B5[result.rho < 0.5].mean() /
                    np.maximum(total[result.rho < 0.5].mean(), 1e-40))

    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5), dpi=150)

    ax[0, 0].plot(profiles.rho, profiles.Te_eV, label="Te (Thomson, measured)")
    ax[0, 0].plot(profiles.rho, profiles.Ti_eV, ls="--",
                  label=f"Ti = {profiles.Ti_eV[0]:.0f} eV (assumed)")
    ax[0, 0].set_ylabel("temperature [eV]")
    ax[0, 0].set_title("Profiles")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3)

    D_case = result.metadata.get("assumed_or_scanned_inputs", {}).get("D_case", "?")
    ax[0, 1].semilogy(result.rho, result.D_m2_s[0], color="C0")
    ax[0, 1].set_ylabel("D [m$^2$/s]")
    ax[0, 1].set_title(f"Assumed D — {D_case} case (not boron-specific)")
    ax[0, 1].grid(alpha=0.3, which="both")

    ax[1, 0].plot(result.rho, total_norm)
    ax[1, 0].set_xlabel("r/a")
    ax[1, 0].set_ylabel("normalized total B")
    ax[1, 0].set_title("Final total boron profile")
    ax[1, 0].grid(alpha=0.3)

    ax[1, 1].plot(result.t, frac)
    ax[1, 1].set_xlabel("time [s]")
    ax[1, 1].set_ylabel("fraction inside r/a < 0.5")
    ax[1, 1].set_title(f"Core fraction  (B5+ core frac \u2248 {B5_frac_core:.2f})")
    ax[1, 1].grid(alpha=0.3)

    for a in ax.ravel():
        a.set_xlim(left=0)

    fig.suptitle(
        f"HSX boron sensitivity study — {case_label}\n"
        "Transport assumed/not boron-specific",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "07_summary_single_run.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    print("=" * 72)
    print("HSX boron 1D radial transport model")
    print("=" * 72)
    print("WARNING: Transport is not boron-specific unless a future boron")
    print("         transport CSV is supplied via --future-boron-transport-csv.")
    print("         Default D, Ti, and pinch are assumed/scanned inputs.")
    print(f"WARNING: Source rate = {args.source_rate:.0e} s^-1 is an order-of-magnitude")
    print("         placeholder — replace with a calibrated dropper or spectroscopy estimate.")
    print("=" * 72)

    profiles = load_profiles(DATA_DIR, n_points=args.n_points, ti_case_eV=args.ti)
    ionization_s, recombination_s, rates_meta = load_boron_adas_rates(profiles, DATA_DIR)
    transport_inputs = build_transport_arrays(
        profiles.rho,
        D_case=args.D_case,
        pinch_case=args.pinch,
        future_boron_transport_csv=args.future_boron_transport_csv,
    )

    source = make_edge_neutral_source(
        profiles.rho,
        total_particles_s=args.source_rate,
        minor_radius_m=profiles.minor_radius_m,
        major_radius_m=profiles.major_radius_m,
        width=args.source_width,
        center=args.source_center,
    )

    metadata = make_metadata(args, profiles, transport_inputs, rates_meta)

    result = solve_boron_transport(
        profiles.rho,
        profiles.r_m,
        transport_inputs.D_by_Z_m2_s,
        transport_inputs.V_by_Z_m_s,
        ionization_s,
        recombination_s,
        source,
        t_end=args.t_end,
        n_save=250,
        metadata=metadata,
    )

    run_tag = f"Ti{int(args.ti)}_{args.D_case}_{args.pinch}"
    run_dir = OUT_DIR / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    np.savez(
        run_dir / "boron_transport_result.npz",
        t=result.t,
        rho=result.rho,
        r_m=result.r_m,
        nZ=result.nZ,
        D_m2_s=result.D_m2_s,
        V_m_s=result.V_m_s,
        Te_eV=profiles.Te_eV,
        ne_m3=profiles.ne_m3,
        Ti_eV=profiles.Ti_eV,
        ionization_s=result.ionization_s,
        recombination_s=result.recombination_s,
        source_m3_s=result.source_m3_s,
    )

    case_label = make_case_label(args)
    is_bs = transport_inputs.is_boron_specific

    plot_profiles(profiles, run_dir)
    plot_rates(profiles, ionization_s, recombination_s, run_dir)
    plot_transport(result, run_dir, is_boron_specific=is_bs)
    plot_damkohler(profiles, result, run_dir, args.source_width)
    plot_charge_states(result, ionization_s, recombination_s, run_dir, case_label)
    plot_core_fraction(result, run_dir, case_label)
    plot_summary(profiles, result, run_dir, case_label)

    frac_final = core_fraction(result)[-1]
    pkf_final  = peaking_factor(result)[-1]
    print(f"Saved outputs to: {run_dir}")
    print(f"Final core fraction (r/a < 0.5): {frac_final:.3e}")
    print(f"Final peaking factor n(0)/<n>:   {pkf_final:.2f}")
    print("Done.")


if __name__ == "__main__":
    main()