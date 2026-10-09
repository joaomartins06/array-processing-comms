import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from doa.covariance import sample_cov
from doa.estimators import bartlett, esprit, music, mvdr
from doa.metrics import resolution_probability, rmse
from doa.model import simulate
from doa.order import estimate_order
from doa.peaks import pick_peaks

#Experiment: two sources with unequal powers. The strong source is kept at a fixed SNR and the
#weak one is made dP dB weaker. The provided datasets always have equal powers and
#monte_carlo.py also uses equal powers, so this is new ground.

FIG = Path(__file__).resolve().parents[1] / "figures"

M, DELTA, N = 5, 0.5, 20  # same array and number of snapshots as the datasets
GRID = np.arange(-90, 90.0001, 0.05)
TRIALS = 500
LOADING = 1e-3

STRONG_SNR = 30                 # SNR of the strong source (dB), fixed
DPS = list(range(0, 31, 5))     # power difference dP (dB): weak source is dP dB below the strong one
PAIRS = {"10 deg apart": [20.0, 30.0], "50 deg apart": [-20.0, 30.0]}  # same angles as the datasets
WEAK = 1                        # index of the weak source (the one at 30 deg)

#peaks.py ignores peaks more than 10 dB below the highest one (min_rel_db=-10). Here the weak
#source is MORE than 10 dB below on purpose, so that rule would hide real peaks. Bartlett and
#MVDR therefore call pick_peaks with a relaxed threshold; peaks.py itself is not changed.
REL_DB = -40

#metrics.py by default accepts an estimate within half the separation (25 deg for the 50 deg pair).
#With the relaxed threshold Bartlett can pick a sidelobe 10-15 deg away from the weak source, which
#that criterion would count as a success. A source therefore only counts as found within TOL degrees
#(passed as tol to metrics.py). For the 10 deg pair this equals the default half-separation criterion.
TOL = 5.0


def bartlett_relaxed(X, d):
    angles, _ = pick_peaks(bartlett.spectrum(sample_cov(X), GRID, DELTA), GRID, d, min_rel_db=REL_DB)
    return angles


def mvdr_relaxed(X, d):
    angles, _ = pick_peaks(mvdr.spectrum(sample_cov(X), GRID, DELTA, loading=LOADING), GRID, d,
                           min_rel_db=REL_DB)
    return angles


# method name -> estimate(X, d). MUSIC already uses no height threshold, ESPRIT has no peaks
METHODS = {
    "Bartlett": bartlett_relaxed,
    "MVDR": mvdr_relaxed,
    "MUSIC": lambda X, d: music.estimate(X, d, GRID, DELTA),
    "ESPRIT": lambda X, d: esprit.estimate(X, d, GRID, DELTA),
}

COLORS = {"Bartlett": "#0072B2", "MVDR": "#E69F00", "MUSIC": "#009E73", "ESPRIT": "#CC79A7"}
MARKERS = {"Bartlett": "o", "MVDR": "s", "MUSIC": "^", "ESPRIT": "D"}


def run_powers(thetas, dp, seed, trials=TRIALS):
    #Monte Carlo point: P_found and RMSE per method, plus how often MDL finds d = 2
    rng = np.random.default_rng(seed)
    snrs = np.full(len(thetas), float(STRONG_SNR))
    snrs[WEAK] -= dp  # weaken one source by dp dB
    ests = {name: [] for name in METHODS}
    mdl_ok = 0
    for _ in range(trials):
        X, *_ = simulate(thetas, M, DELTA, N, snrs, rng)  # one SNR per source
        for name, estimate in METHODS.items():
            ests[name].append(estimate(X, len(thetas)))
        mdl_ok += estimate_order(sample_cov(X), N) == len(thetas)
    #found = right number of angles, each within TOL degrees of the truth
    res = {name: (rmse(e, thetas, TOL), resolution_probability(e, thetas, TOL)) for name, e in ests.items()}
    res["MDL"] = mdl_ok / trials
    return res


def table(title, xs, results):
    #prints RMSE and P_found per method, plus the MDL success rate
    print(f"\n{title}")
    print(f"{'dP (dB)':>10s} | " + " | ".join(f"{n:^19s}" for n in METHODS) + " |  MDL ok")
    print(f"{'':>10s} | " + " | ".join(f"{'RMSE':>9s} {'P_found':>9s}" for _ in METHODS) + " |")
    for x, res in zip(xs, results):
        cells = " | ".join(f"{res[n][0]:9.4f} {res[n][1]:9.3f}" for n in METHODS)
        print(f"{x:>10} | {cells} | {res['MDL']:7.3f}")


def exp_powers():
    results = {}
    for k, (label, thetas) in enumerate(PAIRS.items()):
        results[label] = [run_powers(thetas, dp, 6000 + 100 * k + i) for i, dp in enumerate(DPS)]
        table(f"Unequal powers, {label} {thetas}: strong {STRONG_SNR} dB, weak {STRONG_SNR} - dP dB, "
              f"N={N}, {TRIALS} trials, found = within {TOL:.0f} deg", DPS, results[label])

    #figure 1: P_found (top) and RMSE (bottom) versus power difference, one column per pair
    fig, axes = plt.subplots(2, len(PAIRS), figsize=(11, 7), sharex=True)
    for col, label in enumerate(PAIRS):
        for name in METHODS:
            axes[0, col].plot(DPS, [r[name][1] for r in results[label]], "-", marker=MARKERS[name],
                              color=COLORS[name], ms=4, label=name)
            axes[1, col].semilogy(DPS, [r[name][0] for r in results[label]], "-", marker=MARKERS[name],
                                  color=COLORS[name], ms=4, label=name)
        axes[0, col].plot(DPS, [r["MDL"] for r in results[label]], "k:", label="MDL finds d=2")
        axes[0, col].set(title=f"{label} {PAIRS[label]}", ylim=(-0.03, 1.03))
        axes[1, col].set_xlabel("power difference dP (dB), weak source below the strong one")
        for ax in axes[:, col]:
            ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel(f"P(both found within {TOL:.0f} deg)")
    axes[1, 0].set_ylabel("RMSE (deg), found trials")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Unequal source powers (strong source {STRONG_SNR} dB, N={N}, {TRIALS} trials)")
    fig.tight_layout()
    fig.savefig(FIG / "exp_powers_mc.png", dpi=200)
    plt.close(fig)

    #figure 2: one clean realisation (200 snapshots) for the 50 deg pair at dP = 0, 15, 25 dB.
    #Shows the weak source disappearing under Bartlett's sidelobes, while MVDR keeps a real peak
    #that lies below the default -10 dB rule (red line)
    thetas = PAIRS["50 deg apart"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, dp in zip(axes, [0, 15, 25]):
        snrs = np.array([STRONG_SNR, STRONG_SNR - dp], dtype=float)
        X, *_ = simulate(thetas, M, DELTA, 200, snrs, np.random.default_rng(1))
        R = sample_cov(X)
        spectra = {
            "Bartlett": bartlett.spectrum(R, GRID, DELTA),
            "MVDR": mvdr.spectrum(R, GRID, DELTA, loading=LOADING),
            "MUSIC": music.spectrum(R, GRID, 2, DELTA),
        }
        for name, P in spectra.items():
            ax.plot(GRID, 10 * np.log10(P / P.max()), color=COLORS[name], label=name)
        for i, a in enumerate(esprit.estimate(X, 2, GRID, DELTA)):
            ax.axvline(a, color=COLORS["ESPRIT"], ls=":", lw=1.5, label="ESPRIT" if i == 0 else None)
        for i, t in enumerate(thetas):
            ax.axvline(t, color="k", lw=0.8, alpha=0.5, label="true" if i == 0 else None)
        ax.axhline(-10, color="r", lw=0.8, ls="--", label="default -10 dB peak rule" if dp == 0 else None)
        ax.set(title=f"weak source {dp} dB below the strong one", xlabel="angle (deg)",
               ylim=(-50, 2), xlim=(-90, 90))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pseudo-spectrum (dB, normalised to peak)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Sources at {thetas[0]:.0f} and {thetas[1]:.0f} deg, one realisation, 200 snapshots")
    fig.tight_layout()
    fig.savefig(FIG / "exp_powers_spectra.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for exp in (exp_powers,):
        t0 = time.time()
        exp()
        print(f"  ({exp.__name__}: {time.time() - t0:.0f} s)")
    print(f"\nFigures saved to {FIG}")
