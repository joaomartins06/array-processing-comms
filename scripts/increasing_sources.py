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

#Extra experiments that move away from the ideal case of the provided datasets
#(no noise, two equal-power uncorrelated sources). They complement monte_carlo.py,
#which always uses two equal-power sources.

FIG = Path(__file__).resolve().parents[1] / "figures"

M, DELTA, N = 5, 0.5, 20  # same array and number of snapshots as the datasets
GRID = np.arange(-90, 90.0001, 0.05)
TRIALS = 500
LOADING = 1e-3

# method name -> estimate(X, d). d is an input here because it changes per experiment
METHODS = {
    "Bartlett": lambda X, d: bartlett.estimate(X, d, GRID, DELTA),
    "MVDR": lambda X, d: mvdr.estimate(X, d, GRID, DELTA, loading=LOADING),
    "MUSIC": lambda X, d: music.estimate(X, d, GRID, DELTA),
    "ESPRIT": lambda X, d: esprit.estimate(X, d, GRID, DELTA),
}

COLORS = {"Bartlett": "#0072B2", "MVDR": "#E69F00", "MUSIC": "#009E73", "ESPRIT": "#CC79A7"}
MARKERS = {"Bartlett": "o", "MVDR": "s", "MUSIC": "^", "ESPRIT": "D"}


#experiment A: increasing the number of sources (tests the d < M assumption)
#angles are spread out so that resolution is not the limiting factor, and the d = 5 set is
#deliberately NOT evenly spaced in sin(theta): at d = M the ESPRIT output is evenly spaced
#in sin(theta) whatever the truth, so an evenly spaced test set would look like a false success
SOURCE_SETS = {
    2: [-35.0, 30.0],
    3: [-40.0, 0.0, 35.0],
    4: [-55.0, -20.0, 10.0, 40.0],
    5: [-60.0, -30.0, 5.0, 25.0, 70.0],
}
SNRS = list(range(-10, 41, 5))


def run_sources(thetas, snr_db, seed, trials=TRIALS):
    #Monte Carlo point: resolution probability and RMSE per method, plus how often MDL finds d
    rng = np.random.default_rng(seed)
    d = len(thetas)
    ests = {name: [] for name in METHODS}
    mdl_ok = 0
    for _ in range(trials):
        #new simulated dataset every trial: d unit-power QPSK sources + white noise
        X, *_ = simulate(thetas, M, DELTA, N, snr_db, rng)
        for name, estimate in METHODS.items():
            ests[name].append(estimate(X, d))
        #MDL can only return 0..M-1, so it can never find d = M
        mdl_ok += estimate_order(sample_cov(X), N) == d
    res = {name: (rmse(e, thetas), resolution_probability(e, thetas)) for name, e in ests.items()}
    res["MDL"] = mdl_ok / trials
    return res


def table(title, xlabel, xs, results):
    #prints RMSE and resolution probability per method, plus the MDL success rate
    print(f"\n{title}")
    print(f"{xlabel:>10s} | " + " | ".join(f"{n:^19s}" for n in METHODS) + " |  MDL ok")
    print(f"{'':>10s} | " + " | ".join(f"{'RMSE':>9s} {'P_res':>9s}" for _ in METHODS) + " |")
    for x, res in zip(xs, results):
        cells = " | ".join(f"{res[n][0]:9.4f} {res[n][1]:9.3f}" for n in METHODS)
        print(f"{x:>10} | {cells} | {res['MDL']:7.3f}")


def exp_sources():
    results = {}
    for d, thetas in SOURCE_SETS.items():
        results[d] = [run_sources(thetas, s, 5000 + 100 * d + i) for i, s in enumerate(SNRS)]
        table(f"Experiment A: d = {d} sources at {thetas}, N={N}, {TRIALS} trials",
              "SNR (dB)", SNRS, results[d])

    #figure 1: resolution probability (top) and RMSE (bottom) versus SNR, one column per d
    fig, axes = plt.subplots(2, len(SOURCE_SETS), figsize=(15, 7), sharex=True)
    for col, d in enumerate(SOURCE_SETS):
        for name in METHODS:
            axes[0, col].plot(SNRS, [r[name][1] for r in results[d]], "-", marker=MARKERS[name],
                              color=COLORS[name], ms=4, label=name)
            axes[1, col].semilogy(SNRS, [r[name][0] for r in results[d]], "-", marker=MARKERS[name],
                                  color=COLORS[name], ms=4, label=name)
        axes[0, col].plot(SNRS, [r["MDL"] for r in results[d]], "k:", label="MDL finds d")
        axes[0, col].set(title=f"d = {d} sources" + ("  (d = M)" if d == M else ""), ylim=(-0.03, 1.03))
        axes[1, col].set_xlabel("SNR per source (dB)")
        if all(np.isnan(r[name][0]) for r in results[d] for name in METHODS):
            axes[1, col].text(0.5, 0.5, "no trial resolved\n(all methods fail)", ha="center",
                              va="center", transform=axes[1, col].transAxes)
        for ax in axes[:, col]:
            ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel("resolution probability")
    axes[1, 0].set_ylabel("RMSE (deg), resolved trials")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Increasing the number of sources (M={M}, N={N}, {TRIALS} trials)")
    fig.tight_layout()
    fig.savefig(FIG / "exp_sources_mc.png", dpi=200)
    plt.close(fig)

    #figure 2: one clean realisation per d (30 dB, 200 snapshots) to show WHY the methods fail
    #at d = M: MUSIC has no noise subspace (flat), Bartlett/MVDR have no distinct peaks,
    #ESPRIT returns d confident but wrong angles
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, d in zip(axes, [3, 4, 5]):
        thetas = SOURCE_SETS[d]
        X, *_ = simulate(thetas, M, DELTA, 200, 30, np.random.default_rng(0))
        R = sample_cov(X)
        spectra = {
            "Bartlett": bartlett.spectrum(R, GRID, DELTA),
            "MVDR": mvdr.spectrum(R, GRID, DELTA, loading=LOADING),
            "MUSIC": music.spectrum(R, GRID, d, DELTA),
        }
        for name, P in spectra.items():
            ax.plot(GRID, 10 * np.log10(P / P.max()), color=COLORS[name], label=name)
        for i, a in enumerate(esprit.estimate(X, d, GRID, DELTA)):
            ax.axvline(a, color=COLORS["ESPRIT"], ls=":", lw=1.5, label="ESPRIT" if i == 0 else None)
        for i, t in enumerate(thetas):
            ax.axvline(t, color="k", lw=0.8, alpha=0.5, label="true" if i == 0 else None)
        ax.set(title=f"d = {d}", xlabel="angle (deg)", ylim=(-50, 2), xlim=(-90, 90))
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("pseudo-spectrum (dB, normalised to peak)")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("One realisation per d (SNR 30 dB per source, 200 snapshots)")
    fig.tight_layout()
    fig.savefig(FIG / "exp_sources_spectra.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for exp in (exp_sources,):
        t0 = time.time()
        exp()
        print(f"  ({exp.__name__}: {time.time() - t0:.0f} s)")
    print(f"\nFigures saved to {FIG}")
