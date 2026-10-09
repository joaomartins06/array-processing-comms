import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from doa.covariance import sample_cov
from doa.estimators import bartlett, esprit, music, mvdr
from doa.metrics import resolution_probability, rmse
from doa.order import estimate_order

#Experiment: the real datasets with increasing white noise added on top. The datasets are
#essentially noiseless, so the 20 snapshots (the sources' symbols) are kept fixed and only the
#noise is redrawn in each trial. This complements monte_carlo.py, where the symbols are
#also redrawn, and shows how the actual datasets behave in a more realistic setting.

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"

D = 2  # number of sources
GRID = np.arange(-90, 90.0001, 0.05)
TRIALS = 500
LOADING = 1e-3
TOL = 5.0                             
SNRS = list(range(-5, 41, 5))  # SNR per source (dB) of the added noise
SPECTRA_SNRS = (30, 15, 5)  # SNRs shown in the single-realisation figure
DATASETS = ("10sep", "50sep")

COLORS = {"Bartlett": "#0072B2", "MVDR": "#E69F00", "MUSIC": "#009E73", "ESPRIT": "#CC79A7"}
MARKERS = {"Bartlett": "o", "MVDR": "s", "MUSIC": "^", "ESPRIT": "D"}


def load(name):
    #get X, number of sensors (M), number of snapshots (N) and Delta of a dataset
    m = loadmat(ROOT / "data" / f"spcom_{name}.mat")
    return m["X"], int(m["M"].item()), int(m["N"].item()), float(m["Delta"].item())


def make_methods(delta):
    # method name -> estimate(X). Bartlett and MVDR keep the default pick_peaks thresholds
    # because both sources have the same power here
    return {
        "Bartlett": lambda X: bartlett.estimate(X, D, GRID, delta),
        "MVDR": lambda X: mvdr.estimate(X, D, GRID, delta, loading=LOADING),
        "MUSIC": lambda X: music.estimate(X, D, GRID, delta),
        "ESPRIT": lambda X: esprit.estimate(X, D, GRID, delta),
    }


def add_noise(X, snr_db, p_hat, rng):
    #adds white Gaussian noise with variance p_hat * 10^(-snr/10) per entry,
    #so that the SNR per source is snr_db (p_hat = mean power per source)
    sigma2 = p_hat * 10 ** (-snr_db / 10)
    #we sample a real and an imaginary component, due to the dataset also being in the complex domain
    return X + np.sqrt(sigma2 / 2) * (rng.standard_normal(X.shape) + 1j * rng.standard_normal(X.shape))


def run_noise(X, delta, ref, p_hat, snr_db, seed, trials=TRIALS):
    #Monte Carlo point: P_found and RMSE per method, plus how often MDL finds d
    methods = make_methods(delta)
    rng = np.random.default_rng(seed)
    N = X.shape[1]
    ests = {name: [] for name in methods}
    mdl_ok = 0
    for _ in range(trials):
        #same noisy data for every method
        Xn = add_noise(X, snr_db, p_hat, rng)  
        #run each method on the noisy data
        for name, estimate in methods.items():
            ests[name].append(estimate(Xn))
        mdl_ok += estimate_order(sample_cov(Xn), N) == D
    res = {name: (rmse(e, ref, TOL), resolution_probability(e, ref, TOL)) for name, e in ests.items()}
    res["MDL"] = mdl_ok / trials
    return res


def table(title, xs, results):
    #prints RMSE and P_found per method, plus the MDL success rate
    names = ("Bartlett", "MVDR", "MUSIC", "ESPRIT")
    print(f"\n{title}")
    print(f"{'SNR (dB)':>10s} | " + " | ".join(f"{n:^19s}" for n in names) + " |  MDL ok")
    print(f"{'':>10s} | " + " | ".join(f"{'RMSE':>9s} {'P_found':>9s}" for _ in names) + " |")
    for x, res in zip(xs, results):
        cells = " | ".join(f"{res[n][0]:9.4f} {res[n][1]:9.3f}" for n in names)
        print(f"{x:>10} | {cells} | {res['MDL']:7.3f}")


def exp_noise():
    data, results = {}, {}
    for k, name in enumerate(DATASETS):
        X, M, N, delta = load(name)
        R = sample_cov(X)
        # estimates the mean power per source of the clean data
        p_hat = np.trace(R).real / (M * D)           
        #as ESPRIT is considered the best estimator, we use it as reference
        ref = esprit.estimate(X, D, GRID, delta)       
        data[name] = (X, delta, ref, p_hat)
        print(f"\nDataset {name}: M={M}, N={N}, tr(R)/(M d) = {p_hat:.4f} (power per source), "
              f"reference angles (ESPRIT, clean) = {np.array2string(ref, precision=5)}")
        results[name] = [run_noise(X, delta, ref, p_hat, s, 7000 + 100 * k + i) for i, s in enumerate(SNRS)]
        table(f"Dataset {name} + white noise, N={N}, {TRIALS} trials, found = within {TOL:.0f} deg of "
              f"the reference", SNRS, results[name])

    #figure 1: P_found (top) and RMSE (bottom) versus the SNR of the added noise, one column per dataset
    names = ("Bartlett", "MVDR", "MUSIC", "ESPRIT")
    fig, axes = plt.subplots(2, len(DATASETS), figsize=(11, 7), sharex=True)
    for col, name in enumerate(DATASETS):
        for m in names:
            axes[0, col].plot(SNRS, [r[m][1] for r in results[name]], "-", marker=MARKERS[m],
                              color=COLORS[m], ms=4, label=m)
            axes[1, col].semilogy(SNRS, [r[m][0] for r in results[name]], "-", marker=MARKERS[m],
                                  color=COLORS[m], ms=4, label=m)
        axes[0, col].plot(SNRS, [r["MDL"] for r in results[name]], "k:", label="MDL finds d=2")
        axes[0, col].set(title=f"dataset {name} (reference {np.array2string(data[name][2], precision=1)})",
                         ylim=(-0.03, 1.03))
        axes[1, col].set_xlabel("SNR per source of the added noise (dB)")
        for ax in axes[:, col]:
            ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel(f"P(both found within {TOL:.0f} deg)")
    axes[1, 0].set_ylabel("RMSE (deg), found trials")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Real datasets plus white noise (N=20, {TRIALS} trials)")
    fig.tight_layout()
    fig.savefig(FIG / "exp_noise_mc.png", dpi=200)
    plt.close(fig)

    #figure 2: one noise realisation per panel, to see how the spectra degrade
    fig, axes = plt.subplots(len(DATASETS), len(SPECTRA_SNRS), figsize=(15, 7), sharey=True)
    for row, name in enumerate(DATASETS):
        X, delta, ref, p_hat = data[name]
        for col, snr in enumerate(SPECTRA_SNRS):
            ax = axes[row, col]
            Xn = add_noise(X, snr, p_hat, np.random.default_rng(100 * row + col))
            R = sample_cov(Xn)
            spectra = {
                "Bartlett": bartlett.spectrum(R, GRID, delta),
                "MVDR": mvdr.spectrum(R, GRID, delta, loading=LOADING),
                "MUSIC": music.spectrum(R, GRID, D, delta),
            }
            for m, P in spectra.items():
                ax.plot(GRID, 10 * np.log10(P / P.max()), color=COLORS[m], label=m)
            for i, a in enumerate(esprit.estimate(Xn, D, GRID, delta)):
                ax.axvline(a, color=COLORS["ESPRIT"], ls=":", lw=1.5, label="ESPRIT" if i == 0 else None)
            for i, t in enumerate(ref):
                ax.axvline(t, color="k", lw=0.8, alpha=0.5, label="reference" if i == 0 else None)
            ax.set(title=f"{name}, SNR {snr} dB", ylim=(-50, 2), xlim=(-90, 90))
            ax.grid(alpha=0.3)
            if row == len(DATASETS) - 1:
                ax.set_xlabel("angle (deg)")
        axes[row, 0].set_ylabel("pseudo-spectrum (dB, normalised to peak)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("One noise realisation per panel (datasets plus white noise)")
    fig.tight_layout()
    fig.savefig(FIG / "exp_noise_spectra.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for exp in (exp_noise,):
        t0 = time.time()
        exp()
        print(f"  ({exp.__name__}: {time.time() - t0:.0f} s)")
    print(f"\nFigures saved to {FIG}")
