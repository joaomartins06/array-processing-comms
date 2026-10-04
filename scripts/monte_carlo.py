import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from doa.estimators import bartlett, music, mvdr
from doa.metrics import resolution_probability, rmse
from doa.model import simulate

FIG = Path(__file__).resolve().parents[1] / "figures"

M, DELTA, D = 5, 0.5, 2  # same array as the datasets
GRID = np.arange(-90, 90.0001, 0.05)
TRIALS = 500
LOADING = 1e-3                                  
THETAS_10 = [20.0, 30.0]                        
THETAS_50 = [-20.0, 30.0]                     

# method name -> estimate(X).
METHODS = {
    "Bartlett": lambda X: bartlett.estimate(X, D, GRID, DELTA),
    "MVDR": lambda X: mvdr.estimate(X, D, GRID, DELTA, loading=LOADING),
    "MUSIC": lambda X: music.estimate(X, D, GRID, DELTA),
}

COLORS = {"Bartlett": "#0072B2", "MVDR": "#E69F00", "MUSIC": "#009E73"}
MARKERS = {"Bartlett": "o", "MVDR": "s", "MUSIC": "^"}


def run_config(thetas, N, snr_db, seed, corr=0.0, trials=TRIALS, methods=METHODS):
    rng = np.random.default_rng(seed)
    ests = {name: [] for name in methods}

    for _ in range(trials):
        #generate a simulation 
        X, *_ = simulate(thetas, M, DELTA, N, snr_db, rng, corr=corr)
        for name, estimate in methods.items():
            #estimate the DOAs using the current method
            ests[name].append(estimate(X))
    #compute the RMSE and resolution probability for each method
    return {name: (rmse(e, thetas), resolution_probability(e, thetas)) for name, e in ests.items()}


def table(title, xlabel, xs, results):
    #generates tables for the metrics and each estimator
    print(f"\n{title}")
    print(f"{xlabel:>10s} | " + " | ".join(f"{n:^19s}" for n in METHODS))
    print(f"{'':>10s} | " + " | ".join(f"{'RMSE':>9s} {'P_res':>9s}" for _ in METHODS))
    for x, res in zip(xs, results):
        cells = " | ".join(f"{res[n][0]:9.4f} {res[n][1]:9.3f}" for n in METHODS)
        print(f"{x:>10} | {cells}")


def draw(ax, xs, results, idx, ylabel, logy=False, logx=False):
    # One panel: a line per method of metric idx (0 = RMSE, 1 = resolution probability).
    for name in METHODS:
        ax.plot(xs, [r[name][idx] for r in results], "-", marker=MARKERS[name],
                color=COLORS[name], lw=1.5, ms=5, label=name)
    ax.set_ylabel(ylabel)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    else:
        ax.set_xticks(xs) if len(xs) <= 8 else None
    ax.grid(alpha=0.3, which="both")


def two_panel_figure(path, xlabel, xs, res_by_sep, title, logx=False):
    # 2x2 figure: RMSE (top) and resolution probability (bottom) for the 10 and 50 deg cases.
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for col, sep in enumerate((10, 50)):
        draw(axes[0, col], xs, res_by_sep[sep], 0, "RMSE (deg), resolved trials", logy=True, logx=logx)
        draw(axes[1, col], xs, res_by_sep[sep], 1, "resolution probability", logx=logx)
        axes[1, col].set_ylim(-0.03, 1.03)
        axes[1, col].set_xlabel(xlabel)
        axes[0, col].set_title(f"separation {sep} deg")
    axes[0, 0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


#experiment 1: RMSE vs SNR
def exp1_snr():
    snrs = list(range(0, 31, 5))
    res = {}
    for sep, thetas, base in ((10, THETAS_10, 1000), (50, THETAS_50, 1100)):
        res[sep] = [run_config(thetas, 20, s, base + i) for i, s in enumerate(snrs)]
        table(f"Experiment 1: SNR sweep, separation {sep} deg, N=20, {TRIALS} trials", "SNR (dB)", snrs, res[sep])
    two_panel_figure(FIG / "mc_snr.png", "SNR (dB)", snrs, res, f"RMSE and resolution versus SNR (N=20, {TRIALS} trials)")


#experiment 2: resolution vs separation
def exp2_separation():
    seps = [2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50]
    res = [run_config([-s / 2, s / 2], 20, 20, 2000 + i) for i, s in enumerate(seps)]
    table(f"Experiment 2: separation sweep centred on 0 deg, SNR 20 dB, N=20, {TRIALS} trials", "sep (deg)", seps, res)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    draw(axes[0], seps, res, 1, "resolution probability")
    axes[0].set_ylim(-0.03, 1.03)
    draw(axes[1], seps, res, 0, "RMSE (deg), resolved trials", logy=True)
    for ax in axes:
        ax.set_xlabel("angle separation (deg)")
        ax.set_xticks([2, 5, 10, 15, 20, 25, 30, 40, 50])
    axes[0].legend()
    fig.suptitle(f"Resolution versus separation (SNR 20 dB, N=20, {TRIALS} trials)")
    fig.tight_layout()
    fig.savefig(FIG / "mc_separation.png", dpi=200)
    plt.close(fig)

#experiment 3: RMSE vs snapshots
def exp3_snapshots():
    Ns = [10, 20, 50, 100, 200, 500]
    res = {}
    for sep, thetas, base in ((10, THETAS_10, 3000), (50, THETAS_50, 3100)):
        res[sep] = [run_config(thetas, n, 20, base + i) for i, n in enumerate(Ns)]
        table(f"Experiment 3: N sweep, separation {sep} deg, SNR 20 dB, {TRIALS} trials", "N", Ns, res[sep])
    two_panel_figure(FIG / "mc_snapshots.png", "snapshots N", Ns, res,
                     f"RMSE and resolution versus snapshots (SNR 20 dB, {TRIALS} trials)", logx=True)

#experiment 4: resolution vs source coherence
def exp4_coherence():
    corrs = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.0]
    res = [run_config(THETAS_50, 100, 20, 4000 + i, corr=c) for i, c in enumerate(corrs)]
    table(f"Experiment 4: source correlation, separation 50 deg, SNR 20 dB, N=100, {TRIALS} trials", "corr", corrs, res)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    draw(axes[0], corrs, res, 1, "resolution probability")
    axes[0].set_ylim(-0.03, 1.03)
    draw(axes[1], corrs, res, 0, "RMSE (deg), resolved trials", logy=True)
    for ax in axes:
        ax.set_xlabel("source correlation coefficient")
        ax.set_xticks([0.0, 0.5, 0.8, 0.9, 1.0])
    axes[0].legend()
    fig.suptitle(f"Coherent sources (50 deg apart, SNR 20 dB, N=100, {TRIALS} trials)")
    fig.tight_layout()
    fig.savefig(FIG / "mc_coherence.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    for exp in (exp1_snr, exp2_separation, exp3_snapshots, exp4_coherence):
        t0 = time.time()
        exp()
        print(f"  ({exp.__name__}: {time.time() - t0:.0f} s)")
    print(f"\nFigures saved to {FIG}")
