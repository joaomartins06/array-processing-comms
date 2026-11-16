import os

# Pin BLAS to a single thread, scoped to this script only: eigh/solve/lstsq can otherwise use
# multiple threads, which adds run-to-run jitter to the timing and obscures the algorithmic
# trends this benchmark is meant to show. Must be set before numpy is imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from doa.estimators import bartlett, esprit, music, mvdr
from doa.model import simulate
from doa.timing import summarize, time_calls

#Computing-efficiency benchmark: how long estimate() takes as a function of the size of the
#input data (array size M, snapshots N, grid resolution G, number of sources d). Complements
#monte_carlo.py (accuracy) and validate.py (correctness) with a performance view. The four
#methods have different asymptotic cost: Bartlett/MVDR/MUSIC scan a grid of G angles,
#MUSIC/ESPRIT both pay an eigendecomposition that Bartlett/MVDR don't, and ESPRIT alone has
#no grid dependence at all (it accepts and ignores grid_deg). Absolute timings are machine
#and load dependent; only the shape (slope, flatness) of each curve is the point.

FIG = Path(__file__).resolve().parents[1] / "figures"
RESULTS = Path(__file__).resolve().parents[1] / "results"

DELTA = 0.5
SNR_DB = 20.0
REPEAT, WARMUP = 7, 2

# baseline values held fixed in each sweep except for the swept axis
M0, N0, D0 = 8, 200, 2
GRID0 = np.arange(-90, 90.0001, 0.05)
M_D = 16  # array size used for the d sweep, chosen so every d below stays < M_D

# No try/except around the estimate() calls below: if these ranges are widened later (larger
# M, finer G, larger d) past what an estimator tolerates, the script crashes loudly on the
# first bad point instead of silently degrading one sweep -- intentional for a correctness-
# sensitive benchmark, but worth knowing before changing the ranges.
M_VALUES = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
N_VALUES = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
G_STEPS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]  # grid step in degrees -> G grows
D_VALUES = [1, 2, 3, 4, 6, 8, 10, 12, 14]

# method name -> estimate(X, d, grid). d and grid vary per sweep point, unlike the other
# scripts' METHODS dicts, so both are left as call arguments instead of closed over.
METHODS = {
    "Bartlett": lambda X, d, grid: bartlett.estimate(X, d, grid, DELTA),
    "MVDR": lambda X, d, grid: mvdr.estimate(X, d, grid, DELTA, loading=1e-3),
    "MUSIC": lambda X, d, grid: music.estimate(X, d, grid, DELTA),
    "ESPRIT": lambda X, d, grid: esprit.estimate(X, d, grid, DELTA),  # grid accepted, ignored
}

COLORS = {"Bartlett": "#0072B2", "MVDR": "#E69F00", "MUSIC": "#009E73", "ESPRIT": "#CC79A7"}
MARKERS = {"Bartlett": "o", "MVDR": "s", "MUSIC": "^", "ESPRIT": "D"}


def thetas_for(d):
    #d distinct angles, well separated so each method's peak-picking resolves cleanly. Cost is
    #effectively shape-driven (M, N, grid size, d), not data-driven, EXCEPT for one small
    #wrinkle: peaks.py's merge/dip-check loop can iterate a few extra times on a badly-resolved
    #spectrum. At this fixed high SNR with these well-spaced angles it exits immediately, so
    #the effect is negligible here but is not a universal guarantee for any data.
    return np.array([0.0]) if d == 1 else np.linspace(-80.0, 80.0, d)


def time_point(M, N, grid, d, seed):
    #one simulated dataset per point, reused across all repeats -- safe given the
    #near-shape-driven cost described in thetas_for() above. Returns {method: array of
    #`REPEAT` wall-clock seconds for estimate(X, d, grid)}.
    assert d < M, f"d={d} must be < M={M}"
    X, *_ = simulate(thetas_for(d), M, DELTA, N, SNR_DB, np.random.default_rng(seed))
    return {name: time_calls(lambda f=fn: f(X, d, grid), REPEAT, WARMUP) for name, fn in METHODS.items()}


def sweep_M(values=M_VALUES, seed0=10_000):
    raw = [time_point(M, N0, GRID0, D0, seed0 + i) for i, M in enumerate(values)]
    configs = [dict(M=M, N=N0, G=len(GRID0), d=D0) for M in values]
    return values, raw, configs


def sweep_N(values=N_VALUES, seed0=20_000):
    raw = [time_point(M0, N, GRID0, D0, seed0 + i) for i, N in enumerate(values)]
    configs = [dict(M=M0, N=N, G=len(GRID0), d=D0) for N in values]
    return values, raw, configs


def sweep_G(steps=G_STEPS, seed0=30_000):
    grids = [np.arange(-90, 90.0001, step) for step in steps]
    xs = [len(g) for g in grids]
    raw = [time_point(M0, N0, g, D0, seed0 + i) for i, g in enumerate(grids)]
    configs = [dict(M=M0, N=N0, G=G, d=D0) for G in xs]
    return xs, raw, configs


def sweep_D(values=D_VALUES, seed0=40_000):
    raw = [time_point(M_D, N0, GRID0, d, seed0 + i) for i, d in enumerate(values)]
    configs = [dict(M=M_D, N=N0, G=len(GRID0), d=d) for d in values]
    return values, raw, configs


def to_summary(raw):
    #raw: list of {method: times array}, one entry per sweep point -> list of {method: (median, min, mean)}
    return [{name: summarize(times) for name, times in point.items()} for point in raw]


def table(title, xlabel, xs, summary):
    print(f"\n{title}")
    print(f"{xlabel:>14s} | " + " | ".join(f"{n:^19s}" for n in METHODS))
    print(f"{'':>14s} | " + " | ".join(f"{'median ms':>9s} {'min ms':>9s}" for _ in METHODS))
    for x, s in zip(xs, summary):
        cells = " | ".join(f"{s[n][0] * 1e3:9.4f} {s[n][1] * 1e3:9.4f}" for n in METHODS)
        print(f"{x:>14} | {cells}")


def write_csv(path, sweep, xs, raw, configs):
    #long format, one row per timed repeat: sweep, x_value, method, M, N, G, d, repeat_idx, seconds
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "x_value", "method", "M", "N", "G", "d", "repeat_idx", "seconds"])
        for x, point, cfg in zip(xs, raw, configs):
            for name, times in point.items():
                for i, t in enumerate(times):
                    w.writerow([sweep, x, name, cfg["M"], cfg["N"], cfg["G"], cfg["d"], i, t])


def draw_panel(ax, xs, summary, logx=True):
    #one log-log line per method (median runtime, ms): a polynomial cost O(x^p) is a straight
    #line of slope p, which is what makes ESPRIT's grid-flatness and the M^2-vs-M^3 crossover
    #between methods visually legible.
    for name in METHODS:
        ys = [s[name][0] * 1e3 for s in summary]
        ax.plot(xs, ys, marker=MARKERS[name], color=COLORS[name], lw=1.5, ms=5, label=name)
    ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both")


def figure_single(path, xlabel, xs, summary, title):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    draw_panel(ax, xs, summary)
    ax.set(xlabel=xlabel, ylabel="estimate() runtime, median of 7 (ms)", title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_summary(path, sweeps):
    #sweeps: {key: (xlabel, xs, summary, title)}, same data as the four standalone figures
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (xlabel, xs, summary, title) in zip(axes.flat, sweeps.values()):
        draw_panel(ax, xs, summary)
        ax.set(xlabel=xlabel, title=title)
    axes[0, 0].set_ylabel("runtime (ms)")
    axes[1, 0].set_ylabel("runtime (ms)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("estimate() runtime vs data size (median of 7 repeats, log-log axes)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)

    sweeps = {}
    for key, fn, xlabel, title in (
        ("M", sweep_M, "array size M", f"vs array size (N={N0}, G={len(GRID0)}, d={D0})"),
        ("N", sweep_N, "snapshots N", f"vs snapshots (M={M0}, G={len(GRID0)}, d={D0})"),
        ("G", sweep_G, "grid points G", f"vs grid resolution (M={M0}, N={N0}, d={D0})"),
        ("d", sweep_D, "sources d", f"vs number of sources (M={M_D}, N={N0}, G={len(GRID0)})"),
    ):
        t0 = time.time()
        xs, raw, configs = fn()
        summary = to_summary(raw)
        table(f"Benchmark: runtime vs {xlabel}", xlabel, xs, summary)
        write_csv(RESULTS / f"timing_vs_{key}.csv", key, xs, raw, configs)
        figure_single(FIG / f"bench_vs_{key}.png", xlabel, xs, summary, title)
        sweeps[key] = (xlabel, xs, summary, title)
        print(f"  (sweep_{key}: {time.time() - t0:.0f} s)")

    figure_summary(FIG / "bench_summary.png", sweeps)
    print(f"\nFigures saved to {FIG}")
    print(f"Raw timing data saved to {RESULTS}")
