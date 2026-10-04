import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")  #only saves the files, no display                    
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from doa.covariance import sample_cov
from doa.estimators import bartlett, music, mvdr
from doa.order import eigen_profile, estimate_order, mdl_aic

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

D = 2 #no of sources
#angles to sweep over in the grid
GRID = np.arange(-90, 90.0001, 0.05)
#MVDR loadings to try
LOADINGS = (1e-6, 1e-3, 1e-1)  
#the default MVDR loading value
DEFAULT_LOADING = 1e-3
COLORS = {"Bartlett": "#0072B2", "MVDR": "#E69F00", "MUSIC": "#009E73"}
STYLES = {"Bartlett": "-", "MVDR": "--", "MUSIC": "-."}


def to_db(P: np.ndarray) -> np.ndarray:
    #dB conversion, normalised to the peak
    return 10 * np.log10(P / P.max())


def mark(ax, angles: np.ndarray, db: np.ndarray, color: str) -> None:
    #mark estimated angles on the dB curve
    idx = np.searchsorted(GRID, angles - 1e-9)
    ax.plot(GRID[idx], db[idx], "o", color=color, ms=6, mfc="none", mew=1.5)


def fmt(angles: np.ndarray) -> str:
    #format angles as a string, with an UNRESOLVED flag if fewer than D were found
    s = "[" + ", ".join(f"{a:7.2f}" for a in angles) + "]"
    return s if len(angles) == D else f"{s}  UNRESOLVED (found {len(angles)} of {D})"


def load(name: str) -> tuple[np.ndarray, int, int, float]:
    #get datasets features and data
    m = loadmat(ROOT / "data" / f"spcom_{name}.mat")
    #get X, number of sensors (M), number of snapshots (N), and Delta (sensor spacing)
    return m["X"], int(m["M"].item()), int(m["N"].item()), float(m["Delta"].item())


fig_load, axes_load = plt.subplots(1, 2, figsize=(11, 4), sharey=True)


for col, name in enumerate(("10sep", "50sep")):
    X, M, N, Delta = load(name)
    #check that the loaded data has the expected shape
    assert X.shape == (M, N)
    #compute the sample covariance matrix
    R = sample_cov(X)

    print(f"\n{'=' * 70}\nDataset {name}: M={M}, N={N}, Delta={Delta}, X shape {X.shape}")

    #eigenvalue profile
    lam = eigen_profile(R)
    mdl, aic = mdl_aic(R, N)
    print("eigenvalues (descending):", np.array2string(lam, formatter={"float_kind": lambda v: f"{v:.3e}"}))
    print(f"condition number: {lam[0] / lam[-1]:.2e}")
    print("MDL(k), k=0..M-1:", np.array2string(mdl, formatter={"float_kind": lambda v: f"{v:.1f}"}))
    print("AIC(k), k=0..M-1:", np.array2string(aic, formatter={"float_kind": lambda v: f"{v:.1f}"}))
    #estimate the number of sources using MDL and AIC criteria
    d_mdl, d_aic = estimate_order(R, N, "mdl"), estimate_order(R, N, "aic")
    print(f"estimated order: MDL d={d_mdl}, AIC d={d_aic}  (using d={D})")
    #just in case they mismatch
    if d_mdl != D or d_aic != D:
        print("  WARNING: the information criteria do not agree with d = 2")

    # get the spectra
    P = {
        "Bartlett": bartlett.spectrum(R, GRID, Delta),
        "MVDR": mvdr.spectrum(R, GRID, Delta, loading=DEFAULT_LOADING),
        "MUSIC": music.spectrum(R, GRID, D, Delta),
    }
    #estimate angles using the different methods from the spectra
    est = {
        "Bartlett": bartlett.estimate(X, D, GRID, Delta),
        "MVDR": mvdr.estimate(X, D, GRID, Delta, loading=DEFAULT_LOADING),
        "MUSIC": music.estimate(X, D, GRID, Delta),
    }
    print(f"\nEstimated angles in degrees (grid step {GRID[1] - GRID[0]:.2f}):")
    for method, angles in est.items():
        print(f"  {method:9s} {fmt(angles)}")

    #plot the pseudo-spectra with the estimated angles marked
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for method in P:
        db = to_db(P[method])
        ax.plot(GRID, db, STYLES[method], color=COLORS[method], lw=1.5, label=method)
        mark(ax, est[method], db, COLORS[method])
    ax.set(xlabel="angle (deg)", ylabel="pseudo-spectrum (dB, normalised to peak)",
           title=f"Dataset {name}", xlim=(-90, 90), ylim=(-60, 3))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center", ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / f"spectra_{name}.png", dpi=200)
    plt.close(fig)

    #check the effect of the loading parameter on the MVDR estimates
    ax = axes_load[col]
    print("\nMVDR estimates versus diagonal loading (fraction of tr(R)/M):")
    for load_val in LOADINGS:
        Pm = mvdr.spectrum(R, GRID, Delta, loading=load_val)
        ang = mvdr.estimate(X, D, GRID, Delta, loading=load_val)
        print(f"  loading {load_val:.0e}  eps = {load_val * np.trace(R).real / M:.2e}  {fmt(ang)}")
        ax.plot(GRID, to_db(Pm), lw=1.5, label=f"loading {load_val:.0e}")
    ax.set(xlabel="angle (deg)", title=f"MVDR, dataset {name}", xlim=(-90, 90), ylim=(-60, 3))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower center")

axes_load[0].set_ylabel("pseudo-spectrum (dB, normalised to peak)")
fig_load.tight_layout()
fig_load.savefig(FIG / "mvdr_loading.png", dpi=200)

print(f"\nFigures saved to {FIG}")
