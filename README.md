# array-processing-comms

Direction-of-arrival (DOA) estimation for a uniform linear array (ULA): four classic estimators
(Bartlett, MVDR, MUSIC, ESPRIT) implemented from the same signal model, validated on synthetic
data and on two real measured datasets, with Monte Carlo accuracy studies and a computing-
efficiency benchmark.

## Setup

Python 3.10+, dependencies pinned in `requirements.txt` (numpy, scipy, matplotlib only):

```bash
pip install -r requirements.txt
```

There is no packaging. Every script under `scripts/` inserts `src/` onto `sys.path` itself, so
they are run directly from the repository root:

```bash
python scripts/<name>.py
```

## Repository layout

```
array-processing-comms/
├── data/                        # the two provided real datasets (not generated, not modified)
│   ├── spcom_10sep.mat
│   └── spcom_50sep.mat
├── src/doa/
│   ├── model.py                 # steering_matrix(), simulate() — the signal model and synthetic data generator
│   ├── covariance.py            # sample_cov()
│   ├── peaks.py                 # pick_peaks(), pick_minima() — peak-picking / resolution logic
│   ├── order.py                 # eigen_profile(), mdl_aic(), estimate_order() — MDL/AIC source-count selection
│   ├── metrics.py                # is_resolved(), resolution_probability(), rmse()
│   ├── timing.py                 # time_calls(), summarize() — wall-clock microbenchmark helpers
│   └── estimators/
│       ├── bartlett.py           # classical beamformer (periodogram)
│       ├── mvdr.py               # Capon / MVDR beamformer
│       ├── music.py              # MUSIC
│       └── esprit.py             # ESPRIT (closed form, no angle scan)
├── scripts/
│   ├── validate.py                # plain-assert correctness checks, replaces a pytest suite
│   ├── run_datasets.py            # spectra and estimates on the two real datasets
│   ├── monte_carlo.py             # RMSE / resolution probability vs SNR, separation, snapshots, coherence
│   ├── dataset_noise.py           # real datasets with synthetic noise added on top
│   ├── increasing_sources.py      # behaviour as the source count approaches d = M
│   ├── unequal_powers.py          # weak-source detectability with unequal source powers
│   └── benchmark.py                # computing-efficiency sweep vs array size, snapshots, grid, sources
├── figures/                       # generated plots (git-ignored)
├── results/                       # generated raw benchmark CSVs (git-ignored)
└── requirements.txt
```

## How to run

| Script | What it does | Output |
|---|---|---|
| `python scripts/validate.py` | Correctness checks for every module (signal model, peak-picking, MDL/AIC, all four estimators, metrics, the timing utility). Prints PASS/FAIL per check and exits non-zero on failure. | console only |
| `python scripts/run_datasets.py` | Loads both real datasets, prints eigenvalue profiles and MDL/AIC order estimates, plots the three scanning spectra plus ESPRIT's angles. | `figures/spectra_*.png`, `figures/mvdr_loading.png` |
| `python scripts/monte_carlo.py` | RMSE and resolution probability vs SNR, angle separation, snapshot count, and source coherence (500 trials/point). | `figures/mc_*.png` |
| `python scripts/dataset_noise.py` | Adds synthetic white noise on top of the (near-noiseless) real datasets. | `figures/*.png` |
| `python scripts/increasing_sources.py` | Pushes the source count d up to M, where the `d < M` assumption breaks down. | `figures/exp_sources_*.png` |
| `python scripts/unequal_powers.py` | Detectability of a weak source next to a strong one. | `figures/*.png` |
| `python scripts/benchmark.py` | Computing-efficiency sweep (see below). | `figures/bench_*.png`, `results/timing_vs_*.csv` |

Each script is self-contained and can be run independently and in any order (after `validate.py`
has confirmed the underlying code is correct).

## Data model

- `M` sensors, spacing `Delta` wavelengths, `N` snapshots, `d` sources (`d < M`).
- Steering vector `a(theta)[m] = exp(j 2*pi*m*Delta*sin(theta))`, `m = 0..M-1`.
- `X = A @ S + Noise`, shape `(M, N)` complex; `A` is `(M, d)`, built from the `d` steering vectors.
- Sample covariance `R = X @ X^H / N`.
- Both provided datasets: `M=5, N=20, d=2, Delta=0.5`, effectively noiseless (condition number
  ~1e11), which is why `run_datasets.py` needs MVDR diagonal loading or a pseudo-inverse, and why
  the Monte Carlo study on synthetic data (not the datasets) is what supplies accuracy and
  variance results.


