"""Plain-assert correctness checks. Run: python scripts/validate.py"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from doa.covariance import sample_cov
from doa.model import simulate, steering_matrix

results = []

def check(name):
    """Decorator: run the check, print PASS/FAIL, keep going so all failures are shown."""
    def wrap(fn):
        try:
            fn()
            print(f"PASS  {name}")
            results.append(True)
        except AssertionError as e:
            print(f"FAIL  {name}  {e}")
            results.append(False)
        return fn
    return wrap


M, Delta = 5, 0.5


@check("steering shapes: scalar -> (M,), array -> (M,K)")
def _():
    assert steering_matrix(10.0, M, Delta).shape == (M,)
    assert steering_matrix(np.array([10.0, 20.0, 30.0]), M, Delta).shape == (M, 3)
    assert steering_matrix(np.array([10.0]), M, Delta).shape == (M, 1)


@check("steering entries have unit modulus and ||a||^2 = M")
def _():
    a = steering_matrix(np.linspace(-90, 90, 181), M, Delta)
    assert np.allclose(np.abs(a), 1)
    assert np.allclose(np.sum(np.abs(a) ** 2, axis=0), M)


@check("a(t)^H a(t') equals the Dirichlet kernel")
def _():
    for t, tp in [(0, 10), (-20, 35), (5, 55), (10, 60), (-70, 40)]:
        delta = Delta * (np.sin(np.deg2rad(tp)) - np.sin(np.deg2rad(t)))
        lhs = steering_matrix(t, M, Delta).conj() @ steering_matrix(tp, M, Delta)
        rhs = np.exp(1j * np.pi * (M - 1) * delta) * np.sin(np.pi * M * delta) / np.sin(np.pi * delta)
        assert np.isclose(lhs, rhs), f"theta=({t},{tp}): {lhs} vs {rhs}"


@check("simulate: shapes and reproducibility with the same seed")
def _():
    X1, S1, A1, s2 = simulate([10, 50], M, Delta, 20, 10, np.random.default_rng(1))
    X2, *_ = simulate([10, 50], M, Delta, 20, 10, np.random.default_rng(1))
    assert X1.shape == (M, 20) and S1.shape == (2, 20) and A1.shape == (M, 2) and s2 == 1.0
    assert np.array_equal(X1, X2)


@check("sample covariance of a long simulation approaches A Rs A^H + sigma2 I")
def _():
    thetas, snr, corr, N = [10, 50], np.array([10.0, 3.0]), 0.5, 400_000
    X, S, A, sigma2 = simulate(thetas, M, Delta, N, snr, np.random.default_rng(0), corr=corr)
    amp = np.sqrt(10 ** (snr / 10))
    Rs = np.outer(amp, amp) * np.array([[1, corr], [corr, 1]])
    Rx = A @ Rs @ A.conj().T + sigma2 * np.eye(M)
    err = np.linalg.norm(sample_cov(X) - Rx) / np.linalg.norm(Rx)
    assert err < 0.01, f"relative error {err:.4f}"


@check("per-source SNR and correlation are realised in S; corr=1 is rank 1")
def _():
    N = 200_000
    _, S, _, _ = simulate([10, 50], M, Delta, N, [10.0, 0.0], np.random.default_rng(2), corr=0.8)
    Rs_hat = S @ S.conj().T / N
    assert np.allclose(np.real(np.diag(Rs_hat)), [10.0, 1.0], rtol=0.02)
    rho = Rs_hat[0, 1] / np.sqrt(Rs_hat[0, 0].real * Rs_hat[1, 1].real)
    assert abs(rho - 0.8) < 0.02, f"rho = {rho}"
    _, S, _, _ = simulate([10, 50], M, Delta, 1000, 5.0, np.random.default_rng(3), corr=1.0)
    assert np.linalg.matrix_rank(S) == 1


# ---------------- Phase 2: peaks and model order ----------------
from doa.order import eigen_profile, estimate_order, mdl_aic
from doa.peaks import pick_minima, pick_peaks

grid = np.arange(-90, 90.0001, 0.05)


def bump(center, height=1.0, sigma=2.0):
    return height * np.exp(-((grid - center) ** 2) / (2 * sigma ** 2))


@check("pick_peaks: two well separated peaks are both returned, sorted, ok=True")
def _():
    ang, ok = pick_peaks(bump(25) + bump(-20), grid, 2)
    assert ok and np.allclose(ang, [-20, 25], atol=0.06), f"{ang}, ok={ok}"


@check("pick_peaks: one merged peak -> ok=False, one angle, no padding")
def _():
    ang, ok = pick_peaks(bump(-1.5) + bump(1.5), grid, 2)
    assert not ok and ang.size == 1, f"{ang}, ok={ok}"


@check("pick_peaks: main lobe + sidelobe at -13 dB is rejected; accepted if threshold relaxed")
def _():
    s = bump(0) + bump(30, height=0.05)
    ang, ok = pick_peaks(s, grid, 2)
    assert not ok and ang.size == 1, f"{ang}, ok={ok}"
    ang, ok = pick_peaks(s, grid, 2, min_rel_db=-20)
    assert ok and ang.size == 2


@check("pick_peaks: a genuine weaker peak (-5 dB) is kept")
def _():
    ang, ok = pick_peaks(bump(0) + bump(30, height=0.3), grid, 2)
    assert ok and ang.size == 2


@check("pick_peaks: two maxima with a shallow dip (<3 dB) are not resolved")
def _():
    ang, ok = pick_peaks(bump(-2.5) + bump(2.5), grid, 2)   # two maxima, dip about 0.6 dB
    assert not ok and ang.size == 1, f"{ang}, ok={ok}"


@check("pick_minima: finds the two minima of a cost with two near-zero dips")
def _():
    J = ((grid + 20) * (grid - 25)) ** 2 / 1e6 + 1e-6
    ang, ok = pick_minima(J, grid, 2)
    assert ok and np.allclose(ang, [-20, 25], atol=0.06), f"{ang}, ok={ok}"


@check("eigen_profile is sorted descending and non-negative")
def _():
    X, *_ = simulate([10, 50], M, Delta, 50, 10, np.random.default_rng(4))
    lam = eigen_profile(sample_cov(X))
    assert np.all(np.diff(lam) <= 0) and lam[-1] > 0 and lam.size == M


@check("MDL and AIC recover d = 1, 2, 3 on simulations with a clear eigenvalue gap")
def _():
    for thetas in ([20], [-20, 30], [-40, 0, 40]):
        X, *_ = simulate(thetas, M, Delta, 200, 15, np.random.default_rng(5))
        R = sample_cov(X)
        for method in ("mdl", "aic"):
            d_hat = estimate_order(R, 200, method)
            assert d_hat == len(thetas), f"{method}: d_hat={d_hat}, d={len(thetas)}"
        mdl, aic = mdl_aic(R, 200)
        assert mdl.shape == aic.shape == (M,)



# ---------------- Phase 3: Bartlett ----------------
from doa.estimators import bartlett, mvdr


def halfpower_width(P):
    """Width of the main lobe at half power, in units of Delta*sin(theta)."""
    P = P / P.max()
    i = np.argmax(P)
    lo, hi = i, i
    while P[lo] >= 0.5:
        lo -= 1
    while P[hi] >= 0.5:
        hi += 1
    return Delta * (np.sin(np.deg2rad(grid[hi])) - np.sin(np.deg2rad(grid[lo])))


def sidelobe_db(P):
    """Highest sidelobe in dB below the main peak, or None if the spectrum has no sidelobe."""
    P = P / P.max()
    c = np.where((P[1:-1] > P[:-2]) & (P[1:-1] >= P[2:]))[0] + 1
    c = c[c != np.argmax(P)]
    return 10 * np.log10(P[c].max()) if c.size else None


@check("Bartlett: a single noiseless source peaks at the true angle")
def _():
    for t in (-30.0, 0.0, 20.0, 55.0):
        X, *_ = simulate([t], M, Delta, 20, 120, np.random.default_rng(0))
        P = bartlett.spectrum(sample_cov(X), grid, Delta)
        assert abs(grid[np.argmax(P)] - t) <= 0.05, f"t={t}: peak at {grid[np.argmax(P)]}"


@check("Bartlett equals the average periodogram of the spatial samples (FFT cross-check)")
def _():
    X, *_ = simulate([10, 50], M, Delta, 20, 10, np.random.default_rng(7))
    L = 256
    u = np.fft.fftfreq(L)                                  
    theta = np.rad2deg(np.arcsin(u / Delta))              
    periodogram = np.mean(np.abs(np.fft.fft(X, n=L, axis=0)) ** 2, axis=1) / M
    P = bartlett.spectrum(sample_cov(X), theta, Delta)
    assert np.allclose(P, periodogram, rtol=1e-8), f"max rel err {np.max(np.abs(P / periodogram - 1)):.1e}"


@check("Bartlett half-power width is about 1/M in Delta sin(theta) (0.89/M for a ULA)")
def _():
    X, *_ = simulate([0.0], M, Delta, 20, 120, np.random.default_rng(0))
    w = halfpower_width(bartlett.spectrum(sample_cov(X), grid, Delta))
    assert 0.8 / M < w < 1.0 / M, f"width {w:.3f}, 1/M = {1 / M:.3f}"


@check("Bartlett window: Hamming widens the main lobe and removes the -12 dB sidelobes")
def _():
    X, *_ = simulate([0.0], M, Delta, 20, 120, np.random.default_rng(0))
    R = sample_cov(X)
    P0 = bartlett.spectrum(R, grid, Delta)
    P1 = bartlett.spectrum(R, grid, Delta, window=np.hamming(M))
    assert sidelobe_db(P0) > -13
    assert halfpower_width(P1) > 1.3 * halfpower_width(P0)
    assert sidelobe_db(P1) is None or sidelobe_db(P1) < -20


# ---------------- Phase 4: MVDR ----------------
@check("MVDR: unit gain in the look direction, w^H a = 1 (loading and pinv)")
def _():
    X, *_ = simulate([-20, 30], M, Delta, 200, 20, np.random.default_rng(1))
    R = sample_cov(X)
    for kw in (dict(loading=1e-3), dict(loading=1e-1), dict(pinv=True)):
        for t in (-20.0, 0.0, 30.0, 61.5):
            w = mvdr.weights(R, t, Delta, **kw)
            assert np.isclose(w.conj() @ steering_matrix(t, M, Delta), 1.0), f"{kw}, t={t}"


@check("MVDR resolves two well separated sources on a noisy simulation")
def _():
    X, *_ = simulate([-20, 30], M, Delta, 200, 20, np.random.default_rng(1))
    ang = mvdr.estimate(X, 2, grid, Delta)
    assert ang.size == 2 and np.allclose(ang, [-20, 30], atol=0.3), f"{ang}"


@check("MVDR on rank-deficient R: finite; loading recovers the angles, pinv does not")
def _():
    # exact noiseless data X = A S: R has rank 2, three eigenvalues are ~1e-16 (cond ~ 1e16)
    _, S, A, _ = simulate([-5, 5], M, Delta, 20, 0, np.random.default_rng(0))
    R = sample_cov(A @ S)
    P_load = mvdr.spectrum(R, grid, Delta, loading=1e-3)
    P_pinv = mvdr.spectrum(R, grid, Delta, pinv=True)
    assert np.all(np.isfinite(P_load)) and np.all(np.isfinite(P_pinv))
    ang, ok = pick_peaks(P_load, grid, 2)
    assert ok and np.allclose(ang, [-5, 5], atol=0.06), f"loading: {ang}"
    # documented failure: R^+ keeps only the signal subspace, a^H R^+ a is smallest where a
    # is far from the signal subspace, so the peaks land at angles that are not the sources
    ang, _ = pick_peaks(P_pinv, grid, 2)
    assert ang.size < 2 or np.max(np.abs(ang - [-5, 5])) > 5, f"pinv: {ang}"


# ---------------- Phase 5: MUSIC ----------------
from doa.estimators import music


@check("MUSIC: Un^H a(theta_i) = 0 at the true angles on (almost) noiseless data")
def _():
    thetas = [-12.3, 27.8]
    X, *_ = simulate(thetas, M, Delta, 20, 120, np.random.default_rng(6))   # 120 dB SNR
    _, V = np.linalg.eigh(sample_cov(X))
    Un = V[:, : M - 2]
    for t in thetas:
        r = np.linalg.norm(Un.conj().T @ steering_matrix(t, M, Delta))
        assert r < 1e-4, f"||Un^H a({t})|| = {r:.2e}"
    assert np.all(music.cost(sample_cov(X), grid, 2) >= 0)


@check("MUSIC: recovers noiseless synthetic angles to within half a grid step")
def _():
    thetas = [-12.3, 27.8]
    X, *_ = simulate(thetas, M, Delta, 20, 120, np.random.default_rng(6))
    ang = music.estimate(X, 2, grid)
    assert ang.size == 2 and np.allclose(ang, thetas, atol=0.03), f"{ang}"
    P = music.spectrum(sample_cov(X), grid, 2)
    assert np.all(np.isfinite(P)) and P.max() <= 1 / 1e-8 + 1


@check("MUSIC resolves 10 and 15 deg (below the Rayleigh limit) where Bartlett does not")
def _():
    X, *_ = simulate([10, 15], M, Delta, 200, 30, np.random.default_rng(0))
    ang = music.estimate(X, 2, grid)
    assert ang.size == 2 and np.allclose(ang, [10, 15], atol=0.5), f"MUSIC {ang}"
    _, ok = pick_peaks(bartlett.spectrum(sample_cov(X), grid, Delta), grid, 2)
    assert not ok, "Bartlett unexpectedly resolved the sources"


@check("MUSIC fails on coherent sources (corr = 1): estimates are far from the truth")
def _():
    for seed in range(5):
        X, *_ = simulate([10, 40], M, Delta, 200, 20, np.random.default_rng(seed), corr=1.0)
        ang = music.estimate(X, 2, grid)
        assert ang.size < 2 or np.max(np.abs(ang - [10, 40])) > 2, f"seed {seed}: {ang}"



print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
