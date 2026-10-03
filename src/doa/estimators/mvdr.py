from __future__ import annotations

import numpy as np

from doa.covariance import sample_cov
from doa.model import steering_matrix
from doa.peaks import pick_peaks


def _Rinv_times(R: np.ndarray, A: np.ndarray, loading: float, pinv: bool) -> np.ndarray:
    #computes R^-1 a
    M = R.shape[0]
    #whether to use the pseudo-inverse or diagonal loading
    if pinv:
        return np.linalg.pinv(R, rcond=1e-6, hermitian=True) @ A
    #the loading factor for diagonal loading
    #loading is the fraction of the mean power that we are inserting smooth the inversion (numerical stability)
    eps = loading * np.trace(R).real / M
    return np.linalg.solve(R + eps * np.eye(M), A)


def spectrum(R: np.ndarray, grid_deg: np.ndarray, Delta: float = 0.5,
             loading: float = 1e-3, pinv: bool = False) -> np.ndarray:
    #MVDR pseudo-spectrum P(theta) = 1 / (a^H R^-1 a) (book ET4147 Sec. 3.6.2).
    #compute matrix A
    A = steering_matrix(grid_deg, R.shape[0], Delta) 
    #compute the denominator a^H R^-1 a for each angle in the grid
    denom = np.sum(A.conj() * _Rinv_times(R, A, loading, pinv), axis=0).real 
    #invert it to get the MVDR spectrum
    return 1.0 / np.maximum(denom, np.finfo(float).tiny)


def weights(R: np.ndarray, theta_deg: float, Delta: float = 0.5,
            loading: float = 1e-3, pinv: bool = False) -> np.ndarray:
    #MVDR weights w = R^-1 a / (a^H R^-1 a), shape (M,); w^H a = 1 
    a = steering_matrix(theta_deg, R.shape[0], Delta)      
    Ra = _Rinv_times(R, a[:, None], loading, pinv)[:, 0]
    return Ra / (a.conj() @ Ra)


def estimate(X: np.ndarray, d: int, grid_deg: np.ndarray, Delta: float = 0.5,
             loading: float = 1e-3, pinv: bool = False) -> np.ndarray:
    #estimate angles by picking the d largest peaks of the MVDR spectrum
    angles, _ = pick_peaks(spectrum(sample_cov(X), grid_deg, Delta, loading, pinv), grid_deg, d)
    return angles
