from __future__ import annotations

import numpy as np

from doa.covariance import sample_cov
from doa.model import steering_matrix
from doa.peaks import pick_peaks


def spectrum(R: np.ndarray, grid_deg: np.ndarray, Delta: float = 0.5,
             window: np.ndarray | None = None) -> np.ndarray:
    #Bartlett pseudo-spectrum P(theta) = a^H R a / (a^H a) (book ET4147 Sec. 3.6.1).

    #compute matrix A
    A = steering_matrix(grid_deg, R.shape[0], Delta)

    #in case we have some weighted sensors (tapering)
    if window is not None:
        A = np.diag(window) @ A
    num = np.sum(A.conj() * (R @ A), axis=0).real  # a^H R a, all angles
    #compute the denominator a^H a for each angle in the grid and return the normalized spectrum
    return num / np.sum(np.abs(A) ** 2, axis=0) 


def estimate(X: np.ndarray, d: int, grid_deg: np.ndarray, Delta: float = 0.5,
             window: np.ndarray | None = None) -> np.ndarray:
    #Estimate d DoAs in degrees (sorted) as the d largest peaks of the Bartlett spectrum.
    angles, _ = pick_peaks(spectrum(sample_cov(X), grid_deg, Delta, window), grid_deg, d)
    return angles
