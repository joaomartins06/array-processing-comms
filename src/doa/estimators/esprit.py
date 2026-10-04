from __future__ import annotations

import numpy as np

from doa.covariance import sample_cov


def estimate(X: np.ndarray, d: int, grid_deg: np.ndarray | None = None,
             Delta: float = 0.5) -> np.ndarray:
    #Estimate d DoAs in degrees (sorted) with least-squares ESPRIT. No angle scan.

    #get eigenvectors of the sample covariance matrix
    _, V = np.linalg.eigh(sample_cov(X))  
    #sample the signal subspace
    Us = V[:, -d:]
    #U2 = U1 Psi, where U1 = Us[:-1] and U2 = Us[1:]
    #solve for Psi using least squares
    Psi = np.linalg.lstsq(Us[:-1], Us[1:], rcond=None)[0]   
    #get eigenvalues of Psi
    phi = np.linalg.eigvals(Psi)   
    #convert eigenvalues to spatial frequencies
    #get the phase of phi_i, theta_i = arcsin(angle(phi_i) / (2 pi Delta))
    s = np.clip(np.angle(phi) / (2 * np.pi * Delta), -1.0, 1.0)   # only the phase is used
    return np.sort(np.rad2deg(np.arcsin(s)))
