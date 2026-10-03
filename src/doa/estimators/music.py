from __future__ import annotations

import numpy as np

from doa.covariance import sample_cov
from doa.model import steering_matrix
from doa.peaks import pick_minima


def cost(R: np.ndarray, grid_deg: np.ndarray, d: int, Delta: float = 0.5) -> np.ndarray:
    #MUSIC cost J(theta) = ||Un^H a(theta)||^2 / ||a(theta)||^2 (book Eq. 3.24).
    #Un = eigenvectors of R for the M-d smallest eigenvalues (noise subspace). J is 0 at a
    #true angle because Un^H a(theta_i) = 0. For a ULA ||a||^2 = M for every theta.

    #number of sensors
    M = R.shape[0]
    #get eigenvectors to estimate col space of A
    _, V = np.linalg.eigh(R) #it is already ordered 
    #first M-d columns = noise subspace          
    Un = V[:, : M - d]
    #get matrix A for this angle                 
    A = steering_matrix(grid_deg, M, Delta)   
    #compute 'cost' to be minimized
    return np.sum(np.abs(Un.conj().T @ A) ** 2, axis=0) / M     


def spectrum(R: np.ndarray, grid_deg: np.ndarray, d: int, Delta: float = 0.5,
             eps: float = 1e-8) -> np.ndarray:
    #innvert in order to detect maxima instead of minima (pseudo-spectrum)
    #add and epsilon parameter for numerical stability
    return 1.0 / (cost(R, grid_deg, d, Delta) + eps)


def estimate(X: np.ndarray, d: int, grid_deg: np.ndarray, Delta: float = 0.5) -> np.ndarray:
    #use the pick_minima function to estimate the DoAs from the MUSIC cost function
    J = cost(sample_cov(X), grid_deg, d, Delta)
    angles, _ = pick_minima(J, grid_deg, d, min_rel_db=-np.inf)
    return angles
