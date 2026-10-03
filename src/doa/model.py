from __future__ import annotations

import numpy as np


def steering_matrix(theta_deg, M, Delta):
    m = np.arange(M) # sensor indices 0..M-1
    s = np.sin(np.deg2rad(theta_deg))  # sin(theta), scalar or (K,)
    a = np.exp(1j * 2 * np.pi * Delta * np.outer(m, s)) # (M, K), entry [m,k] = m*sin(theta_k)
    return a[:, 0] if np.ndim(theta_deg) == 0 else a


def simulate(thetas_deg: np.ndarray, M: int, Delta: float, N: int, snr_db: float | np.ndarray,            
    rng: np.random.Generator, source_type: str = "qpsk",  corr: float = 0.0,) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:

    #get the angles
    thetas_deg = np.atleast_1d(thetas_deg)
    #compute the number of sources
    d = len(thetas_deg)
    #compute the steering matrix
    A = steering_matrix(thetas_deg, M, Delta)

    # unit-power QPSK symbols, independent across sources and snapshots
    #this is just for the simulations, otherwise the source signals would come from the actual data
    S0 = (rng.choice([-1, 1], (d, N)) + 1j * rng.choice([-1, 1], (d, N))) / np.sqrt(2)

    # add correlation to the source signals
    C = (1 - corr) * np.eye(d) + corr * np.ones((d, d))
    #compute the eigen decomposition of the correlation matrix
    w, V = np.linalg.eigh(C)
    #get a vector of amplitudes for each source
    amp = np.sqrt(10 ** (snr_db * np.ones(d) / 10))  
    #simulate the source by taking into account the correlation and amplitudes
    S = np.diag(amp) @ V @ np.diag(np.sqrt(w.clip(0))) @ S0

    #add white noise, its strength is fixed as we change the SNR in the input
    sigma2 = 1.0
    Noise = (rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))) / np.sqrt(2)
    
    #return X, S, A, sigma2
    return A @ S + Noise, S, A, sigma2