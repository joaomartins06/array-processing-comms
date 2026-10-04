from __future__ import annotations

import numpy as np


def eigen_profile(R: np.ndarray) -> np.ndarray:
    #sorted eigenvalues in descending order
    return np.linalg.eigvalsh(R)[::-1]


def mdl_aic(R: np.ndarray, N: int) -> tuple[np.ndarray, np.ndarray]:
    #taken from the krim paper, in order to estimate the number of sources using MDL and AIC criteria
    M = R.shape[0]
    lam = np.maximum(eigen_profile(R), 1e-12 * eigen_profile(R)[0])
    k = np.arange(M)
    LL = np.array([
        N * (M - kk) * (np.log(lam[kk:].mean()) - np.log(lam[kk:]).mean()) for kk in k
    ])
    n_par = k * (2 * M - k)
    return LL + 0.5 * n_par * np.log(N), 2 * LL + 2 * n_par


def estimate_order(R: np.ndarray, N: int, method: str = "mdl") -> int:
    #Estimated number of sources: argmin over k of the MDL or AIC criterion.
    #this is done by locating the index of the smallest value of the chosen criterion
    #this is a consequence of the fact that the MDL and AIC criteria are functions of k that dips after the true number of sources
    mdl, aic = mdl_aic(R, N)
    return int(np.argmin(mdl if method == "mdl" else aic))
