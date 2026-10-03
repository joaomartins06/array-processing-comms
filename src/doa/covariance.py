from __future__ import annotations

import numpy as np


def sample_cov(X: np.ndarray) -> np.ndarray:
    #compute covariance matrix of X
    return X @ X.conj().T / X.shape[1]
