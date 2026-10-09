from __future__ import annotations

import numpy as np


def is_resolved(est: np.ndarray, true: np.ndarray, tol: float | None = None) -> bool:
    #checks if the estimated angles match the true angles within a tolerance
    #tol=None: half the smallest separation between true angles; otherwise an absolute tol in degrees
    est, true = np.sort(np.asarray(est, dtype=float)), np.sort(np.asarray(true, dtype=float))
    if est.size != true.size:
        return False

    if tol is None:
        #check the differences in angles between sources
        #tol is the min difference/2, as if some point lands at half the separation, then we dont know
        #which to compare with
        tol = np.min(np.diff(true)) / 2 if true.size > 1 else np.inf #if there is one source, tol is infinite
    #check if all of them are inside tolerance, otherwise it fails (False)
    return bool(np.all(np.abs(est - true) < tol))


def resolution_probability(ests: list[np.ndarray], true: np.ndarray, tol: float | None = None) -> float:
    #resolution probability is resolved/total trials (tol as in is_resolved)
    return float(np.mean([is_resolved(e, true, tol) for e in ests]))


def rmse(ests: list[np.ndarray], true: np.ndarray, tol: float | None = None) -> float:
    #compute the RMSE over resolved trials only, unresolved trials are ignored (tol as in is_resolved)
    #returns nan if no trial is resolved
    #RMSE = sqrt(1/N sum(true - estimate)^2))
    true = np.sort(np.asarray(true, dtype=float))
    errs = [np.sort(e) - true for e in ests if is_resolved(e, true, tol)]
    return float(np.sqrt(np.mean(np.concatenate(errs) ** 2))) if errs else float("nan")
