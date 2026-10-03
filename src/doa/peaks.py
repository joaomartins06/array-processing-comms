from __future__ import annotations

import numpy as np


def pick_peaks(spectrum: np.ndarray, grid_deg: np.ndarray, d: int, min_rel_db: float = -10.0,
    min_dip_db: float = 3.0,) -> tuple[np.ndarray, bool]:

    s = np.asarray(spectrum, dtype=float)
    #pinpoint local maxima by evaluating the left and right neighbors of each middle point
    cand = np.where((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:]))[0] + 1 
    if cand.size == 0:
        return np.array([]), False

    #check if the peaks are above a certain relative height threshold
    cand = cand[s[cand] >= s[cand].max() * 10 ** (min_rel_db / 10)]   # height condition
    sel = np.sort(cand[np.argsort(s[cand])[::-1][:d]])                # d highest, by angle

    while True: 
        # some peak fails are not actually maximums, so we check the dip condition between consecutive peaks
        for i in range(len(sel) - 1):
            lo, hi = sel[i], sel[i + 1]
            lower = min(s[lo], s[hi])
            #the minimum between two proposed peaks must be sufficiently deep
            #otherwise, say we have some noise, the two peaks would merge or swap 
            #so we require a minimum dip between them
            if s[lo:hi + 1].min() > lower * 10 ** (-min_dip_db / 10):
                sel = np.delete(sel, i if s[lo] < s[hi] else i + 1)   # drop the lower peak
                break
        else:
            break

    return grid_deg[sel], len(sel) == d


def pick_minima(cost: np.ndarray, grid_deg: np.ndarray, d: int, **kw) -> tuple[np.ndarray, bool]:
    
    return pick_peaks(1.0 / np.maximum(cost, np.finfo(float).tiny), grid_deg, d, **kw)
