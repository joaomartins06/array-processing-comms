from __future__ import annotations

import time
from typing import Callable

import numpy as np


def time_calls(fn: Callable[[], object], repeat: int = 7, warmup: int = 2) -> np.ndarray:
    #wall-clock seconds for `repeat` calls to fn(), via the monotonic high-resolution
    #clock (not time.time(), which is coarser and not guaranteed monotonic). `warmup`
    #untimed calls run first and are discarded, to avoid biasing the first measurement
    #with one-off costs (first BLAS call, cold cache, page faults).
    for _ in range(warmup):
        fn()
    times = np.empty(repeat)
    for i in range(repeat):
        t0 = time.perf_counter()
        fn()
        times[i] = time.perf_counter() - t0
    return times


def summarize(times: np.ndarray) -> tuple[float, float, float]:
    #(median, min, mean) in seconds. Median is the primary statistic: wall-clock
    #timing noise (OS scheduling, GC) is right-skewed, so the mean is biased upward
    #by rare slow outliers while the median stays representative; min is the
    #classic "best observed" number.
    return float(np.median(times)), float(np.min(times)), float(np.mean(times))
