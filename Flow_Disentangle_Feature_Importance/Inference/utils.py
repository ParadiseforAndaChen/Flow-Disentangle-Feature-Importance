from __future__ import annotations
import abc
import math
import time
from dataclasses import dataclass, field
from functools import partial
from itertools import product, permutations
from pathlib import Path
from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np
import pandas as pd
import scipy as sp
from numpy.random import default_rng
from numpy import ndarray

from typing import Dict, List, Tuple, Union, Optional
from numpy.typing import NDArray

from itertools import combinations

def make_block_cov(d: int, rho: float, blocks: List[Tuple[int, int]]) -> np.ndarray:
    """Return Σ with ones on the diagonal and ρ on the listed off‑diagonals."""
    Σ = np.eye(d)
    for i, j in blocks:
        Σ[i, j] = Σ[j, i] = rho
    return Σ

def timed(fn):
    """Decorator to time functions – useful for the *computation* metric."""
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        return out, time.perf_counter() - t0
    return wrapper

def compute_power_and_type1(p_values, y_true, alpha=0.05):
    reject_null = (p_values < alpha).astype(int)
    TP = np.sum((reject_null == 1) & (y_true == 1))
    FP = np.sum((reject_null == 1) & (y_true == 0))
    FN = np.sum((reject_null == 0) & (y_true == 1))
    TN = np.sum((reject_null == 0) & (y_true == 0))
    power = TP / (TP + FN) if (TP + FN) > 0 else 0
    type_I_error = FP / (FP + TN) if (FP + TN) > 0 else 0
    return power, type_I_error

def evaluate_importance(p_values, y_true, alpha=0.05,
                         pos_idx=None, neg_idx=None, check_idx=None):
    
    p_values = np.asarray(p_values)
    y_true = np.asarray(y_true)
    n = len(y_true)

    if pos_idx is None:
        pos_idx = np.arange(0, 5)     
    if neg_idx is None:
        neg_idx = np.arange(10, n)    
    if check_idx is None:
        check_idx = np.arange(5, 10)  

    reject = (p_values <= alpha).astype(int)

    pos_true = y_true[pos_idx]
    pos_rej  = reject[pos_idx]
    TP = np.sum((pos_rej == 1) & (pos_true == 1))
    FN = np.sum((pos_rej == 0) & (pos_true == 1))
    power = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    neg_true = y_true[neg_idx]
    neg_rej  = reject[neg_idx]
    FP = np.sum((neg_rej == 1) & (neg_true == 0))
    TN = np.sum((neg_rej == 0) & (neg_true == 0))
    type1_error = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    effective_count = np.sum(p_values[check_idx] < alpha)

    return power, type1_error, effective_count

    
