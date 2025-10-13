from .utils import *

def generate_cov_matrix(d: int, rho: float) -> np.ndarray:
    """Generate a covariance matrix with the specified correlation structure."""
    cov_matrix = np.fromiter((rho**abs(i-j) for i in range(d) for j in range(d)), dtype=float).reshape(d, d)
    np.fill_diagonal(cov_matrix, 1.0)  # Set diagonal elements to 1

    return cov_matrix

def generate_cov_matrix_diag(d: int, rho: float, split_at: int) -> np.ndarray:
    """Generate a block-diagonal covariance matrix with the specified correlation structure."""
    cov_matrix_full = np.fromiter((rho**abs(i-j) for i in range(d) for j in range(d)), dtype=float).reshape(d, d)
    np.fill_diagonal(cov_matrix_full, 1.0)

    cov_matrix_1 = cov_matrix_full[:split_at, :split_at]
    cov_matrix_2 = cov_matrix_full[split_at:, split_at:]

    block_matrix = np.zeros((d, d))
    block_matrix[:split_at, :split_at] = cov_matrix_1
    block_matrix[split_at:, split_at:] = cov_matrix_2

    return block_matrix

def generate_cov_matrix_diag_sec_all_rho(d: int, rho: float, split_at: int) -> np.ndarray:
    """Generate a block-diagonal covariance matrix.
    - The top-left block is AR(1)-like: cov[i,j] = rho^|i-j|
    - The bottom-right block is a constant correlation matrix: diag=1, off-diag=rho
    """
    cov_matrix_1 = np.fromiter(
        (rho**abs(i - j) for i in range(split_at) for j in range(split_at)),
        dtype=float
    ).reshape(split_at, split_at)
    np.fill_diagonal(cov_matrix_1, 1.0)

    d2 = d - split_at
    cov_matrix_2 = np.full((d2, d2), rho)
    np.fill_diagonal(cov_matrix_2, 1.0)

    block_matrix = np.zeros((d, d))
    block_matrix[:split_at, :split_at] = cov_matrix_1
    block_matrix[split_at:, split_at:] = cov_matrix_2

    return block_matrix

def generate_cov_matrix_diag_all_rho(d: int, rho: float, split_at: int) -> np.ndarray:

    cov_matrix_1 = np.full((split_at, split_at), rho)
    np.fill_diagonal(cov_matrix_1, 1.0)

    d2 = d - split_at
    cov_matrix_2 = np.full((d2, d2), rho)
    np.fill_diagonal(cov_matrix_2, 1.0)

    block_matrix = np.zeros((d, d))
    block_matrix[:split_at, :split_at] = cov_matrix_1
    block_matrix[split_at:, split_at:] = cov_matrix_2

    return block_matrix

import numpy as np
from numpy.random import default_rng
from dataclasses import dataclass
from typing import Tuple

def generate_cov_matrix_blocked(d: int, rho: float, split_at: int, tail_block_size: int = 10) -> np.ndarray:

    if split_at <= 0 or split_at > d:
        raise ValueError("split_at must be in (0, d].")
    if not ( -1.0/(d-1) < rho < 1.0 ):
        pass

    tail_dim = d - split_at
    blocks = [split_at]
    if tail_dim > 0:
        k, rem = divmod(tail_dim, tail_block_size)
        blocks.extend([tail_block_size] * k)
        if rem > 0:
            blocks.append(rem)

    Sigma = np.zeros((d, d), dtype=float)
    start = 0
    for bsz in blocks:
        B = np.full((bsz, bsz), rho, dtype=float)
        np.fill_diagonal(B, 1.0)
        Sigma[start:start+bsz, start:start+bsz] = B
        start += bsz

    return Sigma

def generate_cov_matrix_blocked_exp_structure(
    d: int,
    rho: float,
    block_size: int = 5
) -> np.ndarray:

    if d <= 0:
        raise ValueError("d must be positive.")
    if not (-1.0/4 < rho < 1.0):
        raise ValueError("rho must satisfy -1/4 < rho < 1 to ensure PD for 5x5 compound-symmetry blocks.")

    Sigma = np.zeros((d, d), dtype=float)

    if d >= block_size:

        b0 = slice(0, block_size)
        idxA = np.array([0, 1])
        idxB = np.array([2, 3, 4])

        Sigma[np.ix_(idxA, idxA)] = rho

        Sigma[np.ix_(idxB, idxB)] = rho

        Sigma[np.ix_(idxA, idxB)] = 0.0
        Sigma[np.ix_(idxB, idxA)] = 0.0

        Sigma[idxA, idxA] = 1.0
        Sigma[idxB, idxB] = 1.0

        start = block_size
    else:
        B = np.full((d, d), rho, dtype=float)
        np.fill_diagonal(B, 1.0)
        Sigma[:d, :d] = B
        return Sigma
    while start < d:
        end = min(start + block_size, d)
        bsz = end - start
        B = np.full((bsz, bsz), rho, dtype=float)
        np.fill_diagonal(B, 1.0)
        Sigma[start:end, start:end] = B
        start = end

    return Sigma

@dataclass
class Exp1:
    d: int = 50   
    rho: float = 0.6  

    def generate(self, n: int, rho: float, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        rng = default_rng(seed)
        Sigma = generate_cov_matrix_blocked(self.d, rho, split_at=10, tail_block_size=10)
        X = rng.multivariate_normal(np.zeros(self.d), Sigma, size=n)

        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        ε = rng.normal(scale=1.0, size=n)

        y = ( np.arctan(X0 + X1) * (X2 > 0) ) + ( np.sin((X3 * X4)) * (X2 < 0) ) + ε

        return X, y
    
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from numpy.random import default_rng

@dataclass
class Exp2:
    d: int = 50          
    rho: float = 0.6    

    def generate(
        self,
        n: int,
        rho1: Optional[float] = None,          
        rho2: Optional[float] = None,          
        seed: Optional[int] = None,
        mix_weight: float = 0.5,              
        split_at: int = 10,
        tail_block_size: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = default_rng(seed)

        if rho1 is None:
            rho1 = self.rho
        if rho2 is None:
            rho2 = self.rho / 2.0  

        if not (0.0 < mix_weight < 1.0):
            raise ValueError("mix_weight has to be within (0,1) ")

        Sigma1 = generate_cov_matrix_blocked(self.d, rho1, split_at=split_at, tail_block_size=tail_block_size)
        Sigma2 = generate_cov_matrix_blocked(self.d, rho2, split_at=split_at, tail_block_size=tail_block_size)

        comp = rng.binomial(n=1, p=mix_weight, size=n)  

        X = np.empty((n, self.d), dtype=float)
        idx1 = np.where(comp == 1)[0]
        idx2 = np.where(comp == 0)[0]
        if idx1.size > 0:
            X[idx1] = rng.multivariate_normal(mean=np.zeros(self.d), cov=Sigma1, size=idx1.size)
        if idx2.size > 0:
            X[idx2] = rng.multivariate_normal(mean=np.zeros(self.d), cov=Sigma2, size=idx2.size)

        X0, X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
        eps = rng.normal(scale=1.0, size=n)
        y = (np.arctan(X0 + X1) * (X2 > 0)) + (np.sin(X3 * X4) * (X2 < 0)) + eps

        return X, y
