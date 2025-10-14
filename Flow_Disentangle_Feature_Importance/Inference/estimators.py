from .utils import *
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.covariance import MinCovDet
from sklearn.model_selection import KFold
from sklearn.base import clone
import copy
import time
from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import Parallel, delayed
from tqdm import trange
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt

@dataclass
class ImportanceEstimator(abc.ABC):
    random_state: int = 0
    regressor: any = field(default_factory=lambda: RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        random_state=0,
        n_jobs=-1
    ))
    use_cross_fitting: bool = True
    n_folds: int = 2
    name: str = field(init=False)  
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):
        ...

    def importance(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None, **kwargs) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """
        Return importance scores.
        
        Args:
            X: Feature matrix
            y: Target vector
            j: Feature index. If None, return importance for all features.
               If int, return importance for feature j only.
        
        Returns:
            If j is None: array of shape [d] with importance for all features
            If j is int: scalar importance for feature j
        """
        if not self.use_cross_fitting:
            self.n_folds = 1
        return self._cross_fit_importance(X, y, j, **kwargs)
        
    def _cross_fit_importance(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None, **kwargs) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        """Cross-fitting procedure to reduce overfitting bias."""
        if self.n_folds < 2:
            indices = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]  # Single fold, no splitting
        else:
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            indices = kf.split(X)

        ueifs = np.zeros((X.shape[0], X.shape[1]))
        ueifs_Z = np.zeros((X.shape[0], X.shape[1]))
        n_eifs = np.zeros(X.shape[0])

        for train_idx, test_idx in indices:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Create a fresh copy of the estimator for this fold
            fold_estimator = copy.deepcopy(self)
            fold_estimator.fit(X_train, y_train, j)
            
            # Evaluate on test fold
            fi, ueif = fold_estimator._single_fold_importance(X_test, y_test, j, **kwargs)

            ueifs[test_idx, :] += ueif
            if self.name == "DFI": ueifs_Z[test_idx, :] += fold_estimator.ueifs_Z
            n_eifs[test_idx] += 1

        ueifs = ueifs / n_eifs[:, None]
        self.ueifs = ueifs # (n,d)

        id_null_features = np.mean(ueifs, axis=0) < 0
        ueifs[:, id_null_features] = 0
        fi = np.nanmean(ueifs, axis=0) # (d,)
        
        var = np.nanvar(ueifs, axis=0, ddof=1)   

        var_y = float(np.nanvar(y, ddof=1))
  
        c = min(np.sqrt(var_y), var_y, var_y**2, var_y**4) /  ( (X.shape[1])**2 )
       
        sigma = np.sqrt(var + c) + 1e-9

        sqn_eff = np.sqrt(len(X))
        if self.n_folds > 1: sqn_eff *= np.sqrt((self.n_folds - 1) / self.n_folds)
        se = sigma / sqn_eff

        if self.name == "DFI":
            ueifs_Z = ueifs_Z / n_eifs[:, None]
            self.ueifs_Z = ueifs_Z
            id_null_features = np.mean(ueifs_Z, axis=0) < 0
            ueifs_Z[:, id_null_features] = 0
            self.phi_Z = np.mean(ueifs_Z, axis=0)
            self.std_Z = np.std(ueifs_Z, axis=0, ddof=1) / sqn_eff

        if j is None:
            return fi, se
        else:
            return fi[j], se[j]

    def _single_fold_importance(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None, **kwargs) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Compute importance for a single fold.
        
        Returns:
            importance_scores: Array of shape [d] with importance for all features
            ueif: Array of shape [n, d] with uncentered EIF for all features
        """
        n, d = X.shape
        
        if j is not None:
            # Single feature importance
            return self._compute_single_feature_importance(X, y, j, **kwargs)
        else:
            # All features importance
            importance_scores = np.zeros(d)
            ueif = np.full_like(X, np.nan, dtype=float)
            
            for j in range(d):
                importance_scores[j], ueif[:, j] = self._compute_single_feature_importance(X, y, j, **kwargs)

            return importance_scores, ueif

    @abc.abstractmethod
    def _compute_single_feature_importance(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None) -> Tuple[float, np.ndarray]:
        """
        Compute importance of feature j for a single fold.
        
        Returns:
            scalar
        """
        ...


###############################################################################
#
# CPI
#
###############################################################################

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import numpy as np
from sklearn.base import clone
from sklearn.utils import shuffle
from tqdm import tqdm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

@dataclass
class CPIEstimator(ImportanceEstimator):

    name: str = field(default="CPI", init=False)

    permuter: any = field(default_factory=lambda:
    make_pipeline(
        StandardScaler(),
        LassoLarsIC(criterion="bic")  
    )
    )
    B: int = 50
    random_state: Optional[int] = None
    show_tqdm: bool = True


    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):
        self.mu_hat = clone(self.regressor)
        self.mu_hat.fit(X, y)
        return self

    def _single_fold_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        j: Optional[int] = None,
        **kwargs
    ) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

        n, d = X.shape

        pred_full = self.mu_hat.predict(X)    # (n,)
        loss_full = (y - pred_full) ** 2      # (n,)

        ueifs = np.zeros((n, d)) if (j is None) else np.zeros((n, 1))
        feats = range(d) if j is None else [j]
        iterator = feats
        if self.show_tqdm and (j is None) and d > 1:
            iterator = tqdm(feats, desc="CPI conditional permutation")

        base_seed = self.random_state if self.random_state is not None else np.random.randint(0, 10**6)

        for idx, jj in enumerate(iterator):
            X_minus_j = np.delete(X, jj, axis=1)
            x_j = X[:, jj]

            rg = clone(self.permuter)
            rg.fit(X_minus_j, x_j)
            x_j_hat = rg.predict(X_minus_j)
            eps = x_j - x_j_hat

            X_tilde = np.tile(X[None, :, :], (self.B, 1, 1))
            for b in range(self.B):
                eps_perm = shuffle(eps, random_state=base_seed + jj * 1_000_003 + b)
                X_tilde[b, :, jj] = x_j_hat + eps_perm

            X_tilde_flat = X_tilde.reshape(-1, d)                 # (B*n, d)
            y_tilde_flat = self.mu_hat.predict(X_tilde_flat)      # (B*n,)
            y_tilde = y_tilde_flat.reshape(self.B, n)             # (B, n)
            loss_tilde = (y[None, :] - y_tilde) ** 2              # (B, n)

            delta = loss_tilde - loss_full[None, :]               # (B, n)
            ueif_j = 0.5 * delta.mean(axis=0)                     # (n,)

            if j is None:
                ueifs[:, jj] = ueif_j
            else:
                ueifs[:, 0] = ueif_j

        if j is None:
            phi_all = np.maximum(np.mean(ueifs, axis=0), 0.0)      # (d,)
            return phi_all, ueifs
        else:
            phi_j = float(max(np.mean(ueifs[:, 0]), 0.0))
            return phi_j, ueifs[:, 0]

    def _compute_single_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        j: int,
        **kwargs
    ) -> Tuple[float, np.ndarray]:
        phi_j, ueif_j = self._single_fold_importance(X, y, j=j, **kwargs)
        return phi_j, ueif_j

###############################################################################
#
# FDFI_Z(CPI)
#
###############################################################################

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import numpy as np
from numpy.random import default_rng
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from tqdm import tqdm

@dataclass
class CPIZ_Flow_Model_Estimator(ImportanceEstimator):

    name: str = field(default="CPI_Z_Flow_Model", init=False)
    flow_model: Optional[any] = None     
    permuter: any = field(default_factory=lambda: LinearRegression())

    B: int = 50                          
    sampling_method: str = "resample"    
    random_state: Optional[int] = None
    show_tqdm: bool = True               

    Z_full: np.ndarray | None = field(default=None, init=False)

    def _encode_to_Z(self, X: np.ndarray) -> np.ndarray:
        assert self.flow_model is not None, "flow_model is not set."
        import torch
        with torch.no_grad():
            Z = self.flow_model.sample_batch(X, t_span=(1, 0)).cpu().numpy()
        return Z

    def _decode_to_X(self, Z: np.ndarray) -> np.ndarray:
        assert self.flow_model is not None, "flow_model is not set."
        import torch
        with torch.no_grad():
            X_hat = self.flow_model.sample_batch(Z, t_span=(0, 1)).cpu().numpy()
        return X_hat

    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):
        assert self.flow_model is not None, "flow_model is not set."
        self.mu_hat = clone(self.regressor)
        self.mu_hat.fit(X, y)
        self.Z_full = self._encode_to_Z(X) if self.sampling_method == "resample" else None
        return self

    def _single_fold_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        j: Optional[int] = None,
        **kwargs
    ) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

        n, d = X.shape
        Z = self._encode_to_Z(X)      # (n, d)
        X_hat = self._decode_to_X(Z)  # (n, d)
        pred_full = self.mu_hat.predict(X_hat)     # (n,)
        loss_full = (y - pred_full) ** 2           # (n,)

        ueifs = np.zeros((n, d))
        feats = range(d) if j is None else [j]
        iterator = feats
        if self.show_tqdm and (j is None) and d > 1:
            iterator = tqdm(feats, desc=f"CPI-Z [{self.sampling_method}] in Z")

        base_seed = self.random_state if self.random_state is not None else np.random.randint(0, 10**6)

        for jj in iterator:
            rng = default_rng(self.random_state + jj if self.random_state is not None else jj)

            Z_tilde = np.tile(Z[None, :, :], (self.B, 1, 1))  # (B, n, d)

            if self.sampling_method == "resample":
                resample_indices = rng.choice(self.Z_full.shape[0], size=(self.B, n), replace=True)
                Z_tilde[:, :, jj] = self.Z_full[resample_indices, jj]

            elif self.sampling_method == "permutation":
                perm_indices = np.array([rng.permutation(n) for _ in range(self.B)])
                Z_tilde[:, :, jj] = Z[perm_indices, jj]

            elif self.sampling_method == "normal":
                Z_tilde[:, :, jj] = rng.normal(0, 1, size=(self.B, n))

            elif self.sampling_method == "condperm":
                Z_minus_j = np.delete(Z, jj, axis=1)  
                z_j       = Z[:, jj]                  
                rg = clone(self.permuter)
                rg.fit(Z_minus_j, z_j)
                z_j_hat = rg.predict(Z_minus_j)      
                eps     = z_j - z_j_hat              
                for b in range(self.B):
                    eps_perm = shuffle(eps, random_state=base_seed + jj * 1_000_003 + b)
                    Z_tilde[b, :, jj] = z_j_hat + eps_perm
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")

            Z_tilde_flat = Z_tilde.reshape(-1, d)               # (B*n, d)
            Xhat_tilde_flat = self._decode_to_X(Z_tilde_flat)   # (B*n, d)
            y_tilde_flat = self.mu_hat.predict(Xhat_tilde_flat) # (B*n,)
            y_tilde = y_tilde_flat.reshape(self.B, n)           # (B, n)
            loss_tilde = (y[None, :] - y_tilde) ** 2            # (B, n)

            delta = loss_tilde - loss_full[None, :]   # (B, n)
            ueifs[:, jj] = 0.5 * delta.mean(axis=0)  # (n,)

        if j is None:
            phi_all = np.maximum(np.mean(ueifs, axis=0), 0.0)  # (d,)
            return phi_all, ueifs
        else:
            phi_j = float(max(np.mean(ueifs[:, j]), 0.0))
            return phi_j, ueifs[:, j]

    def _compute_single_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        j: int,
        **kwargs
    ) -> Tuple[float, np.ndarray]:
        phi_j, ueif_j = self._single_fold_importance(X, y, j=j, **kwargs)
        return phi_j, ueif_j


###############################################################################
#
# FDFI(CPI)
#
###############################################################################

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import time
import numpy as np
from numpy.random import default_rng
from sklearn.base import clone
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed, parallel_backend
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

@dataclass
class CPI_Flow_Model_Estimator(ImportanceEstimator):
    name: str = field(default="CPI_Flow_Model", init=False)
    flow_model: Optional[any] = None             
    permuter: any = field(default_factory=lambda: LinearRegression())
    B: int = 50
    sampling_method: str = "resample"
    random_state: Optional[int] = None
    show_tqdm: bool = True
    n_jobs_jac: int = 30                 
    use_threads_for_jac: bool = True     
    H_max_samples: int = 0          
    H_: np.ndarray | None = field(default=None, init=False)  

    Z_full: np.ndarray | None = field(default=None, init=False)

    def _encode_to_Z(self, X: np.ndarray) -> np.ndarray:
        assert self.flow_model is not None, "flow_model is not set."
        import torch
        with torch.no_grad():
            Z = self.flow_model.sample_batch(X, t_span=(1, 0)).cpu().numpy()
        return Z

    def _decode_to_X(self, Z: np.ndarray) -> np.ndarray:
        assert self.flow_model is not None, "flow_model is not set."
        import torch
        with torch.no_grad():
            X_hat = self.flow_model.sample_batch(Z, t_span=(0, 1)).cpu().numpy()
        return X_hat

    def _compute_H(self, Z: np.ndarray) -> np.ndarray:
        n, D = Z.shape
        if self.H_max_samples > 0 and self.H_max_samples < n:
            idx = np.random.choice(n, self.H_max_samples, replace=False)
            Z_sub = Z[idx]
        else:
            Z_sub = Z

        def _jac_square(z_row: np.ndarray) -> np.ndarray:
            y0 = np.concatenate([z_row, np.eye(D).reshape(-1)])
            J = self.flow_model.Jacobi_N(y0=y0)   
            return J ** 2

        if self.use_threads_for_jac:
            with parallel_backend("threading", n_jobs=self.n_jobs_jac):
                S_list = Parallel()(delayed(_jac_square)(z) for z in Z_sub)
        else:
            S_list = Parallel(n_jobs=self.n_jobs_jac, prefer="threads")(
                delayed(_jac_square)(z) for z in Z_sub
            )

        H_array = np.stack(S_list, axis=0)  
        return H_array.mean(axis=0)        

    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):
        assert self.flow_model is not None, "flow_model is not set."
        self.mu_hat = clone(self.regressor)
        self.mu_hat.fit(X, y)

        if self.sampling_method == "resample":
            self.Z_full = self._encode_to_Z(X)
        else:
            self.Z_full = None
        self.H_ = None
        return self

    def _single_fold_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        j: Optional[int] = None,
        **kwargs
    ) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

        n, d = X.shape

        Z = self._encode_to_Z(X)          # (n, d)
        X_hat = self._decode_to_X(Z)      # (n, d)

        pred_full = self.mu_hat.predict(X_hat)   # (n,)
        loss_full = (y - pred_full) ** 2         # (n,)

        ueifs = np.zeros((n, d))
        iterator = range(d) if j is None else [j]
        if self.show_tqdm and (j is None) and d > 1:
            iterator = tqdm(iterator, desc=f"CPI@Z ({self.sampling_method}) → decode → X")

        base_seed = self.random_state if self.random_state is not None else np.random.randint(0, 10**6)

        for jj in iterator:
            rng = default_rng(self.random_state + jj if self.random_state is not None else jj)

            B = self.B
            Z_tilde = np.tile(Z[None, :, :], (B, 1, 1))  # (B, n, d)

            if self.sampling_method == "resample":
                resample_indices = rng.choice(self.Z_full.shape[0], size=(B, n), replace=True)
                Z_tilde[:, :, jj] = self.Z_full[resample_indices, jj]

            elif self.sampling_method == "permutation":
                perm_indices = np.array([rng.permutation(n) for _ in range(B)])
                Z_tilde[:, :, jj] = Z[perm_indices, jj]

            elif self.sampling_method == "normal":
                Z_tilde[:, :, jj] = rng.normal(0, 1, size=(B, n))

            elif self.sampling_method == "condperm":
                Z_minus_j = np.delete(Z, jj, axis=1)  
                z_j       = Z[:, jj]                  

                rg = clone(self.permuter)             
                rg.fit(Z_minus_j, z_j)
                z_j_hat = rg.predict(Z_minus_j)       
                eps_z   = z_j - z_j_hat               

                for b in range(B):
                    eps_perm = shuffle(eps_z, random_state=base_seed + b + jj * 1000003)
                    Z_tilde[b, :, jj] = z_j_hat + eps_perm
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")

            Z_tilde_flat = Z_tilde.reshape(-1, d)                 # (B*n, d)
            Xhat_tilde_flat = self._decode_to_X(Z_tilde_flat)     # (B*n, d)
            y_tilde_flat = self.mu_hat.predict(Xhat_tilde_flat)   # (B*n,)
            y_tilde = y_tilde_flat.reshape(B, n)                  # (B, n)
            loss_tilde = (y[None, :] - y_tilde) ** 2              # (B, n)

            delta = loss_tilde - loss_full[None, :]               # (B, n)
            avg_loss_per_sample = delta.mean(axis=0)              # (n,)
            ueifs[:, jj] = 0.5 * avg_loss_per_sample              

        if self.H_ is None:
            t_s0 = time.time()
            self.H_ = self._compute_H(Z)  # (d, d)
            t_s1 = time.time()
            print(f"H: {t_s1 - t_s0:.3f}s")
        else:
            print("H cached, skip computing.")

        ueifs_mapped = ueifs @ self.H_.T   # (n, d)

        phi_all = np.maximum(np.mean(ueifs_mapped, axis=0), 0.0)  # (d,)
        if j is None:
            return phi_all, ueifs_mapped
        else:
            return float(max(phi_all[j], 0.0)), ueifs_mapped[:, j]

    def _compute_single_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        j: int,
        **kwargs
    ) -> Tuple[float, np.ndarray]:
        phi_j, ueif_j = self._single_fold_importance(X, y, j=j, **kwargs)
        return phi_j, ueif_j
    
    def _compute_H_stats(self, Z: np.ndarray, return_array: bool = False):

        n, D = Z.shape
        if self.H_max_samples > 0 and self.H_max_samples < n:
            idx = np.random.choice(n, self.H_max_samples, replace=False)
            Z_sub = Z[idx]
        else:
            Z_sub = Z

        def _jac_square(z_row: np.ndarray) -> np.ndarray:
            y0 = np.concatenate([z_row, np.eye(D).reshape(-1)])
            J = self.flow_model.Jacobi_N(y0=y0)   
            return J ** 2

        if self.use_threads_for_jac:
            with parallel_backend("threading", n_jobs=self.n_jobs_jac):
                S_list = Parallel()(delayed(_jac_square)(z) for z in Z_sub)
        else:
            S_list = Parallel(n_jobs=self.n_jobs_jac, prefer="threads")(
                delayed(_jac_square)(z) for z in Z_sub
            )

        H_array = np.stack(S_list, axis=0)  
        self.H_ = H_array.mean(axis=0)
        self.H_ = self.H_.T
        return (self.H_, H_array) if return_array else self.H_

    def plot_H(
        self,
        Z: np.ndarray | None = None,
        threshold_mean: float = 0.1,
        annotate: bool = True,
        savepath: str | None = None,
        dpi: int = 160,
        export_txt_path: str | None = None,  
    ):

        if self.H_ is None:
            if Z is None:
                raise ValueError("H is None")
            self._compute_H_stats(Z)


        if export_txt_path is not None:
            dir_txt = os.path.dirname(export_txt_path)
            if dir_txt:  
                os.makedirs(dir_txt, exist_ok=True)
            np.savetxt(export_txt_path, self.H_, fmt="%.6f", delimiter="\t")
            print(f"[INFO] H_ has been saved at: {os.path.abspath(export_txt_path)}")

        d = self.H_.shape[0]
        fig, ax = plt.subplots(figsize=(11, 10), dpi=dpi)

        cmap = sns.color_palette("Reds", as_cmap=True)

        if annotate:
            sns.heatmap(self.H_,
                        cmap=cmap,
                        annot=True,
                        fmt=".2f",
                        linewidths=0.5,
                        linecolor="white",
                        cbar=True,
                        ax=ax,
                        annot_kws={"fontsize":8, "color":"white"},
                        mask=(self.H_ <= threshold_mean))
        else:
            sns.heatmap(self.H_,
                        cmap=cmap,
                        annot=False,
                        linewidths=0.5,
                        linecolor="white",
                        cbar=True,
                        ax=ax)

        ax.set_xlabel("Features")
        ax.set_ylabel("Features")
        fig.tight_layout()

        if savepath:
            dir_png = os.path.dirname(savepath)
            if dir_png:
                os.makedirs(dir_png, exist_ok=True)
            fig.savefig(savepath, bbox_inches='tight', dpi=dpi)
            print(f"[INFO] heatmap has been saved at: {os.path.abspath(savepath)}")
            plt.close(fig)
        else:
            plt.show()
    





    




###############################################################################
#
# LOCO 
#
###############################################################################
@dataclass
class LOCOEstimator(ImportanceEstimator):
    mu_full: any = field(default=None, init=False)
    mu_reduced: any = field(default=None, init=False)
    name: str = field(default="LOCO", init=False)

    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):
        self.mu_full = clone(self.regressor)
        self.mu_full.fit(X, y)
        
        self.mu_reduced = [clone(self.regressor) for _ in range(X.shape[1])]
        if j is not None:
            self.mu_reduced[j].fit(np.delete(X, j, axis=1), y)
        else:
            for j in range(X.shape[1]):
                self.mu_reduced[j].fit(np.delete(X, j, axis=1), y)
        
        return self

    def _compute_single_feature_importance(self, X: np.ndarray, y: np.ndarray, j: int) -> Tuple[float, np.ndarray]:
        pred_full = self.mu_full.predict(X)
        X_reduced = np.delete(X, j, axis=1)
        pred_reduced = self.mu_reduced[j].predict(X_reduced)
        ueif = (y - pred_reduced)**2 - (y - pred_full)**2 
        loco = float(max(ueif.mean(axis=0), 0.0))
        return loco, ueif

###############################################################################
#
# FDFI_Z(SCPI)
#
###############################################################################
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import numpy as np
from numpy.random import default_rng
from sklearn.base import clone
from joblib import Parallel, delayed
from tqdm import trange
import torch

@dataclass
class SCPIZ_Flow_Model_Estimator(ImportanceEstimator):
    name: str = field(default="SCPI_Z_Flow_Model", init=False)
    flow_model: Optional[any] = None  
    mu_full: any = field(default=None, init=False)
    Z_full: np.ndarray | None = field(default=None, init=False)

    n_samples: int = 50                       
    sampling_method: str = "resample"         
    random_state: Optional[int] = None

    permuter: any = field(default_factory=lambda: LinearRegression())

    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):
        assert self.flow_model is not None, "flow_model is not set."
        self.mu_full = clone(self.regressor).fit(X, y)
        with torch.no_grad():
            self.Z_full = self.flow_model.sample_batch(X, t_span=(1, 0)).cpu().numpy()
        return self

    def _encode_to_Z(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            Z = self.flow_model.sample_batch(X, t_span=(1, 0)).cpu().numpy()
        return Z

    def _decode_to_X(self, Z: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X_hat = self.flow_model.sample_batch(Z, t_span=(0, 1)).cpu().numpy()
        return X_hat

    def _single_fold_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        j: Optional[int] = None,
        **kwargs
    ) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        n, d = X.shape

        Z = self._encode_to_Z(X)          # (n, d)
        X_hat = self._decode_to_X(Z)      # (n, d)
        y_pred_full = self.mu_full.predict(X_hat)      # (n,)
        base_loss = (y - y_pred_full) ** 2             # (n,)

        ueifs = np.zeros((n, d))
        feats = range(d) if j is None else [j]

        base_seed = self.random_state if self.random_state is not None else np.random.randint(0, 10**6)
        B = self.n_samples

        for jj in trange(len(feats), desc="Feature-wise UEIF computation", leave=False):
            j_idx = feats[jj]

            rng = default_rng(self.random_state + j_idx if self.random_state is not None else j_idx)

            Z_tilde = np.tile(Z[None, :, :], (B, 1, 1))  # (B, n, d)

            if self.sampling_method == "resample":
                assert self.Z_full is not None
                resample_indices = rng.choice(self.Z_full.shape[0], size=(B, n), replace=True)
                Z_tilde[:, :, j_idx] = self.Z_full[resample_indices, j_idx]

            elif self.sampling_method == "permutation":
                perm_indices = np.array([rng.permutation(n) for _ in range(B)])
                Z_tilde[:, :, j_idx] = Z[perm_indices, j_idx]

            elif self.sampling_method == "normal":
                Z_tilde[:, :, j_idx] = rng.normal(0, 1, size=(B, n))

            elif self.sampling_method == "condperm":
                Z_minus_j = np.delete(Z, j_idx, axis=1)  
                z_j       = Z[:, j_idx]                 
                rg = clone(self.permuter)                
                rg.fit(Z_minus_j, z_j)
                z_j_hat = rg.predict(Z_minus_j)          
                eps_z   = z_j - z_j_hat                  

                for b in range(B):
                    eps_perm = shuffle(eps_z, random_state=base_seed + b + j_idx * 1000003)
                    Z_tilde[b, :, j_idx] = z_j_hat + eps_perm

            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")

            Z_tilde_flat = Z_tilde.reshape(-1, d)                 # (B*n, d)
            X_hat_from_Z_tilde = self._decode_to_X(Z_tilde_flat)  # (B*n, d)

            y_perm_flat = self.mu_full.predict(X_hat_from_Z_tilde)    # (B*n,)
            y_perm = y_perm_flat.reshape(B, n).mean(axis=0)           # (n,)

            ueifs[:, j_idx] = (y - y_perm) ** 2 - base_loss  # (n,)

        phi_all = np.maximum(ueifs.mean(axis=0), 0.0)  # (d,)
        if j is None:
            return phi_all, ueifs
        else:
            j0 = j
            return float(max(phi_all[j0], 0.0)), ueifs[:, j0]

    def _compute_single_feature_importance(
        self, X: np.ndarray, y: np.ndarray, j: int
    ) -> Tuple[float, np.ndarray]:
        phi_j, ueif_j = self._single_fold_importance(X, y, j=j)
        return phi_j, ueif_j
    
    
###############################################################################
#
# FDFI(SCPI)
#
###############################################################################

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import time
import numpy as np
from numpy.random import default_rng
from sklearn.base import clone
from sklearn.linear_model import Lasso
from sklearn.utils import shuffle
from joblib import Parallel, parallel_backend, delayed
from tqdm import tqdm

@dataclass
class SCPI_Flow_Model_Estimator(SCPIZ_Flow_Model_Estimator):
    name: str = field(default="SCPI_Flow_Model", init=False)
    n_samples: int = 50                         
    sampling_method: str = "resample"          
    random_state: Optional[int] = None
    n_jobs_jac: int = 30
    use_threads_for_jac: bool = True

    H_max_samples: int = 0                
    H_: np.ndarray | None = field(default=None, init=False)  

    Z_full: np.ndarray | None = field(default=None, init=False)

    permuter: any = field(default_factory=lambda: LinearRegression())

    def _encode_to_Z(self, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            Z = self.flow_model.sample_batch(X, t_span=(1, 0)).cpu().numpy()
        return Z

    def _decode_to_X(self, Z: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            X_hat = self.flow_model.sample_batch(Z, t_span=(0, 1)).cpu().numpy()
        return X_hat

    def _compute_H(self, Z: np.ndarray) -> np.ndarray:
        n, D = Z.shape
        Z_sub = Z

        def _jac_square(z_row: np.ndarray) -> np.ndarray:
            y0 = np.concatenate([z_row, np.eye(D).reshape(-1)])
            J = self.flow_model.Jacobi_N(y0=y0)   
            return J ** 2

        if self.use_threads_for_jac:
            with parallel_backend("threading", n_jobs=self.n_jobs_jac):
                S_list = Parallel()(delayed(_jac_square)(z) for z in Z_sub)
        else:
            S_list = Parallel(n_jobs=self.n_jobs_jac, prefer="threads")(
                delayed(_jac_square)(z) for z in Z_sub
            )
        H_array = np.stack(S_list, axis=0)  # (n, D, D)
        return H_array.mean(axis=0)         # (D, D)

    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):
        assert self.flow_model is not None, "flow_model is not set."
        d = X.shape[1]
        self.mu_full = clone(self.regressor)
        self.mu_full.fit(X, y)
        self.Z_full = self._encode_to_Z(X)
        self.H_ = None
        return self

    def _single_fold_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        j: Optional[int] = None,
        **kwargs
    ) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:

        Z = self._encode_to_Z(X)      # (n, d)
        X_hat = self._decode_to_X(Z)  # (n, d)
        n, d = X.shape

        pred_full = self.mu_full.predict(X_hat)  # (n,)
        base_loss = (y - pred_full) ** 2         # (n,)

        ueifs = np.zeros((n, d))
        feats = range(d) if j is None else [j]
        iterator = feats
        if (j is None) and (len(list(feats)) > 1):
            iterator = tqdm(feats, desc="Feature-wise UEIF computation", leave=False)

        base_seed = self.random_state if self.random_state is not None else np.random.randint(0, 10**6)

        for j_idx in iterator:
            rng = default_rng(self.random_state + j_idx if self.random_state is not None else j_idx)

            B = self.n_samples
            Z_tilde = np.tile(Z[None, :, :], (B, 1, 1))  # (B, n, d)

            if self.sampling_method == "resample":
                resample_indices = rng.choice(self.Z_full.shape[0], size=(B, n), replace=True)
                Z_tilde[:, :, j_idx] = self.Z_full[resample_indices, j_idx]

            elif self.sampling_method == "permutation":
                perm_indices = np.array([rng.permutation(n) for _ in range(B)])
                Z_tilde[:, :, j_idx] = Z[perm_indices, j_idx]

            elif self.sampling_method == "normal":
                Z_tilde[:, :, j_idx] = rng.normal(0, 1, size=(B, n))

            elif self.sampling_method == "condperm":
                Z_minus_j = np.delete(Z, j_idx, axis=1)  
                z_j       = Z[:, j_idx]                  

                rg = clone(self.permuter)                
                rg.fit(Z_minus_j, z_j)
                z_j_hat = rg.predict(Z_minus_j)          
                eps_z   = z_j - z_j_hat                 
                for b in range(B):
                    eps_perm = shuffle(eps_z, random_state=base_seed + b + j_idx * 1000003)
                    Z_tilde[b, :, j_idx] = z_j_hat + eps_perm

            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")

            Z_tilde_flat = Z_tilde.reshape(-1, d)                 # (B*n, d)
            X_hat_from_Z_tilde = self._decode_to_X(Z_tilde_flat)  # (B*n, d)

            y_perm_flat = self.mu_full.predict(X_hat_from_Z_tilde)     # (B*n,)
            y_perm = y_perm_flat.reshape(B, n).mean(axis=0)            # (n,)

            ueifs[:, j_idx] = (y - y_perm) ** 2 - base_loss  # (n,)

        if self.H_ is None:
            t_s0 = time.time()
            self.H_ = self._compute_H(Z)  # (d, d)
            t_s1 = time.time()
            print(f"H: {t_s1 - t_s0:.3f}s")
        else:
            print("H cached, skip computing.")

        ueifs_mapped = ueifs @ self.H_.T  # (n, d)

        phi_all = np.maximum(np.mean(ueifs_mapped, axis=0), 0.0)  # (d,)
        if j is None:
            return phi_all, ueifs_mapped
        else:
            return float(max(phi_all[j], 0.0)), ueifs_mapped[:, j]


###############################################################################
#
# LOCO (normalized)
#
###############################################################################
@dataclass
class nLOCOEstimator(LOCOEstimator):
    mu_full: any = field(default=None, init=False)
    mu_reduced: any = field(default=None, init=False)
    nu: any = field(default=None, init=False)
    name: str = field(default="nLOCO", init=False)

    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):
        super().fit(X, y, j)

        self.nu = [clone(self.regressor) for _ in range(X.shape[1])]
        if j is not None:
            # If j is specified, only fit the conditional model for that feature
            self.nu[j].fit(np.delete(X, j, axis=1), X[:, j])
        else:
            # Fit conditional models for all features
            for j in range(X.shape[1]):
                self.nu[j].fit(np.delete(X, j, axis=1), X[:, j])
            
        return self
    
    def _compute_single_feature_importance(self, X: np.ndarray, y: np.ndarray, j: int) -> Tuple[float, np.ndarray]:
        """Compute normalized LOCO importance for a single feature j."""
        raw_importance, ueif = super()._compute_single_feature_importance(X, y, j)

        # Normalize by conditional variance
        cond_var = self._compute_conditional_variance(X, j)
        ueif /= np.sqrt(cond_var)
        return raw_importance / cond_var if cond_var > 0 else np.nan, ueif

    def _compute_conditional_variance(self, X: np.ndarray, j: int) -> float:
        """Estimate Var(X_j | X_{-j}) = E[(X_j - ν(X_{-j}))²]"""
        W = X[:, j]
        Z = np.delete(X, j, axis=1)
        nu_pred = self.nu[j].predict(Z)
        residuals = W - nu_pred
        return float(np.maximum(np.mean(residuals ** 2), 1e-8))


###############################################################################
#
# Decorrelated LOCO
#
###############################################################################
@dataclass
class dLOCOEstimator(LOCOEstimator):
    reps: int = 1000
    batch_size: int = 32
    mu_full: any = field(default=None, init=False)
    var: np.ndarray = field(default=None, init=False)
    name: str = field(default="dLOCO", init=False)

    def fit(self, X: ndarray, y: ndarray, j: Optional[int] = None):
        super().fit(X, y, j)
        if self.var is None:
            self.var = np.var(X, axis=0, ddof=1)
        return self
    
    def _single_fold_importance(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        vi, ueif = super()._single_fold_importance(X, y, j)
        var = self.var if j is None else self.var[j]
        ueif = ueif / np.sqrt(var)
        return vi / var, ueif

    def _compute_single_feature_importance(self, X: np.ndarray, y: np.ndarray, j: int) -> Tuple[float, np.ndarray]:
        """Compute dLOCO for feature j with pre-fitted model."""
        n, d = X.shape
        rng = default_rng(self.random_state + j)  # Different seed per feature

        # Sample indices for reference samples
        id_j = rng.choice(n, size=min(self.reps, n), replace=False)
        
        # Pre-compute all predictions for original X
        original_preds = self.mu_full.predict(X[id_j])

         # Initialize array to store average predictions
        mu0_j_values = np.zeros(len(id_j))
        
        # Process in batches
        for batch_start in range(0, len(id_j), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(id_j))
            batch_ids = id_j[batch_start:batch_end]
            current_batch_size = len(batch_ids)
            
            # Create 3D array: [batch_size, n_samples, n_features]
            # Each slice is all samples with reference features from one j
            Z_batch = np.zeros((current_batch_size, n, d))
            
            for i, j_ref in enumerate(batch_ids):
                # Fill with reference sample, but keep feature j original
                Z_batch[i, :, :] = np.tile(X[j_ref:j_ref+1], (n, 1))
                Z_batch[i, :, j] = X[:, j]
            
            # Reshape and predict
            Z_batch_flat = Z_batch.reshape(-1, d)
            preds_batch = self.mu_full.predict(Z_batch_flat)
            preds_batch = preds_batch.reshape(current_batch_size, n)
            
            # Average across sample dimension and accumulate
            mu0_j_values[batch_start:batch_end] = np.mean(preds_batch, axis=1)
        
        # Average and compute U-statistic
        U_values = (original_preds - mu0_j_values)**2
        uinf = np.full((n,), np.nan, dtype=float)  # ueif is not used in dLOCO, return empty array
        uinf[:U_values.shape[0]] = U_values
        return float(np.mean(U_values)), uinf



###############################################################################
#
# Shapley value (approximated by sampling)
#
###############################################################################
@dataclass
class ShapleyEstimator(LOCOEstimator):
    n_mc: int = 100
    random_state: Optional[int] = 0
    exact: bool = False 
    name: str = field(default="Shapley", init=False)

    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):
        self.X_train_ = X
        self.y_train_ = y
        self.d = X.shape[1]

        self.mu_full = clone(self.regressor).fit(X, y)

        self._model_cache: Dict[Tuple[int, ...], any] = {}

        return self

    def compute_coalition_and_weight(self, S, j):
        S = tuple(sorted(S))               # coalition w/out j
        Sj = tuple(sorted(S + (j,)))       # coalition with j
        size_S = len(S)
        weight = 1 / (math.comb(self.d - 1, size_S) * self.d)
        return (S, Sj), weight

    def _fit_cached(self, feats: Tuple[int, ...]):
        if feats in self._model_cache:
            return self._model_cache[feats]

        if len(feats) == 0:                 # constant model for empty set
            class _MeanModel:
                def __init__(self, c): self.c = float(c)
                def predict(self, X): return np.full(len(X), self.c)
            mdl = _MeanModel(self.y_train_.mean())
        else:
            mdl = clone(self.regressor).fit(self.X_train_[:, feats], self.y_train_)

        self._model_cache[feats] = mdl
        return mdl


    def _compute_single_feature_importance(
        self, X_test: np.ndarray, y_test: np.ndarray, j: int
    ) -> Tuple[float, np.ndarray]:

        n_test, d = X_test.shape
        
        all_S = [(size, S) for size in range(self.d) for S in combinations([i for i in range(self.d) if i != j], size)]

        results = Parallel(n_jobs=-1)(
            delayed(self.compute_coalition_and_weight)(S, j) for _, S in all_S
        )

        coalitions, weights = zip(*results)
        weights = np.array(weights, dtype=float)
        
        if not self.exact:
            def sample_coalitions_with_weights(coalitions: List[Tuple[Tuple[int, ...], Tuple[int, ...]]], weights: np.ndarray, n_samples: int, rng: np.random.Generator) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
                sampled_indices = rng.choice(len(coalitions), size=n_samples, replace=True, p=weights/weights.sum())
                return [coalitions[i] for i in sampled_indices]

            rng = default_rng(None if self.random_state is None else self.random_state + j)
            coalitions = sample_coalitions_with_weights(coalitions, weights, self.n_mc, rng)
            weights = np.ones(len(coalitions)) / len(coalitions)  
        
        def compute_contrib(S: Tuple[int, ...], Sj: Tuple[int, ...]) -> Tuple[float, np.ndarray]:
            mu_S  = self._fit_cached(S)
            mu_Sj = self._fit_cached(Sj)

            pred_S  = mu_S.predict( X_test[:, S]  if len(S)  else X_test )
            pred_Sj = mu_Sj.predict(X_test[:, Sj])

            contrib = (pred_Sj - pred_S) ** 2 
            psi_r   = contrib.mean()
            return psi_r, contrib

        results = Parallel(n_jobs=-1)(
            delayed(compute_contrib)(S, Sj) for S, Sj in coalitions
        )

        psi_draws, contribs = zip(*results)
        psi_draws = np.array(psi_draws) * weights 
        contribs = np.array(contribs) * weights[:, None]

        psi_hat = float(np.sum(psi_draws)) 
        ueif = contribs.sum(axis=0) 

        return psi_hat, ueif


###############################################################################
#
# DFI of the transformed variable Z
#
###############################################################################
@dataclass
class DFIZEstimator(ImportanceEstimator):
    regularize: float = 1e-6
    name: str = field(default="DFI_Z", init=False)
    
    mean: np.ndarray | None = field(default=None, init=False)  
    cov: np.ndarray | None = field(default=None, init=False)  
    L: np.ndarray | None = field(default=None, init=False)
    L_inv: np.ndarray | None = field(default=None, init=False)

    mu_full: any = field(default=None, init=False)  
    Z_full: np.ndarray | None = field(default=None, init=False) 
    n_samples : int = 50  
    refit_cov: bool = False  
    refit_mu: bool = True  
    robust: bool = False  
    support_fraction: float = 0.8  # Support fraction for robust covariance estimation
    sampling_method: str = 'resample'  
    
    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):        
        if self.refit_cov or (self.L is None or self.L_inv is None):
            if self.robust:
                cov = MinCovDet(support_fraction=self.support_fraction, random_state=self.random_state).fit(X)
                self.mean = cov.location_[None, :]
                self.cov = cov.covariance_
            else:
                self.mean = np.mean(X, axis=0, keepdims=True)
                self.cov = np.cov(X - self.mean, rowvar=False, ddof=0) 
                self.cov = (self.cov+self.cov.T)/2
        
            eigenvals, eigenvecs = np.linalg.eigh(self.cov)
            eigenvals = np.maximum(eigenvals, self.regularize)

            # L_hat = Σ^{1/2} (whitening matrix)            
            self.L = eigenvecs @ np.diag(eigenvals**0.5) @ eigenvecs.T
        
            # L_hat^{-1} = Σ^{-1/2} (inverse whitening matrix)
            self.L_inv = eigenvecs @ np.diag(eigenvals**-0.5) @ eigenvecs.T
        
        if self.refit_mu or (self.mu_full is None):
            self.mu_full = clone(self.regressor)
            self.mu_full.fit(X - self.mean, y) 

        if self.Z_full is None:
            self.Z_full = (X - self.mean) @ self.L_inv 

    def _compute_single_feature_importance(self, X: np.ndarray, y: np.ndarray, j: int) -> Tuple[float, np.ndarray]:
        raise NotImplementedError("DFIZEstimator does not support single feature importance directly. Use importance method instead.")

    def _single_fold_importance(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None, **kwargs) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        Z = (X - self.mean) @ self.L_inv
        phi_Z, ueif = self._phi_Z(Z, y)

        if j is not None:
            return phi_Z[j], ueif[:,j]
        else:
            return phi_Z, ueif

    def _phi_Z(self, Z: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n, d = Z.shape       
        y_pred = self.mu_full.predict(Z @ self.L) 

        def compute_ueif_for_sample(j):
            rng = default_rng(j)
            Z_tilde = np.tile(Z[None, :, :], (self.n_samples, 1, 1))  

            if self.sampling_method == 'resample':
                resample_indices = rng.choice(self.Z_full.shape[0], size=(self.n_samples, n), replace=True)
                Z_tilde[:, :, j] = self.Z_full[resample_indices, j]

            elif self.sampling_method == 'permutation':
                perm_indices = np.array([rng.permutation(n) for _ in range(self.n_samples)])
                Z_tilde[:, :, j] = Z[perm_indices, j]

            elif self.sampling_method == 'normal':
                Z_tilde[:, :, j] = rng.normal(0, 1, size=(self.n_samples, n))
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")

            Z_tilde_flat = Z_tilde.reshape(-1, d) 

            y_perm_flat = self.mu_full.predict(Z_tilde_flat @ self.L) 
            y_perm = y_perm_flat.reshape(self.n_samples, n).mean(axis=0) 
      
            return y_perm 


        y_perm = np.array(Parallel(n_jobs=-1)(
            delayed(compute_ueif_for_sample)(j) for j in range(d)
        )).T 

        ueif = ((y[:, None] - y_perm) ** 2 - (y - y_pred)[:, None] ** 2) 

        phi_Z = np.maximum(np.mean(ueif, axis=0), 0.0) 

        return phi_Z, ueif

###############################################################################
#
# DFI of the original variable X
#
###############################################################################
@dataclass 
class DFIEstimator(DFIZEstimator):
    name: str = field(default="DFI", init=False)

    def _single_fold_importance(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None, **kwargs) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        Z = (X - self.mean) @ self.L_inv
        self.phi_Z, self.ueifs_Z = self._phi_Z(Z, y)

        ueif = self.ueifs_Z @ (self.L ** 2).T
        phi_X = np.maximum(np.mean(ueif, axis=0), 0.0)
       
        if j is not None:
            return phi_X[j], ueif[:,j]
        else:
            return phi_X, ueif




