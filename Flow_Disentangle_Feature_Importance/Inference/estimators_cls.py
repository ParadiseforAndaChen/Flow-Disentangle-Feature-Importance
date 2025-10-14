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
import os, random

def _seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        except Exception:
            pass
    except Exception:
        pass

class SimpleMLPRegressor:
    def __init__(self, input_dim, hidden_dims, lr=1e-3, epochs=200, batch_size=128, seed=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers).to(self.device)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dataloader:
                optimizer.zero_grad()
                preds = self.model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()
        return preds


@dataclass
class ImportanceEstimator(abc.ABC):
    """Interface: `.fit(X, y)` then `.importance(X, y, j=None)` returns scalar or array."""
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

        d = int(X.shape[1])

        y_nonan = y[~np.isnan(y)].astype(float)
        vals = np.unique(y_nonan)

        num_classes = len(vals)
        y_onehot = np.eye(num_classes)[y.astype(int)]

        p = np.mean(y_onehot, axis=0)  

        if num_classes == 2:
            Gini_Impurity = np.sum(p * (1 - p)) / 2   # In this case, Var = p(1-p) = Gini Impurity / 2
        else:
            Gini_Impurity= np.sum(p * (1 - p)) # Gini Impurity      

        c = min(np.sqrt(Gini_Impurity), Gini_Impurity, Gini_Impurity**2, Gini_Impurity ** 4) / (d)**2

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
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
import numpy as np
from sklearn.base import clone
from sklearn.utils import shuffle
from tqdm import tqdm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC
from sklearn.model_selection import KFold
import numpy as np
from numpy.random import default_rng

@dataclass
class CPIEstimator_cls(ImportanceEstimator):

    name: str = field(default="CPI_cls", init=False)

    permuter: any = field(
        default_factory=lambda: RandomForestRegressor(
                 n_estimators=300,
                 max_depth=None,
                 min_samples_leaf=3,
                 random_state=42,
                 n_jobs=20
        )
    )
    
    B: int = 50
    random_state: Optional[int] = 0
    show_tqdm: bool = True
    use_predict_proba_first: bool = True  
    eps: float = 1e-12  

    mu_hat: any = field(default=None, init=False)

    classes_: np.ndarray = field(default=None, init=False)
    pos_label_: any = field(default=None, init=False)
    _label_map_: dict = field(default=None, init=False)


    def __post_init__(self):
        if self.random_state is None:
            self.random_state = 0

    def _init_label_mapping(self, y: np.ndarray):
        cls = np.unique(y)
        if len(cls) < 2:
            raise ValueError(f"CPI requires at least two classes; detected {len(cls)} class(es): {cls}.")
        self.classes_ = cls
        self._label_map_ = {cls[i]: i for i in range(len(cls))}

    def _y_to_binary(self, y: np.ndarray) -> np.ndarray:
        if self._label_map_ is None:
            raise RuntimeError("Label map not initialized. Call fit() first.")
        return np.eye(len(self.classes_))[y.astype(int)]  

    def _predict_proba_pos(self, model, X: np.ndarray) -> np.ndarray:
        if self.use_predict_proba_first and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return np.clip(proba, self.eps, 1.0 - self.eps)

        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
            p = 1.0 / (1.0 + np.exp(-s))
            return np.clip(p, self.eps, 1.0 - self.eps)

        yhat = model.predict(X)
        return self._y_to_binary(yhat).astype(float)

    def _ce_loss(self, y_bin: np.ndarray, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, 1.0 - self.eps)

        if y_bin.ndim == 1:  
            n_samples = len(y_bin)
            n_classes = p.shape[1]
            y_bin = np.eye(n_classes)[y_bin]  

        if p.ndim == 2:  # (n, num_classes)
            return -np.sum(y_bin * np.log(p), axis=1)  # (n,)

        elif p.ndim == 3: 
            yb = y_bin[None, :, :]  # (1, n, num_classes)
            yb = np.tile(yb, (p.shape[0], 1, 1))  # (B, n, num_classes)
            return -np.sum(yb * np.log(p), axis=2)  # (B, n)

        else:
            raise ValueError("p must have shape (n, num_classes) or (B, n, num_classes).")

    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):
        rs = int(self.random_state)
        _seed_everything(rs)

        y = np.asarray(y).ravel()
        self._init_label_mapping(y)

        self.mu_hat = clone(self.regressor)
        if hasattr(self.mu_hat, "random_state"):
            self.mu_hat.random_state = rs    
        self.mu_hat.fit(X, y)
        return self

    def _single_fold_importance(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None, **kwargs) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        rs = int(self.random_state)
        _seed_everything(rs)                

        X = np.asarray(X)
        y = np.asarray(y).ravel()
        y_bin = self._y_to_binary(y)

        n, d = X.shape
        p_full = self._predict_proba_pos(self.mu_hat, X)  # (n, num_classes)
        loss_full = self._ce_loss(y_bin, p_full)  # (n,)

        ueifs = np.zeros((n, d)) if (j is None) else np.zeros((n, 1))
        feats = range(d) if j is None else [j]
        iterator = feats
        if self.show_tqdm and (j is None) and d > 1:
            iterator = tqdm(feats, desc="CPI conditional permutation")

        base_seed = rs

        for col_idx, jj in enumerate(iterator):
            X_minus_j = np.delete(X, jj, axis=1)
            x_j = X[:, jj]

            rg = clone(self.permuter)
            if hasattr(rg, "random_state"):
                rg.random_state = base_seed + 7 + 97 * int(jj)

            rg.fit(X_minus_j, x_j)
            x_j_hat = rg.predict(X_minus_j)
            eps = x_j - x_j_hat  

            X_tilde = np.tile(X[None, :, :], (self.B, 1, 1))  # (B, n, d)
            for b in range(self.B):
                eps_perm = shuffle(eps, random_state=base_seed + jj * 1_000_003 + b)
                X_tilde[b, :, jj] = x_j_hat + eps_perm

            X_tilde_flat = X_tilde.reshape(-1, d)  # (B*n, d)
            p_tilde_flat = self._predict_proba_pos(self.mu_hat, X_tilde_flat)  # (B*n, num_classes)
            p_tilde = p_tilde_flat.reshape(self.B, n, p_tilde_flat.shape[1])  # (B, n, num_classes)

            loss_tilde = self._ce_loss(y_bin, p_tilde)  # (B, n)
            delta = loss_tilde - loss_full[None, :]  # (B, n)

            ueif_j = delta.mean(axis=0) * 0.5

            if j is None:
                ueifs[:, jj] = ueif_j
            else:
                ueifs[:, 0] = ueif_j

        if j is None:
            phi_all = np.maximum(ueifs.mean(axis=0), 0.0)  # (d,)
            return phi_all, ueifs
        else:
            phi_j = float(max(ueifs[:, 0].mean(), 0.0))
            return phi_j, ueifs[:, 0]

    def _compute_single_feature_importance(self, X: np.ndarray, y: np.ndarray, j: int, **kwargs) -> Tuple[float, np.ndarray]:
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

from sklearn.base import clone
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsIC

from joblib import Parallel, delayed, parallel_backend
from numpy.random import default_rng
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class CPI_Flow_Model_Estimator_cls(ImportanceEstimator):
    name: str = field(default="CPI_Flow_Model_cls", init=False)
    flow_model: Optional[any] = None  
    permuter: any = field(default_factory=lambda: make_pipeline(StandardScaler(), LassoLarsIC(criterion="bic")))
    B: int = 50
    sampling_method: str = "resample"
    random_state: Optional[int] = 0
    show_tqdm: bool = True
    use_predict_proba_first: bool = True   
    eps: float = 1e-12                     
    n_jobs_jac: int = 30                 
    use_threads_for_jac: bool = True   

    H_max_samples: int = 0          
    H_: np.ndarray | None = field(default=None, init=False)  

    Z_full: np.ndarray | None = field(default=None, init=False)
    mu_hat: any = field(default=None, init=False)

    classes_: np.ndarray = field(default=None, init=False)
    pos_label_: any = field(default=None, init=False)
    _label_map_: dict = field(default=None, init=False)

    def __post_init__(self):
        if self.random_state is None:
            self.random_state = 0

    def _init_label_mapping(self, y: np.ndarray):
        cls = np.unique(y)
        if len(cls) < 2:
            raise ValueError(f"FDFI(CPI) requires at least two classes; detected {len(cls)} class(es): {cls}.")
        self.classes_ = cls
        self._label_map_ = {cls[i]: i for i in range(len(cls))}

    def _y_to_binary(self, y: np.ndarray) -> np.ndarray:
        if self._label_map_ is None:
            raise RuntimeError("Label map not initialized. Call fit() first.")
        return np.vectorize(self._label_map_.get)(y)

    def _predict_proba_pos(self, model, X: np.ndarray) -> np.ndarray:
        if self.use_predict_proba_first and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return np.clip(proba, self.eps, 1.0 - self.eps)

        if hasattr(model, "decision_function"):
            s = model.decision_function(X)  
            p = 1.0 / (1.0 + np.exp(-s))
            if p.ndim == 1:
                p = np.stack([1.0 - p, p], axis=1)
            return np.clip(p, self.eps, 1.0 - self.eps)

        yhat = model.predict(X)
        K = len(self.classes_)
        yhat_int = self._y_to_binary(yhat)
        return np.eye(K)[yhat_int]

    def _ce_loss(self, y_bin: np.ndarray, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, 1.0 - self.eps)
        if y_bin.ndim == 1:
            n_samples = len(y_bin)
            n_classes = p.shape[-1]
            y_bin = np.eye(n_classes, dtype=float)[y_bin]  

        if p.ndim == 2:  
            return -np.sum(y_bin * np.log(p), axis=1)  

        elif p.ndim == 3:  
            yb = y_bin[None, :, :]  
            yb = np.tile(yb, (p.shape[0], 1, 1))  
            return -np.sum(yb * np.log(p), axis=2)  

        else:
            raise ValueError("p must have shape (n, K) or (B, n, K).")

    def _encode_to_Z(self, X: np.ndarray) -> np.ndarray:
        assert self.flow_model is not None, "flow_model is not set."
        _seed_everything(self.random_state)  
        import torch
        with torch.no_grad():
            Z = self.flow_model.sample_batch(X, t_span=(1, 0)).cpu().numpy()
        return Z

    def _decode_to_X(self, Z: np.ndarray) -> np.ndarray:
        assert self.flow_model is not None, "flow_model is not set."
        _seed_everything(self.random_state)  
        import torch
        with torch.no_grad():
            X_hat = self.flow_model.sample_batch(Z, t_span=(0, 1)).cpu().numpy()
        return X_hat

    def _compute_H(self, Z: np.ndarray) -> np.ndarray:
        n, D = Z.shape
        if self.H_max_samples > 0 and self.H_max_samples < n:
            rng = default_rng(int(self.random_state) + 911)
            idx = rng.choice(n, self.H_max_samples, replace=False)
            Z_sub = Z[idx]
        else:
            Z_sub = Z

        def _jac_square(z_row: np.ndarray) -> np.ndarray:
            y0 = np.concatenate([z_row, np.eye(D).reshape(-1)])
            J = self.flow_model.Jacobi_N(y0=y0)  # (D, D)
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
        _seed_everything(int(self.random_state))
        y = np.asarray(y).ravel()
        self._init_label_mapping(y)
        self.mu_hat = clone(self.regressor)
        if hasattr(self.mu_hat, "random_state"):
            self.mu_hat.random_state = int(self.random_state)
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

        _seed_everything(int(self.random_state))
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        y_bin = self._y_to_binary(y)

        n, d = X.shape

        Z = self._encode_to_Z(X)              # (n, d)
        X_hat = self._decode_to_X(Z)          # (n, d)

        p_full = self._predict_proba_pos(self.mu_hat, X_hat)  # (n, K)
        loss_full = self._ce_loss(y_bin, p_full)              # (n,)

        ueifs = np.zeros((n, d))
        iterator = range(d) if j is None else [j]
        if self.show_tqdm and (j is None) and d > 1:
            iterator = tqdm(iterator, desc=f"CPI@Z ({self.sampling_method}) → decode → X (cls CE)")

        base_seed = int(self.random_state)  

        for jj in iterator:
            rng = default_rng(base_seed + 1000003 * int(jj) + 17)

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
                z_j = Z[:, jj]
                rg = clone(self.permuter)
                if hasattr(rg, "random_state"):
                    rg.random_state = base_seed + 7 + 97 * int(jj)
                rg.fit(Z_minus_j, z_j)
                z_j_hat = rg.predict(Z_minus_j)
                eps_z = z_j - z_j_hat
                for b in range(self.B):
                    eps_perm = shuffle(eps_z, random_state=base_seed + b + jj * 1000003)
                    Z_tilde[b, :, jj] = z_j_hat + eps_perm
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")

            Z_tilde_flat = Z_tilde.reshape(-1, d)                        
            Xhat_tilde_flat = self._decode_to_X(Z_tilde_flat)             
            p_tilde_flat = self._predict_proba_pos(self.mu_hat, Xhat_tilde_flat)  
            p_tilde = p_tilde_flat.reshape(self.B, n, p_tilde_flat.shape[1])      

            loss_tilde = self._ce_loss(y_bin, p_tilde)                   
            delta = loss_tilde - loss_full[None, :]                       
            ueif_j = delta.mean(axis=0) * 0.5                            

            ueifs[:, jj] = ueif_j

        if self.H_ is None:
            self.H_ = self._compute_H(Z)  
        ueifs_mapped = ueifs @ self.H_.T       

        phi_all = np.maximum(np.mean(ueifs_mapped, axis=0), 0.0)  
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

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np
from sklearn.base import clone

@dataclass
class LOCOEstimator_cls(ImportanceEstimator):

    regressor: any = None  
    use_predict_proba_first: bool = True   
    eps: float = 1e-12                     
    tau: float = 0.5                       
    mu_full: any = field(default=None, init=False)       
    mu_reduced: any = field(default=None, init=False)    
    name: str = field(default="LOCO_cls", init=False)
    classes_: np.ndarray = field(default=None, init=False)
    pos_label_: any = field(default=None, init=False)
    _label_map_: dict = field(default=None, init=False)  

    def _init_label_mapping(self, y: np.ndarray):
        cls = np.unique(y)
        if len(cls) < 2:
            raise ValueError(f"LOCO requires at least two classes; detected {len(cls)} class(es): {cls}.")
        self.classes_ = cls
        self._label_map_ = {cls[i]: i for i in range(len(cls))}

    def _y_to_binary(self, y: np.ndarray) -> np.ndarray:
        if self._label_map_ is None:
            self._init_label_mapping(y)
        return np.vectorize(self._label_map_.get)(y)

    def _predict_proba_pos(self, model, X: np.ndarray) -> np.ndarray:
        if self.use_predict_proba_first and hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            return np.clip(proba, self.eps, 1.0 - self.eps)

        if hasattr(model, "decision_function"):
            s = model.decision_function(X)
            exp_s = np.exp(s - np.max(s, axis=1, keepdims=True))  
            p = exp_s / np.sum(exp_s, axis=1, keepdims=True)  
            return np.clip(p, self.eps, 1.0 - self.eps)
        
        yhat = model.predict(X)
        return self._y_to_binary(yhat).astype(float)

    def _ce_loss(self, y_bin: np.ndarray, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, 1.0 - self.eps)

        if y_bin.ndim == 1:  
            n_samples = len(y_bin)
            n_classes = p.shape[1]  
            y_bin = np.eye(n_classes)[y_bin]  

        if p.ndim == 2:  
            return -np.sum(y_bin * np.log(p), axis=1)  

        elif p.ndim == 3:  
            yb = y_bin[None, :, :] 
            yb = np.tile(yb, (p.shape[0], 1, 1))  
            return -np.sum(yb * np.log(p), axis=2)  
        
        else:
            raise ValueError("p must have shape (n, num_classes) or (B, n, num_classes).")

    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):

        y = np.asarray(y).ravel()
        self._init_label_mapping(y)

        self.mu_full = clone(self.regressor)
        self.mu_full.fit(X, y)

        d = X.shape[1]
        self.mu_reduced = [clone(self.regressor) for _ in range(d)]
        if j is not None:
            X_red = np.delete(X, j, axis=1)
            self.mu_reduced[j].fit(X_red, y)
        else:
            for jj in range(d):
                X_red = np.delete(X, jj, axis=1)
                self.mu_reduced[jj].fit(X_red, y)
        return self

    def _compute_single_feature_importance(self, X: np.ndarray, y: np.ndarray, j: int) -> Tuple[float, np.ndarray]:
        y = np.asarray(y).ravel()
        y_bin = self._y_to_binary(y)

        p_full = self._predict_proba_pos(self.mu_full, X)              
        loss_full = self._ce_loss(y_bin, p_full)                      

        X_reduced = np.delete(X, j, axis=1)
        p_reduced = self._predict_proba_pos(self.mu_reduced[j], X_reduced)  
        loss_reduced = self._ce_loss(y_bin, p_reduced)                 

        ueif = loss_reduced - loss_full                               
        loco = float(max(ueif.mean(), 0.0))
        return loco, ueif



from joblib import Parallel, delayed, parallel_backend
from sklearn.base import clone
from sklearn.covariance import MinCovDet
from sklearn.utils import shuffle

@dataclass
class DFIZEstimator_cls(ImportanceEstimator):
    regularize: float = 1e-6
    name: str = field(default="DFI_Z_cls", init=False)

    mean: np.ndarray | None = field(default=None, init=False)
    cov:  np.ndarray | None = field(default=None, init=False)
    L:    np.ndarray | None = field(default=None, init=False)      
    L_inv:np.ndarray | None = field(default=None, init=False)      

    mu_full: any = field(default=None, init=False)                  
    Z_full: np.ndarray | None = field(default=None, init=False)     

    n_samples: int = 50
    sampling_method: str = 'resample'  

    refit_cov: bool = False
    refit_mu:  bool = True

    robust: bool = False
    support_fraction: float = 0.8
    random_state: Optional[int] = 0

    use_predict_proba_first: bool = True
    eps: float = 1e-12

    classes_: np.ndarray = field(default=None, init=False)
    pos_label_: any = field(default=None, init=False)
    _label_map_: dict = field(default=None, init=False)

    def _init_label_mapping(self, y: np.ndarray):
        cls = np.unique(y)
        if len(cls) < 2:
            raise ValueError(f"DFI requires at least two classes; detected {len(cls)} class(es): {cls}.")
        self.classes_   = cls
        self._label_map_ = {cls[i]: i for i in range(len(cls))}

    def _y_to_binary(self, y: np.ndarray) -> np.ndarray:
        if self._label_map_ is None:
            raise RuntimeError("Label map not initialized. Call fit() first.")
        return np.vectorize(self._label_map_.get)(y)

    def _predict_proba_pos(self, model, Xc: np.ndarray) -> np.ndarray:

        if self.use_predict_proba_first and hasattr(model, "predict_proba"):
            proba = model.predict_proba(Xc)
            return np.clip(proba, self.eps, 1.0 - self.eps)

        if hasattr(model, "decision_function"):
            s = model.decision_function(Xc)
            exp_s = np.exp(s - np.max(s, axis=1, keepdims=True)) 
            p = exp_s / np.sum(exp_s, axis=1, keepdims=True)  
            return np.clip(p, self.eps, 1.0 - self.eps)

        yhat = model.predict(Xc)
        p = self._y_to_binary(yhat).astype(float)
        return np.clip(p, self.eps, 1.0 - self.eps)

    def _ce_loss(self, y_bin: np.ndarray, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, self.eps, 1.0 - self.eps)

        if y_bin.ndim == 1:  
            n_samples = len(y_bin)
            n_classes = p.shape[1]  
            y_bin = np.eye(n_classes)[y_bin]  

        if p.ndim == 2:  
            return -np.sum(y_bin * np.log(p), axis=1)  

        elif p.ndim == 3:  
            yb = y_bin[None, :, :]  
            yb = np.tile(yb, (p.shape[0], 1, 1)) 
            return -np.sum(yb * np.log(p), axis=2)  

        else:
            raise ValueError("p must have shape (n, num_classes) or (B, n, num_classes).")

    def fit(self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None):
        rs = int(self.random_state)
        _seed_everything(rs)     

        y = np.asarray(y).ravel()
        self._init_label_mapping(y)

        if self.refit_cov or (self.L is None or self.L_inv is None):
            if self.robust:
                cov_est = MinCovDet(support_fraction=self.support_fraction,
                                    random_state=rs).fit(X)
                self.mean = cov_est.location_[None, :]
                self.cov  = cov_est.covariance_
            else:
                self.mean = np.mean(X, axis=0, keepdims=True)
                self.cov  = np.cov(X - self.mean, rowvar=False, ddof=0)
                self.cov  = (self.cov + self.cov.T) / 2

            evals, evecs = np.linalg.eigh(self.cov)
            evals = np.maximum(evals, self.regularize)
            self.L     = evecs @ np.diag(evals**0.5)  @ evecs.T
            self.L_inv = evecs @ np.diag(evals**-0.5) @ evecs.T

        if self.refit_mu or (self.mu_full is None):
            self.mu_full = clone(self.regressor)
            if hasattr(self.mu_full, "random_state"):
                self.mu_full.random_state = rs                
            self.mu_full.fit(X - self.mean, y)     

        if self.Z_full is None:
            self.Z_full = (X - self.mean) @ self.L_inv

        return self

    def _compute_single_feature_importance(self, X: np.ndarray, y: np.ndarray, j: int) -> Tuple[float, np.ndarray]:
        raise NotImplementedError("DFIZEstimator_cls does not support single feature importance directly. Use importance().")

    def _single_fold_importance(
        self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None, **kwargs
    ) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        Z = (X - self.mean) @ self.L_inv
        phi_Z, ueif = self._phi_Z(Z, y)
        if j is not None:
            return phi_Z[j], ueif[:, j]
        else:
            return phi_Z, ueif

    def _phi_Z(self, Z: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rs = int(self.random_state)
        _seed_everything(rs)                                

        n, d = Z.shape
        y = np.asarray(y).ravel()
        y_bin = self._y_to_binary(y)

        Xc      = Z @ self.L
        p_full  = self._predict_proba_pos(self.mu_full, Xc)     
        CE_full = self._ce_loss(y_bin, p_full)                  

        def per_feature(jj: int) -> np.ndarray:
            rng = default_rng(rs + 1_000_003 * int(jj) + 17)
            Z_tilde = np.tile(Z[None, :, :], (self.n_samples, 1, 1))   

            if self.sampling_method == 'resample':
                res_idx = rng.choice(self.Z_full.shape[0], size=(self.n_samples, n), replace=True)
                Z_tilde[:, :, jj] = self.Z_full[res_idx, jj]
            elif self.sampling_method == 'permutation':
                perm_idx = np.array([rng.permutation(n) for _ in range(self.n_samples)])
                Z_tilde[:, :, jj] = Z[perm_idx, jj]
            elif self.sampling_method == 'normal':
                Z_tilde[:, :, jj] = rng.normal(0, 1, size=(self.n_samples, n))
            else:
                raise ValueError(f"Unknown sampling_method: {self.sampling_method}")

            Z_tilde_flat  = Z_tilde.reshape(-1, d)                # (B*n, d)
            Xc_tilde_flat = Z_tilde_flat @ self.L                 # (B*n, d)
            p_flat        = self._predict_proba_pos(self.mu_full, Xc_tilde_flat)  # (B*n, num_classes)
            p_tilde       = p_flat.reshape(self.n_samples, n, p_flat.shape[1])    # (B, n, num_classes)

            p_minus  = p_tilde.mean(axis=0)                       # (n, num_classes)
            CE_minus = self._ce_loss(y_bin, p_minus)              # (n,)

            ueif_j   = CE_minus - CE_full                         # (n,)
            return ueif_j

        ueif_list = Parallel(n_jobs=-1)(delayed(per_feature)(j) for j in range(d))
        ueif = np.stack(ueif_list, axis=1)                      # (n, d)

        phi_Z = np.maximum(np.mean(ueif, axis=0), 0.0)          # (d,)
        return phi_Z, ueif


@dataclass
class DFIEstimator_cls(DFIZEstimator_cls):
    name: str = field(default="DFI_cls", init=False)

    def _single_fold_importance(
        self, X: np.ndarray, y: np.ndarray, j: Optional[int] = None, **kwargs
    ) -> Union[Tuple[float, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        Z = (X - self.mean) @ self.L_inv
        self.phi_Z, self.ueifs_Z = self._phi_Z(Z, y)           
        ueif_X = self.ueifs_Z @ (self.L ** 2).T 
                        
        phi_X  = np.maximum(np.mean(ueif_X, axis=0), 0.0)
        if j is not None:
            return phi_X[j], ueif_X[:, j]
        else:
            return phi_X, ueif_X








