import numpy as np
from scipy.optimize import minimize
import warnings

class MultiHorizonARIMAX:
    """
    Multi-Horizon ARIMAX(p,d,q) with Exogenous Variables.
    
    This version extends v3 by adding support for exogenous variables (X).
    The model equation becomes:
    z[t] = c + Σφᵢz[t-i] + Σθⱼe[t-j] + Σβₖxₖ[t] + e[t]
    
    Where:
    - z[t] is the differenced series
    - x[t] are the exogenous variables at time t
    - β are the coefficients for exogenous variables
    
    Parameters:
    -----------
    p : int
        Order of autoregressive component
    d : int
        Order of differencing
    q : int
        Order of moving average component
    exog_dim : int
        Number of exogenous variables (0 for standard ARIMA)
    """
    def __init__(self, p=1, d=0, q=0, exog_dim=0):
        if p < 0 or d < 0 or q < 0 or exog_dim < 0:
            raise ValueError("Orders must be non-negative integers")
        if p == 0 and q == 0 and exog_dim == 0:
            raise ValueError("Model must have at least one component (p, q, or exog)")
            
        self.p = p
        self.d = d
        self.q = q
        self.exog_dim = exog_dim
        self.params_ = None
        self.fitted_ = False
        self.series_history_ = None
        
    def difference(self, series, return_history=False):
        """Apply differencing of order d."""
        z = np.asarray(series, dtype=float)
        history = []
        
        for _ in range(self.d):
            if return_history:
                history.append(z[0])
            z = np.diff(z)
            
        if return_history:
            return z, history
        return z
    
    def inverse_difference(self, z_diff, history):
        """Reverse differencing."""
        z = np.asarray(z_diff, dtype=float)
        for initial_val in reversed(history):
            z = np.cumsum(np.concatenate([[initial_val], z]))
        return z
    
    def compute_residuals(self, z, exog, params):
        """
        Compute recursive residuals with exogenous variables.
        
        e[t] = z[t] - (c + AR_term + MA_term + Exog_term)
        """
        c = params[0]
        
        # Extract parameters
        idx = 1
        phi = params[idx:idx+self.p] if self.p > 0 else np.array([])
        idx += self.p
        theta = params[idx:idx+self.q] if self.q > 0 else np.array([])
        idx += self.q
        beta = params[idx:idx+self.exog_dim] if self.exog_dim > 0 else np.array([])
        
        T = len(z)
        e = np.zeros(T)
        
        # Pre-compute exogenous term if present
        exog_term = np.zeros(T)
        if self.exog_dim > 0:
            # exog shape: (T, exog_dim)
            # Ensure exog matches z length (differencing reduces length)
            # We assume exog passed here is already aligned with z
            exog_term = np.dot(exog, beta)
            
        for t in range(T):
            # AR component
            ar_term = 0.0
            for i in range(self.p):
                if t - i - 1 >= 0:
                    ar_term += phi[i] * z[t - i - 1]
            
            # MA component
            ma_term = 0.0
            for j in range(self.q):
                if t - j - 1 >= 0:
                    ma_term += theta[j] * e[t - j - 1]
            
            # Compute residual
            e[t] = z[t] - c - ar_term - ma_term - exog_term[t]
            
        return e
    
    def check_stationarity(self, phi):
        if self.p == 0: return True
        poly_coeffs = np.concatenate([[1], -phi])
        roots = np.roots(poly_coeffs)
        return np.all(np.abs(roots) > 1.0)
    
    def check_invertibility(self, theta):
        if self.q == 0: return True
        poly_coeffs = np.concatenate([[1], theta])
        roots = np.roots(poly_coeffs)
        return np.all(np.abs(roots) > 1.0)
    
    def neg_log_likelihood(self, params, series_list, exog_list):
        log_sigma2 = params[-1]
        sigma2 = np.exp(log_sigma2)
        
        if sigma2 <= 0 or not np.isfinite(sigma2):
            return 1e10
        
        model_params = params[:-1]
        total_ll = 0.0
        
        for i, series in enumerate(series_list):
            z = self.difference(series)
            
            # Handle exogenous variables
            exog = None
            if self.exog_dim > 0:
                # Exog needs to be trimmed to match differenced series length
                # If d=1, z starts at t=1. So exog should also start at t=1.
                # exog_list[i] is full length.
                exog_full = np.asarray(exog_list[i])
                exog = exog_full[self.d:] 
                
                if len(exog) != len(z):
                    # This should be caught in fit validation, but safety check
                    return 1e10
            
            if len(z) < max(self.p, self.q):
                return 1e10
            
            e = self.compute_residuals(z, exog, model_params)
            
            ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (e**2) / sigma2)
            
            if not np.isfinite(ll):
                return 1e10
                
            total_ll += ll
        
        return -total_ll
    
    def fit(self, series_list, exog_list=None, init_params=None, maxiter=1000, method='L-BFGS-B'):
        """
        Fit ARIMAX model.
        
        Parameters:
        -----------
        series_list : list of arrays
            Target time series
        exog_list : list of arrays, optional
            Exogenous variables. Each element must be shape (n_samples, exog_dim)
            Must match length of corresponding series in series_list.
        """
        if len(series_list) == 0:
            raise ValueError("series_list must contain at least one series")
            
        if self.exog_dim > 0:
            if exog_list is None:
                raise ValueError(f"exog_list required for exog_dim={self.exog_dim}")
            if len(exog_list) != len(series_list):
                raise ValueError("exog_list must have same number of series as series_list")
                
        # Validate series and exog
        for i, series in enumerate(series_list):
            series_arr = np.asarray(series)
            if len(series_arr) <= self.d + max(self.p, self.q):
                raise ValueError(f"Series {i} too short")
                
            if self.exog_dim > 0:
                exog_arr = np.asarray(exog_list[i])
                if exog_arr.ndim == 1:
                    exog_arr = exog_arr.reshape(-1, 1)
                if exog_arr.shape[1] != self.exog_dim:
                    raise ValueError(f"Exog {i} has wrong dimension. Expected {self.exog_dim}, got {exog_arr.shape[1]}")
                if len(exog_arr) != len(series_arr):
                    raise ValueError(f"Exog {i} length mismatch. Series: {len(series_arr)}, Exog: {len(exog_arr)}")
        
        self.series_history_ = []
        for series in series_list:
            _, history = self.difference(series, return_history=True)
            self.series_history_.append(history)
            
        # Initialize parameters
        # [c, AR(p), MA(q), Beta(exog_dim), log(sigma2)]
        n_params = 1 + self.p + self.q + self.exog_dim + 1
        
        if init_params is None:
            init_params = np.zeros(n_params)
            
            # Smart initialization
            all_diff = np.concatenate([self.difference(s) for s in series_list])
            init_params[0] = np.mean(all_diff) # Constant
            
            idx = 1
            if self.p > 0:
                init_params[idx:idx+self.p] = 0.1 # AR
                idx += self.p
            if self.q > 0:
                init_params[idx:idx+self.q] = 0.1 # MA
                idx += self.q
            if self.exog_dim > 0:
                init_params[idx:idx+self.exog_dim] = 0.0 # Beta
                idx += self.exog_dim
                
            init_params[-1] = np.log(np.var(all_diff) + 1e-6) # Variance
            
        result = minimize(
            self.neg_log_likelihood,
            init_params,
            args=(series_list, exog_list),
            method=method,
            options={'maxiter': maxiter, 'disp': False}
        )
        
        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")
            
        self.params_ = result.x
        self.fitted_ = True
        
        return self

    def forecast(self, series, exog_history=None, exog_future=None, steps=1, return_differenced=False):
        """
        Forecast with exogenous variables.
        
        Parameters:
        -----------
        series : array-like
            Historical series
        exog_history : array-like, optional
            Exogenous variables corresponding to history (required if exog_dim > 0)
        exog_future : array-like, optional
            Future exogenous variables for forecast steps (required if exog_dim > 0)
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted first")
            
        if self.exog_dim > 0:
            if exog_history is None or exog_future is None:
                raise ValueError("exog_history and exog_future required")
            
            exog_hist = np.asarray(exog_history)
            exog_fut = np.asarray(exog_future)
            
            if exog_hist.ndim == 1: exog_hist = exog_hist.reshape(-1, 1)
            if exog_fut.ndim == 1: exog_fut = exog_fut.reshape(-1, 1)
            
            if len(exog_fut) < steps:
                raise ValueError(f"exog_future length ({len(exog_fut)}) < steps ({steps})")
        
        series = np.asarray(series, dtype=float)
        z, history = self.difference(series, return_history=True)
        
        # Extract params
        c = self.params_[0]
        idx = 1
        phi = self.params_[idx:idx+self.p] if self.p > 0 else np.array([])
        idx += self.p
        theta = self.params_[idx:idx+self.q] if self.q > 0 else np.array([])
        idx += self.q
        beta = self.params_[idx:idx+self.exog_dim] if self.exog_dim > 0 else np.array([])
        
        # Compute residuals for history
        exog_hist_aligned = None
        if self.exog_dim > 0:
            exog_hist_aligned = exog_hist[self.d:]
        
        # Reconstruct params for compute_residuals (excluding sigma2)
        model_params = self.params_[:-1] 
        e = self.compute_residuals(z, exog_hist_aligned, model_params)
        
        z_list = list(z)
        e_list = list(e)
        forecasts_diff = []
        
        for step in range(steps):
            # AR term
            ar_term = 0.0
            for i in range(self.p):
                idx_z = len(z_list) - i - 1
                if idx_z >= 0:
                    ar_term += phi[i] * z_list[idx_z]
            
            # MA term
            ma_term = 0.0
            for j in range(self.q):
                idx_e = len(e_list) - j - 1
                if idx_e >= 0:
                    ma_term += theta[j] * e_list[idx_e]
            
            # Exog term
            exog_term = 0.0
            if self.exog_dim > 0:
                exog_term = np.dot(exog_fut[step], beta)
            
            z_next = c + ar_term + ma_term + exog_term
            e_next = 0.0
            
            z_list.append(z_next)
            e_list.append(e_next)
            forecasts_diff.append(z_next)
            
        forecasts_diff = np.array(forecasts_diff)
        
        if return_differenced:
            return forecasts_diff
            
        combined = np.concatenate([z, forecasts_diff])
        combined_original = self.inverse_difference(combined, history)
        return combined_original[len(series):]

    def get_params(self):
        if not self.fitted_:
            raise ValueError("Model not fitted")
            
        idx = 1
        ar_coefs = self.params_[idx:idx+self.p] if self.p > 0 else np.array([])
        idx += self.p
        ma_coefs = self.params_[idx:idx+self.q] if self.q > 0 else np.array([])
        idx += self.q
        exog_coefs = self.params_[idx:idx+self.exog_dim] if self.exog_dim > 0 else np.array([])
        
        return {
            'constant': self.params_[0],
            'ar_coefs': ar_coefs,
            'ma_coefs': ma_coefs,
            'exog_coefs': exog_coefs,
            'sigma2': np.exp(self.params_[-1])
        }
