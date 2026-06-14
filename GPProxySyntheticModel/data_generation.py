import torch
import numpy as np

def rbf_kernel(X1, X2, lengthscale=1.0):
    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)
    X1 = X1.double()
    X2 = X2.double()
    lengthscale = torch.tensor(lengthscale, dtype=torch.float64) if not isinstance(lengthscale, torch.Tensor) else lengthscale.double()
    dist_sq = torch.cdist(X1, X2, p=2)**2
    K = torch.exp(-0.5 * dist_sq / (lengthscale**2))
    return K

def median_heuristic(X):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    X = X.double()
    dist = torch.cdist(X, X).triu(diagonal=1)
    return torch.median(dist[dist > 0])

def standardize_data(data, mean=None, std=None, eps=1e-8):
    if mean is None or std is None:
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0) + eps
    if torch.any(std == 0):
        std = torch.where(std == 0, torch.ones_like(std), std)
    standardized = (data - mean) / std
    return standardized, mean, std

def unstandardize_data(data, mean, std):
    return data * std + mean

def generate_data(n_samples=100, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    xi_2 = torch.rand(n_samples, dtype=torch.float64) * 3 - 1
    xi_1 = torch.zeros(n_samples, dtype=torch.float64)
    condition = (xi_2 > 0) & (xi_2 < 1)
    indices_condition = torch.where(condition)[0]
    indices_else = torch.where(~condition)[0]
    xi_1[indices_condition] = torch.rand(len(indices_condition), dtype=torch.float64) - 1
    xi_1[indices_else] = torch.rand(len(indices_else), dtype=torch.float64)
    z_1 = xi_1 + torch.rand(n_samples, dtype=torch.float64)
    z_2 = xi_2 + torch.normal(mean=0, std=1**0.5, size=(n_samples,), dtype=torch.float64)
    w_1 = xi_1 + torch.normal(mean=0, std=1**0.5, size=(n_samples,), dtype=torch.float64)
    w_2 = xi_2 + torch.rand(n_samples, dtype=torch.float64)
    x = xi_2 + torch.normal(mean=0, std=1**0.5, size=(n_samples,), dtype=torch.float64)
    y = 3 * torch.cos(2 * (0.3 * xi_1 + 0.3 * xi_2 + 0.2) + 1.5 * x) + torch.normal(mean=0, std=1**0.5, size=(n_samples,), dtype=torch.float64)
    z = torch.stack([z_1, z_2], dim=1)
    w = torch.stack([w_1, w_2], dim=1)
    z_std, z_mean, z_std_val = standardize_data(z)
    w_std, w_mean, w_std_val = standardize_data(w)
    x_std, x_mean, x_std_val = standardize_data(x)
    y_std, y_mean, y_std_val = standardize_data(y)
    return z_std, w_std, x_std, y_std, x_mean, x_std_val, y_mean, y_std_val

def compute_true_structural_function(t_values, t_mean, t_std, y_mean, y_std, n_mc=10000, standardized=True):
    if standardized:
        t_values_orig = unstandardize_data(t_values, t_mean, t_std)
    else:
        t_values_orig = t_values
    t_values_orig = t_values_orig.double()
    torch.manual_seed(42)
    h_orig = torch.zeros_like(t_values_orig, dtype=torch.float64)
    for i, t in enumerate(t_values_orig):
        xi_2_mc = torch.rand(n_mc, dtype=torch.float64) * 3 - 1
        xi_1_mc = torch.zeros(n_mc, dtype=torch.float64)
        condition = (xi_2_mc > 0) & (xi_2_mc < 1)
        indices_condition = torch.where(condition)[0]
        indices_else = torch.where(~condition)[0]
        xi_1_mc[indices_condition] = torch.rand(len(indices_condition), dtype=torch.float64) - 1
        xi_1_mc[indices_else] = torch.rand(len(indices_else), dtype=torch.float64)
        noise_mc = torch.normal(mean=0, std=1**0.5, size=(n_mc,), dtype=torch.float64)
        Y_t = 3 * torch.cos(2 * (0.3 * xi_1_mc + 0.3 * xi_2_mc + 0.2) + 1.5 * t) + noise_mc
        h_orig[i] = torch.mean(Y_t)
    if standardized:
        h_std = (h_orig - y_mean) / y_std
        return h_std
    else:
        return h_orig