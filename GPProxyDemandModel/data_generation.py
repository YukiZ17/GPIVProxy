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

def g_func(xi):
    return 2 * (((xi - 5)**4) / 600 + torch.exp(-4 * (xi - 5)**2) + xi / 10 - 2)

def generate_data(n_samples=1000, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    xi = torch.rand(n_samples, dtype=torch.float64) * 10
    epsilon_1 = torch.randn(n_samples, dtype=torch.float64)
    epsilon_2 = torch.randn(n_samples, dtype=torch.float64)
    epsilon_3 = torch.randn(n_samples, dtype=torch.float64)
    epsilon_4 = torch.randn(n_samples, dtype=torch.float64)
    epsilon_5 = torch.randn(n_samples, dtype=torch.float64)
    g_xi = g_func(xi)
    Z1 = 2 * torch.sin(2 * torch.pi * xi / 10) + epsilon_1
    Z2 = 2 * torch.cos(2 * torch.pi * xi / 10) + epsilon_2
    Z = torch.stack([Z1, Z2], dim=1)
    U = 7 * g_xi + 45 + epsilon_3
    T = 35 + (Z1 + 3) * g_xi + Z2 + epsilon_4
    exp_term = torch.exp((U - T) / 10)
    min_term = torch.minimum(exp_term, torch.tensor(2.0, dtype=torch.float64))
    Y = T * min_term - 5 * g_xi + epsilon_5
    Z_std, z_mean, z_std = standardize_data(Z)
    U_std, u_mean, u_std = standardize_data(U)
    T_std, t_mean, t_std = standardize_data(T)
    Y_std, y_mean, y_std = standardize_data(Y)
    return (Z_std, U_std, T_std, Y_std,
            t_mean.squeeze(), t_std.squeeze(),
            y_mean.squeeze(), y_std.squeeze())