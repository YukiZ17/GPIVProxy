import torch
import numpy as np

def h_function(t):
    return 2 * ((t-5)**4 / 600 + torch.exp(-4*(t-5)**2) + t/10 - 2)

def generate_demand_data(n_samples=500, rho=0.9, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    S = torch.randint(1, 8, (n_samples,), dtype=torch.float64)
    T = torch.rand(n_samples, dtype=torch.float64) * 10
    C = torch.randn(n_samples, dtype=torch.float64)
    V = torch.randn(n_samples, dtype=torch.float64)
    P = 25 + (C + 3) * h_function(T) + V
    epsilon = rho * V + torch.sqrt(torch.tensor(1 - rho**2, dtype=torch.float64)) * torch.randn(n_samples, dtype=torch.float64)
    h_T = h_function(T)
    f_struct = 100 + (10 + P) * S * h_T - 2 * P
    Y = f_struct + epsilon
    return Y, P, T, S, C

def standardize_tensor(tensor):
    if tensor.dim() == 0:
        return tensor, tensor, torch.tensor(1.0, dtype=tensor.dtype)
    mean = torch.mean(tensor)
    std = torch.std(tensor)
    if std == 0:
        std = torch.tensor(1.0, dtype=tensor.dtype)
    standardized = (tensor - mean) / std
    return standardized, mean, std

def standardize_data(data_dict):
    scalers = {}
    scaled_data = {}
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            standardized, mean, std = standardize_tensor(value)
            scaled_data[key] = standardized
            scalers[key] = {'mean': mean, 'std': std}
        else:
            value_tensor = torch.tensor(value, dtype=torch.float64)
            standardized, mean, std = standardize_tensor(value_tensor)
            scaled_data[key] = standardized
            scalers[key] = {'mean': mean, 'std': std}
    return scaled_data, scalers

def rbf_kernel(X1, X2, lengthscale=1.0, amplitude=1.0):
    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)
    dist_sq = torch.cdist(X1, X2, p=2)**2
    K = (amplitude**2) * torch.exp(-0.5 * dist_sq / (lengthscale**2))
    return K

def median_heuristic(X):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    dist = torch.cdist(X, X).triu(diagonal=1)
    return torch.median(dist[dist > 0])