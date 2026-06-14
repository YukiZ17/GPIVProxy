import torch
import numpy as np

def generate_data(n_samples=100, f_type='absolute', seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    covariance_matrix = torch.tensor([
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float64)
    mean = torch.zeros(3, dtype=torch.float64)
    multivariate_normal = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    samples = multivariate_normal.sample((n_samples,))
    e, v, w = samples[:, 0], samples[:, 1], samples[:, 2]
    normal_dist = torch.distributions.Normal(0, 1)
    alpha = 0.5
    x = normal_dist.cdf((alpha*w + (1-alpha)*v))
    z = normal_dist.cdf(w)
    if f_type == 'sine':
        h_x = 2 * torch.sin(2 * torch.pi * x)
    elif f_type == 'log':
        h_x = torch.log(torch.abs(16 * x - 8) + 1) * torch.sgn(x - 0.5)
    elif f_type == 'linear':
        h_x = 4 * x - 2
    elif f_type == 'absolute':
        h_x = torch.abs(4 * x - 2) - 1
    else:
        raise ValueError(f"Unknown function type: {f_type}")
    y = h_x + e
    return x, y, z, h_x

def median_heuristic(X):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    dist = torch.cdist(X, X).triu(diagonal=1)
    return torch.median(dist[dist > 0])

def rbf_kernel(X1, X2, lengthscale=1.0):
    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)
    dist_sq = torch.cdist(X1, X2, p=2)**2
    K = torch.exp(-0.5 * dist_sq / (lengthscale**2))
    return K