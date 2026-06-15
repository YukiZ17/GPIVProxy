import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import gpytorch
from torch.optim import Adam
import math
from typing import Optional, Dict, List, Tuple, Any

def rbf_kernel(X1, X2, lengthscale=1.0):
    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)

    dist_sq = torch.cdist(X1, X2, p=2)**2
    K = torch.exp(-0.5 * dist_sq / (lengthscale**2))
    return K

def median_heuristic(X):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
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

def generate_data(n_samples: int = 1000, seed: Optional[int] = None):
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

class ProxyGaussianProcess:
    def __init__(self, lengthscale_t=1.0, lengthscale_u=1.0, lengthscale_z=1.0,
                 sigma=0.1, eta=0.1, optimize_t=True, optimize_u=True,
                 optimize_z=False, optimize_sigma=True):
        self.lengthscale_t = torch.tensor(lengthscale_t, dtype=torch.float64, requires_grad=optimize_t)
        self.lengthscale_u = torch.tensor(lengthscale_u, dtype=torch.float64, requires_grad=optimize_u)
        self.lengthscale_z = torch.tensor(lengthscale_z, dtype=torch.float64, requires_grad=optimize_z)
        self.sigma = torch.tensor(sigma, dtype=torch.float64, requires_grad=optimize_sigma)
        self.eta = torch.tensor(eta, dtype=torch.float64, requires_grad=False)

        self.optimize_t = optimize_t
        self.optimize_u = optimize_u
        self.optimize_z = optimize_z
        self.optimize_sigma = optimize_sigma

        self.t_mean = None
        self.t_std = None
        self.y_mean = None
        self.y_std = None

    def set_standardization_params(self, t_mean, t_std, y_mean, y_std):
        self.t_mean = t_mean
        self.t_std = t_std
        self.y_mean = y_mean
        self.y_std = y_std

    def set_data(self, z, u, t, y):
        self.u = u.double()
        self.z = z.double()
        self.y = y.double().reshape(-1, 1)
        self.t = t.double()
        self.n_train = len(self.t)

    def compute_kernel_matrices(self):
        self.Ktt = rbf_kernel(self.t, self.t, self.lengthscale_t).double()
        self.Kzz = rbf_kernel(self.z, self.z, self.lengthscale_z).double()
        self.Kuu = rbf_kernel(self.u, self.u, self.lengthscale_u).double()
        self.Ktz = self.Ktt * self.Kzz

        n = len(self.u)
        self.Ru = (n**(-1)) * torch.sum(self.Kuu, dim=1)
        self.Mu = (n**(-2)) * torch.sum(self.Kuu)

        Ktz_noise = self.Ktz + (self.eta) * torch.eye(len(self.z), dtype=torch.float64)

        try:
            self.invKtz = torch.inverse(Ktz_noise)
        except:
            self.invKtz = torch.pinverse(Ktz_noise)

        self.C = self.Ktt * (self.Ktz @ self.invKtz @ self.Kuu @ self.invKtz @ self.Ktz +
                             (self.sigma)**2 * torch.eye(len(self.z), dtype=torch.float64))

        self.invC = torch.inverse(self.C).double()

    def posterior_mean(self, t_new, standardized=True):
        if not standardized:
            t_new_std = (t_new - self.t_mean) / self.t_std
        else:
            t_new_std = t_new

        t_new_std = t_new_std.double()
        Kt = rbf_kernel(t_new_std.reshape(1, -1), self.t, self.lengthscale_t)
        mean_std = (Kt * (self.Ru @ self.invKtz @ self.Ktz)) @ self.invC @ self.y

        if not standardized:
            mean = unstandardize_data(mean_std, self.y_mean, self.y_std)
            return mean.squeeze()
        else:
            return mean_std.squeeze()

    def posterior_variance(self, t_new, standardized=True):
        return self.posterior_covariance(t_new, t_new, standardized)

    def posterior_covariance(self, t_new, t_new_prime, standardized=True):
        if not standardized:
            t_new_std = (t_new - self.t_mean) / self.t_std
            t_new_prime_std = (t_new_prime - self.t_mean) / self.t_std
        else:
            t_new_std = t_new
            t_new_prime_std = t_new_prime

        t_new_std = t_new_std.double()
        t_new_prime_std = t_new_prime_std.double()

        Kt = rbf_kernel(t_new_std.reshape(1, -1), self.t, self.lengthscale_t)
        Kt_prime = rbf_kernel(t_new_prime_std.reshape(1, -1), self.t, self.lengthscale_t)

        B = Kt * (self.Ru @ self.invKtz @ self.Ktz)
        B_prime = Kt_prime * (self.Ru @ self.invKtz @ self.Ktz)

        ktt_prime = rbf_kernel(t_new_std.reshape(1, -1), t_new_prime_std.reshape(1, -1), self.lengthscale_t)
        covariance_std = ktt_prime * self.Mu - B @ self.invC @ B_prime.T

        if not standardized:
            covariance = covariance_std * (self.y_std**2)
            return covariance.squeeze()
        else:
            return covariance_std.squeeze()

    def predict(self, t_new, standardized=True):
        mean = self.posterior_mean(t_new, standardized)
        variance = self.posterior_variance(t_new, standardized)
        return mean, variance

    def negative_log_marginal_likelihood(self):
        try:
            L = torch.linalg.cholesky(self.C)
            alpha = torch.cholesky_solve(self.y, L)
            data_fit = torch.matmul(self.y.t(), alpha).squeeze()
            log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        except RuntimeError:
            y_t = self.y.t()
            data_fit = torch.matmul(torch.matmul(y_t, self.invC), self.y).squeeze()
            sign, log_det_val = torch.slogdet(self.C)
            log_det = log_det_val

        nll = data_fit + log_det
        return nll

    def optimize_hyperparameters(self, n_iterations=200, lr=0.01, verbose=True):
        params = []
        if self.optimize_sigma:
            params.append(self.sigma)
        if self.optimize_t:
            params.append(self.lengthscale_t)
        if self.optimize_u:
            params.append(self.lengthscale_u)
        if self.optimize_z:
            params.append(self.lengthscale_z)

        if not params:
            print("Warning: No parameters to optimize!")
            return {
                'losses': [],
                'lengthscales_t': [],
                'lengthscales_u': [],
                'lengthscales_z': [],
                'sigmas': []
            }

        optimizer = Adam(params, lr=lr)

        losses = []
        lengthscales_t = []
        lengthscales_u = []
        lengthscales_z = []
        sigmas = []

        if verbose:
            print("Starting hyperparameter optimization...")
            print(f"Initial values: lengthscale_t={self.lengthscale_t.item():.3f}, "
                  f"lengthscale_u={self.lengthscale_u.item():.3f}, "
                  f"lengthscale_z={self.lengthscale_z.item():.3f}, "
                  f"sigma={self.sigma.item():.3f}, eta={self.eta.item():.3f} (fixed)")

        for i in range(n_iterations):
            optimizer.zero_grad()
            self.compute_kernel_matrices()
            loss = self.negative_log_marginal_likelihood()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if self.optimize_t:
                    self.lengthscale_t.data = torch.clamp(self.lengthscale_t, min=1e-3)
                if self.optimize_u:
                    self.lengthscale_u.data = torch.clamp(self.lengthscale_u, min=1e-3)
                if self.optimize_z:
                    self.lengthscale_z.data = torch.clamp(self.lengthscale_z, min=1e-3)
                if self.optimize_sigma:
                    self.sigma.data = torch.clamp(self.sigma, min=1e-6)

            losses.append(loss.item())
            lengthscales_t.append(self.lengthscale_t.item())
            lengthscales_u.append(self.lengthscale_u.item())
            lengthscales_z.append(self.lengthscale_z.item())
            sigmas.append(self.sigma.item())

            if verbose and (i + 1) % 50 == 0:
                print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.3f}, "
                      f"lengthscale_t={self.lengthscale_t.item():.3f}, "
                      f"lengthscale_u={self.lengthscale_u.item():.3f}, "
                      f"lengthscale_z={self.lengthscale_z.item():.3f}, "
                      f"sigma={self.sigma.item():.3f}")

        if verbose:
            print(f"Optimization completed: lengthscale_t={self.lengthscale_t.item():.3f}, "
                  f"lengthscale_u={self.lengthscale_u.item():.3f}, "
                  f"lengthscale_z={self.lengthscale_z.item():.3f}, "
                  f"sigma={self.sigma.item():.3f}")

        return {
            'losses': losses,
            'lengthscales_t': lengthscales_t,
            'lengthscales_u': lengthscales_u,
            'lengthscales_z': lengthscales_z,
            'sigmas': sigmas
        }

    def get_hyperparameters(self):
        return {
            'lengthscale_t': self.lengthscale_t.item(),
            'lengthscale_u': self.lengthscale_u.item(),
            'lengthscale_z': self.lengthscale_z.item(),
            'sigma': self.sigma.item(),
            'eta': self.eta.item()
        }

    def compute_true_structural_function(self, t_values, n_mc=10000, standardized=True):
        if self.t_mean is None or self.t_std is None or self.y_mean is None or self.y_std is None:
            raise ValueError("Standardization parameters must be set before calling compute_true_structural_function.")

        if standardized:
            t_original = unstandardize_data(t_values, self.t_mean, self.t_std)
        else:
            t_original = t_values

        t_original = t_original.double()
        torch.manual_seed(42)

        xi_mc = torch.rand(n_mc, dtype=torch.float64) * 10
        epsilon_3_mc = torch.randn(n_mc, dtype=torch.float64)
        epsilon_5_mc = torch.randn(n_mc, dtype=torch.float64)

        g_xi_mc = g_func(xi_mc)
        W_mc = 7 * g_xi_mc + 45 + epsilon_3_mc

        h_original = torch.zeros_like(t_original, dtype=torch.float64)
        for i, t in enumerate(t_original):
            exp_term = torch.exp((W_mc - t) / 10)
            min_term = torch.minimum(exp_term, torch.tensor(2.0, dtype=torch.float64))
            Y_t = t * min_term - 5 * g_xi_mc + epsilon_5_mc
            h_original[i] = torch.mean(Y_t)

        if standardized:
            h_std = (h_original - self.y_mean) / self.y_std
            return h_std
        else:
            return h_original

    def evaluate_predictions(self, t_test, h_true, standardized=False):
        predictions = []
        variances = []

        for t in t_test:
            pred, var = self.predict(t, standardized=standardized)
            predictions.append(pred.item())
            variances.append(var.item())

        predictions = torch.tensor(predictions, dtype=torch.float64)
        variances = torch.tensor(variances, dtype=torch.float64)
        stds = torch.sqrt(variances)

        errors = torch.abs(predictions - h_true)
        mae = torch.mean(errors)
        mse_orig = torch.mean((predictions - h_true)**2)

        predictions_std = (predictions - self.y_mean) / self.y_std
        h_true_std = (h_true - self.y_mean) / self.y_std
        mse_std = torch.mean((predictions_std - h_true_std)**2)

        lower = predictions - 1.96 * stds
        upper = predictions + 1.96 * stds
        coverage_mask = (h_true >= lower) & (h_true <= upper)
        coverage = torch.mean(coverage_mask.float()).item()

        return mae.item(), mse_orig.item(), mse_std.item(), predictions, variances, errors, coverage

    def evaluate_predictions_with_bootstrap_variance(self, t_test, h_true, bootstrap_means, bootstrap_variances, standardized=False):
        predictions = []

        for t in t_test:
            pred, _ = self.predict(t, standardized=standardized)
            predictions.append(pred.item())

        predictions = torch.tensor(predictions, dtype=torch.float64)

        bootstrap_means_array = np.array(bootstrap_means)
        bootstrap_variances_array = np.array(bootstrap_variances)
        total_variance = np.var(bootstrap_means_array, axis=0) + np.mean(bootstrap_variances_array, axis=0)
        variances = torch.tensor(total_variance, dtype=torch.float64)
        stds = torch.sqrt(variances)

        errors = torch.abs(predictions - h_true)
        mae = torch.mean(errors)
        mse_orig = torch.mean((predictions - h_true)**2)

        predictions_std = (predictions - self.y_mean) / self.y_std
        h_true_std = (h_true - self.y_mean) / self.y_std
        mse_std = torch.mean((predictions_std - h_true_std)**2)

        lower = predictions - 1.96 * stds
        upper = predictions + 1.96 * stds
        coverage_mask = (h_true >= lower) & (h_true <= upper)
        coverage = torch.mean(coverage_mask.float()).item()

        return mae.item(), mse_orig.item(), mse_std.item(), predictions, variances, errors, coverage

    def compute_arc_analysis(self, t_test_orig, h_true_orig, quantile=0.75, remove_step=10,
                             use_bootstrap=False, bootstrap_means=None, bootstrap_variances=None):
        if use_bootstrap and bootstrap_means is not None and bootstrap_variances is not None:
            mae, mse_orig, mse_std, predictions, variances, errors, coverage = self.evaluate_predictions_with_bootstrap_variance(
                t_test_orig, h_true_orig, bootstrap_means, bootstrap_variances, standardized=False
            )
        else:
            mae, mse_orig, mse_std, predictions, variances, errors, coverage = self.evaluate_predictions(
                t_test_orig, h_true_orig, standardized=False
            )

        quantile_threshold = torch.quantile(errors, quantile).item()
        sorted_indices = torch.argsort(variances, descending=True)
        sorted_errors = errors[sorted_indices]

        n_points = len(sorted_errors)
        proportions_below_threshold = []
        removed_counts = []
        auc_vector = []

        for i in range(0, n_points, remove_step):
            if i < n_points:
                remaining_errors = sorted_errors[i:]
                if len(remaining_errors) > 0:
                    below_threshold = (remaining_errors < quantile_threshold).float().sum().item()
                    proportion = below_threshold / len(remaining_errors)
                else:
                    proportion = 0

                proportions_below_threshold.append(proportion)
                removed_counts.append(i)

                if len(proportions_below_threshold) > 1:
                    auc = np.trapz(proportions_below_threshold, removed_counts) / (removed_counts[-1] - removed_counts[0]) if (removed_counts[-1] > removed_counts[0]) else 0
                else:
                    auc = 0
                auc_vector.append(auc)

        final_auc = np.trapz(proportions_below_threshold, removed_counts) / (removed_counts[-1] - removed_counts[0]) if len(removed_counts) > 1 and (removed_counts[-1] > removed_counts[0]) else 0

        return {
            'proportions_below_threshold': proportions_below_threshold,
            'removed_counts': removed_counts,
            'quantile_threshold': quantile_threshold,
            'final_auc': final_auc,
            'auc_vector': auc_vector,
            'coverage': coverage,
            'mae': mae,
            'mse_orig': mse_orig,
            'mse_std': mse_std,
            'errors': errors.numpy(),
            'variances': variances.numpy(),
            'predictions': predictions.numpy(),
            'true_values': h_true_orig.numpy()
        }

def set_seed(seed=442):
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_experiment(config: Dict):
    seed = config['seed']
    n_samples = config['n_samples']
    n_iterations = config['n_iterations']
    lr = config['lr']
    optimize_t = config['optimize_t']
    optimize_u = config['optimize_u']
    optimize_z = config['optimize_z']
    optimize_sigma = config['optimize_sigma']
    initial_sigma = config['initial_sigma']
    eta = config['eta']
    use_median_heuristic = config['use_median_heuristic']
    t_test_start = config['t_test_start']
    t_test_end = config['t_test_end']
    n_test_points = config['n_test_points']
    quantiles = config['quantiles']
    use_bootstrap = config['use_bootstrap']
    bootstrap_iterations = config['bootstrap_iterations']

    set_seed(seed)

    Z_std, U_std, T_std, Y_std, T_mean, T_std_val, Y_mean, Y_std_val = generate_data(
        n_samples=n_samples, seed=seed
    )

    if use_median_heuristic:
        lengthscale_t = median_heuristic(T_std).item()
        lengthscale_u = median_heuristic(U_std).item()
        lengthscale_z = median_heuristic(Z_std).item()
    else:
        lengthscale_t = config.get('initial_lengthscale_t', 1.0)
        lengthscale_u = config.get('initial_lengthscale_u', 1.0)
        lengthscale_z = config.get('initial_lengthscale_z', 1.0)

    gp = ProxyGaussianProcess(
        lengthscale_t=lengthscale_t,
        lengthscale_u=lengthscale_u,
        lengthscale_z=lengthscale_z,
        sigma=initial_sigma,
        eta=eta,
        optimize_t=optimize_t,
        optimize_u=optimize_u,
        optimize_z=optimize_z,
        optimize_sigma=optimize_sigma
    )

    gp.set_data(z=Z_std, u=U_std, t=T_std, y=Y_std)
    gp.set_standardization_params(T_mean, T_std_val, Y_mean, Y_std_val)

    gp.compute_kernel_matrices()

    print(f"Optimizing main model hyperparameters...")
    history = gp.optimize_hyperparameters(n_iterations=n_iterations, lr=lr, verbose=False)
    print(f"Main model optimization completed")

    t_test_orig = torch.linspace(t_test_start, t_test_end, n_test_points, dtype=torch.float64)
    h_true_orig = gp.compute_true_structural_function(t_test_orig, n_mc=10000, standardized=False)

    bootstrap_means = []
    bootstrap_variances = []
    bootstrap_models = []

    if use_bootstrap:
        print(f"Starting bootstrap, iterations: {bootstrap_iterations}")
        for b in range(bootstrap_iterations):
            print(f"  Bootstrap iteration {b+1}/{bootstrap_iterations}")
            bootstrap_seed = 500 * seed + b
            torch.manual_seed(bootstrap_seed)
            np.random.seed(bootstrap_seed)

            n = len(Z_std)
            indices = np.random.choice(n, size=n, replace=True)
            Z_bootstrap = Z_std[indices]
            U_bootstrap = U_std[indices]
            T_bootstrap = T_std[indices]
            Y_bootstrap = Y_std[indices]

            _, T_mean_b, T_std_b = standardize_data(T_bootstrap)
            _, Y_mean_b, Y_std_b = standardize_data(Y_bootstrap)

            if use_median_heuristic:
                lengthscale_t_b = median_heuristic(T_bootstrap).item()
                lengthscale_u_b = median_heuristic(U_bootstrap).item()
                lengthscale_z_b = median_heuristic(Z_bootstrap).item()
            else:
                lengthscale_t_b = gp.lengthscale_t.item()
                lengthscale_u_b = gp.lengthscale_u.item()
                lengthscale_z_b = gp.lengthscale_z.item()

            gp_bootstrap = ProxyGaussianProcess(
                lengthscale_t=lengthscale_t_b,
                lengthscale_u=lengthscale_u_b,
                lengthscale_z=lengthscale_z_b,
                sigma=gp.sigma.item(),
                eta=eta,
                optimize_t=optimize_t,
                optimize_u=optimize_u,
                optimize_z=optimize_z,
                optimize_sigma=optimize_sigma
            )

            gp_bootstrap.set_data(z=Z_bootstrap, u=U_bootstrap, t=T_bootstrap, y=Y_bootstrap)
            gp_bootstrap.set_standardization_params(T_mean_b, T_std_b, Y_mean_b, Y_std_b)

            gp_bootstrap.compute_kernel_matrices()
            gp_bootstrap.optimize_hyperparameters(n_iterations=n_iterations, lr=lr, verbose=False)

            means_b = []
            variances_b = []
            for t in t_test_orig:
                mean_b, var_b = gp_bootstrap.predict(t, standardized=False)
                means_b.append(mean_b.item())
                variances_b.append(var_b.item())

            bootstrap_means.append(means_b)
            bootstrap_variances.append(variances_b)
            bootstrap_models.append(gp_bootstrap)

        print(f"Bootstrap completed")

    arc_results = {}
    if use_bootstrap:
        print(f"Computing ARC analysis with bootstrap variance...")
        for q in quantiles:
            arc_results[q] = gp.compute_arc_analysis(
                t_test_orig, h_true_orig, quantile=q, remove_step=10,
                use_bootstrap=True, bootstrap_means=bootstrap_means, bootstrap_variances=bootstrap_variances
            )
    else:
        print(f"Computing ARC analysis with model's own variance...")
        for q in quantiles:
            arc_results[q] = gp.compute_arc_analysis(
                t_test_orig, h_true_orig, quantile=q, remove_step=10
            )

    hyperparams = gp.get_hyperparameters()
    return arc_results, hyperparams, bootstrap_models

def run_multiple_experiments(config: Dict):
    n_experiments = config['n_experiments']
    base_seed = config['base_seed']
    quantiles = config['quantiles']
    use_bootstrap = config['use_bootstrap']

    all_arc_results = {q: [] for q in quantiles}
    all_hyperparameters = []
    all_coverages = []
    all_mse_orig = []
    all_mse_std = []
    all_bootstrap_models = [] if use_bootstrap else None

    for i in range(n_experiments):
        seed = base_seed + i
        print(f"\n{'='*60}")
        print(f"Running experiment {i+1}/{n_experiments}, seed: {seed}...")
        print(f"{'='*60}")

        try:
            experiment_config = config.copy()
            experiment_config['seed'] = seed

            arc_results, hyperparams, bootstrap_models = run_experiment(experiment_config)

            for q in quantiles:
                all_arc_results[q].append(arc_results[q])

            all_hyperparameters.append(hyperparams)
            all_coverages.append(arc_results[quantiles[0]]['coverage'])
            all_mse_orig.append(arc_results[quantiles[0]]['mse_orig'])
            all_mse_std.append(arc_results[quantiles[0]]['mse_std'])

            if use_bootstrap:
                all_bootstrap_models.append(bootstrap_models)

            print(f"  Experiment {i+1} completed: coverage={arc_results[quantiles[0]]['coverage']:.4f}, "
                  f"mse_orig={arc_results[quantiles[0]]['mse_orig']:.4f}, "
                  f"mse_std={arc_results[quantiles[0]]['mse_std']:.4f}")

        except Exception as e:
            print(f"  Experiment {i+1} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Successfully completed {len(all_coverages)} out of {n_experiments} experiments.")
    print(f"{'='*60}")

    if len(all_coverages) > 0:
        auc_vectors = {}
        for q in quantiles:
            auc_vectors_list = [exp['auc_vector'] for exp in all_arc_results[q]]
            min_len = min(len(vec) for vec in auc_vectors_list)
            auc_vectors_truncated = np.array([vec[:min_len] for vec in auc_vectors_list])
            auc_mean = np.mean(auc_vectors_truncated, axis=0)
            auc_se = np.std(auc_vectors_truncated, axis=0) / np.sqrt(len(auc_vectors_truncated))
            auc_vectors[q] = {
                'mean': auc_mean,
                'se': auc_se,
                'removed_counts': all_arc_results[q][0]['removed_counts'][:min_len]
            }

        accuracy_vectors = {}
        for q in quantiles:
            acc_vectors_list = [exp['proportions_below_threshold'] for exp in all_arc_results[q]]
            min_len = min(len(vec) for vec in acc_vectors_list)
            acc_vectors_truncated = np.array([vec[:min_len] for vec in acc_vectors_list])
            acc_mean = np.mean(acc_vectors_truncated, axis=0)
            acc_se = np.std(acc_vectors_truncated, axis=0) / np.sqrt(len(acc_vectors_truncated))
            accuracy_vectors[q] = {'mean': acc_mean, 'se': acc_se}

        mse_orig_mean = np.mean(all_mse_orig)
        mse_orig_se = np.std(all_mse_orig) / np.sqrt(len(all_mse_orig))
        mse_std_mean = np.mean(all_mse_std)
        mse_std_se = np.std(all_mse_std) / np.sqrt(len(all_mse_std))
        coverage_mean = np.mean(all_coverages)
        coverage_se = np.std(all_coverages) / np.sqrt(len(all_coverages))

        print("\n" + "="*80)
        print("Aggregated Results Across Experiments")
        print("="*80)
        print(f"Number of experiments: {len(all_coverages)}")
        print(f"Using bootstrap variance: {use_bootstrap}")
        if use_bootstrap:
            print(f"Bootstrap iterations: {config['bootstrap_iterations']}")
        print(f"Coverage: mean = {coverage_mean:.4f}, SE = {coverage_se:.4f}")
        for q in quantiles:
            final_auc_values = [all_arc_results[q][i]['final_auc'] for i in range(len(all_arc_results[q]))]
            auc_mean = np.mean(final_auc_values)
            auc_se = np.std(final_auc_values) / np.sqrt(len(final_auc_values))
            print(f"Quantile {q} AUC: mean = {auc_mean:.4f}, SE = {auc_se:.4f}")
        print("="*80)

        plot_accuracy_curves(accuracy_vectors, auc_vectors, quantiles, use_bootstrap)

        results = {
            'auc_vectors': {q: auc_vectors[q]['mean'] for q in quantiles},
            'auc_se_vectors': {q: auc_vectors[q]['se'] for q in quantiles},
            'accuracy_mean_vectors': {q: accuracy_vectors[q]['mean'] for q in quantiles},
            'accuracy_se_vectors': {q: accuracy_vectors[q]['se'] for q in quantiles},
            'mse_orig_vector': all_mse_orig,
            'mse_std_vector': all_mse_std,
            'coverage_vector': all_coverages,
            'all_arc_results': all_arc_results,
            'all_hyperparameters': all_hyperparameters,
            'all_bootstrap_models': all_bootstrap_models,
            'removed_counts': auc_vectors[quantiles[0]]['removed_counts'],
            'summary_stats': {
                'coverage_mean': coverage_mean,
                'coverage_se': coverage_se,
                'mse_orig_mean': mse_orig_mean,
                'mse_orig_se': mse_orig_se,
                'mse_std_mean': mse_std_mean,
                'mse_std_se': mse_std_se
            }
        }
        return results
    else:
        print("No successful experiments to analyze.")
        return None

def plot_accuracy_curves(accuracy_vectors, auc_vectors, quantiles, use_bootstrap):
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r']
    for i, q in enumerate(quantiles):
        removed_counts = auc_vectors[q]['removed_counts']
        plt.plot(removed_counts, accuracy_vectors[q]['mean'],
                 color=colors[i], linewidth=2, label=f'Quantile {q}')
        plt.fill_between(removed_counts,
                         accuracy_vectors[q]['mean'] - accuracy_vectors[q]['se'],
                         accuracy_vectors[q]['mean'] + accuracy_vectors[q]['se'],
                         alpha=0.2, color=colors[i])
    plt.xlabel('Number of High-Variance Points Removed')
    plt.ylabel('Accuracy (Proportion Below Threshold)')
    plt.title(f'Accuracy Curves ({len(accuracy_vectors[quantiles[0]]["mean"])} points, Bootstrap: {use_bootstrap})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    print("="*80)
    print("Proxy Gaussian Process Experiments (with Bootstrap Variance Estimation)")
    print("="*80)

    config = {
        'n_experiments': 15,
        'base_seed': 52,
        'n_samples': 1000,
        'n_iterations': 250,
        'lr': 0.25,
        'optimize_t': True,
        'optimize_u': True,
        'optimize_z': False,
        'optimize_sigma': True,
        'initial_sigma': 0.5,
        'eta': 0.1,
        'use_median_heuristic': True,
        't_test_start': 10,
        't_test_end': 40,
        'n_test_points': 300,
        'quantiles': [0.65, 0.75, 0.85],
        'use_bootstrap': True,
        'bootstrap_iterations': 25,
    }

    print("Configuration parameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    results = run_multiple_experiments(config)

    if results:
        print("\n" + "="*80)
        print("Final Results Summary (Coverage and AUC)")
        print("="*80)
        print(f"Coverage: mean = {results['summary_stats']['coverage_mean']:.4f}, SE = {results['summary_stats']['coverage_se']:.4f}")
        for q in config['quantiles']:
            auc_vec = results['auc_vectors'][q]
            auc_se_vec = results['auc_se_vectors'][q]
            print(f"Quantile {q} AUC vector (mean): length={len(auc_vec)}, mean={np.mean(auc_vec):.4f}")
            print(f"Quantile {q} AUC vector (standard error): length={len(auc_se_vec)}, mean={np.mean(auc_se_vec):.4f}")
        return results
    else:
        print("Experiment failed, no results returned.")
        return None

if __name__ == "__main__":
    results = main()