import torch
import numpy as np
from typing import Optional, Tuple, List, Dict
import math
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

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

def generate_data(n_samples: int = 100, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    return Z_std, U_std, T_std, Y_std, t_mean.squeeze(), t_std.squeeze(), y_mean.squeeze(), y_std.squeeze()

class ProxyGaussianProcess:
    def __init__(self, lengthscale_t=1.0, lengthscale_u=1.0, lengthscale_z=1.0,
                 lambda_val=0.5, eta=0.1):
        self.lengthscale_t = torch.tensor(lengthscale_t, dtype=torch.float64)
        self.lengthscale_u = torch.tensor(lengthscale_u, dtype=torch.float64)
        self.lengthscale_z = torch.tensor(lengthscale_z, dtype=torch.float64)
        self.lambda_val = torch.tensor(lambda_val, dtype=torch.float64)
        self.eta = torch.tensor(eta, dtype=torch.float64)

    def set_data(self, z, u, t, y, t_mean=None, t_std=None, y_mean=None, y_std=None):
        self.u = u.double()
        self.z = z.double()
        self.y = y.double().reshape(-1, 1)
        self.t = t.double()

        self.t_mean = t_mean if t_mean is not None else torch.tensor(0.0, dtype=torch.float64)
        self.t_std = t_std if t_std is not None else torch.tensor(1.0, dtype=torch.float64)
        self.y_mean = y_mean if y_mean is not None else torch.tensor(0.0, dtype=torch.float64)
        self.y_std = y_std if y_std is not None else torch.tensor(1.0, dtype=torch.float64)

        self._compute_kernel_matrices()

    def _compute_kernel_matrices(self):
        self.Ktt = rbf_kernel(self.t, self.t, self.lengthscale_t).double()
        self.Kzz = rbf_kernel(self.z, self.z, self.lengthscale_z).double()
        self.Kuu = rbf_kernel(self.u, self.u, self.lengthscale_u).double()
        self.Ktz = self.Ktt * self.Kzz

        n = len(self.u)
        self.Ru = (n**(-1)) * torch.sum(self.Kuu, dim=1)

        Ktz_noise = self.Ktz + self.eta * torch.eye(len(self.z), dtype=torch.float64)

        try:
            self.invKtz = torch.inverse(Ktz_noise)
        except:
            self.invKtz = torch.pinverse(Ktz_noise)

        self.C = self.Ktt * (self.Ktz @ self.invKtz @ self.Kuu @ self.invKtz @ self.Ktz) + self.lambda_val * torch.eye(len(self.z), dtype=torch.float64)

        self.invC = torch.inverse(self.C).double()

    def predict(self, t_new, standardized=True):
        if not standardized:
            t_new_std = (t_new - self.t_mean) / self.t_std
        else:
            t_new_std = t_new

        t_new_std = t_new_std.double()
        Kt = rbf_kernel(t_new_std.reshape(1, -1), self.t, self.lengthscale_t)

        pred_std = (Kt * (self.Ru @ self.invKtz @ self.Ktz)) @ self.invC @ self.y

        if not standardized:
            pred = unstandardize_data(pred_std, self.y_mean, self.y_std)
            return pred.squeeze()
        else:
            return pred_std.squeeze()

    def compute_true_structural_function(self, t_values, n_mc=10000, standardized=True):
        if standardized:
            t_original = unstandardize_data(t_values, self.t_mean, self.t_std)
        else:
            t_original = t_values

        t_original = t_original.double()
        torch.manual_seed(42)

        h_orig = torch.zeros_like(t_original, dtype=torch.float64)

        xi_mc = torch.rand(n_mc, dtype=torch.float64) * 10
        epsilon_3_mc = torch.randn(n_mc, dtype=torch.float64)
        epsilon_5_mc = torch.randn(n_mc, dtype=torch.float64)

        g_xi_mc = g_func(xi_mc)
        W_mc = 7 * g_xi_mc + 45 + epsilon_3_mc

        for i, t in enumerate(t_original):
            exp_term = torch.exp((W_mc - t) / 10)
            min_term = torch.minimum(exp_term, torch.tensor(2.0, dtype=torch.float64))
            Y_t = t * min_term - 5 * g_xi_mc + epsilon_5_mc
            h_orig[i] = torch.mean(Y_t)

        if standardized:
            h_std = (h_orig - self.y_mean) / self.y_std
            return h_std
        else:
            return h_orig

    def evaluate_predictions(self, t_test, h_true, standardized=False):
        predictions = []

        for t in t_test:
            pred = self.predict(t, standardized=standardized)
            predictions.append(pred.item())

        predictions = torch.tensor(predictions, dtype=torch.float64)

        errors = torch.abs(predictions - h_true)
        mse_orig = torch.mean((predictions - h_true)**2)

        predictions_std = (predictions - self.y_mean) / self.y_std
        h_true_std = (h_true - self.y_mean) / self.y_std
        mse_std = torch.mean((predictions_std - h_true_std)**2)

        return mse_orig.item(), mse_std.item(), predictions, errors

def cross_validate_lambda(z, u, t, y, t_mean, t_std, y_mean, y_std,
                         lengthscale_t, lengthscale_u, lengthscale_z,
                         lambda_candidates=None, n_folds=5, eta=0.1, seed=42, verbose=False):
    if lambda_candidates is None:
        lambda_candidates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    n_samples = len(y)
    np.random.seed(seed)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    cv_scores = {l: [] for l in lambda_candidates}

    if verbose:
        print(f"Cross-validation with {n_folds} folds...")
        print(f"Lambda candidates: {lambda_candidates}")

    for lambda_val in lambda_candidates:
        if verbose:
            print(f"\nEvaluating lambda = {lambda_val:.3f}")

        fold_scores = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
            z_train, z_val = z[train_idx], z[val_idx]
            u_train, u_val = u[train_idx], u[val_idx]
            t_train, t_val = t[train_idx], t[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            t_train_std, t_train_mean, t_train_std_val = standardize_data(t_train)
            y_train_std, y_train_mean, y_train_std_val = standardize_data(y_train)

            t_val_std = (t_val - t_train_mean) / t_train_std_val
            y_val_std = (y_val - y_train_mean) / y_train_std_val

            gp = ProxyGaussianProcess(
                lengthscale_t=lengthscale_t,
                lengthscale_u=lengthscale_u,
                lengthscale_z=lengthscale_z,
                lambda_val=lambda_val,
                eta=eta
            )

            gp.set_data(
                z=z_train,
                u=u_train,
                t=t_train_std,
                y=y_train_std,
                t_mean=t_train_mean,
                t_std=t_train_std_val,
                y_mean=y_train_mean,
                y_std=y_train_std_val
            )

            mse_val = 0.0
            for i, t_point in enumerate(t_val_std):
                pred = gp.predict(t_point, standardized=True)
                true = y_val_std[i]
                mse_val += (pred.item() - true.item()) ** 2
            mse_val /= len(t_val)
            fold_scores.append(mse_val)

            if verbose and ((fold_idx + 1) % 2 == 0 or (fold_idx + 1) == n_folds):
                print(f"  Fold {fold_idx+1}/{n_folds}: MSE = {mse_val:.4f}")

        avg_mse = np.mean(fold_scores)
        std_mse = np.std(fold_scores)
        cv_scores[lambda_val] = {
            'mean_mse': avg_mse,
            'std_mse': std_mse,
            'fold_scores': fold_scores
        }

        if verbose:
            print(f"  Average MSE for lambda={lambda_val:.3f}: {avg_mse:.4f} (±{std_mse:.4f})")

    best_lambda = min(cv_scores.keys(), key=lambda l: cv_scores[l]['mean_mse'])

    if verbose:
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION RESULTS")
        print("=" * 70)
        for l in lambda_candidates:
            mean_mse = cv_scores[l]['mean_mse']
            std_mse = cv_scores[l]['std_mse']
            marker = " <-- BEST" if l == best_lambda else ""
            print(f"lambda={l:.3f}: MSE = {mean_mse:.4f} (±{std_mse:.4f}){marker}")
        print("=" * 70)

    return best_lambda, cv_scores

def set_seed(seed=442):
    torch.manual_seed(seed)
    np.random.seed(seed)

def run_single_experiment(experiment_seed, params):
    n_samples = params['n_samples']
    use_median_heuristic = params['use_median_heuristic']
    lengthscale_t = params['lengthscale_t']
    lengthscale_u = params['lengthscale_u']
    lengthscale_z = params['lengthscale_z']
    eta = params['eta']
    use_cross_validation = params['use_cross_validation']
    lambda_candidates = params['lambda_candidates']
    n_folds = params['n_folds']
    manual_lambda = params['manual_lambda']
    t_test_start = params['t_test_start']
    t_test_end = params['t_test_end']
    n_test_points = params['n_test_points']
    verbose = params.get('verbose', False)

    set_seed(experiment_seed)

    Z_std, U_std, T_std, Y_std, T_mean, T_std_val, Y_mean, Y_std_val = generate_data(
        n_samples=n_samples, seed=experiment_seed
    )

    if use_median_heuristic:
        lengthscale_t = median_heuristic(T_std).item()
        lengthscale_u = median_heuristic(U_std).item()
        lengthscale_z = median_heuristic(Z_std).item()

    final_lambda = None
    if use_cross_validation:
        final_lambda, _ = cross_validate_lambda(
            z=Z_std,
            u=U_std,
            t=T_std,
            y=Y_std,
            t_mean=T_mean,
            t_std=T_std_val,
            y_mean=Y_mean,
            y_std=Y_std_val,
            lengthscale_t=lengthscale_t,
            lengthscale_u=lengthscale_u,
            lengthscale_z=lengthscale_z,
            lambda_candidates=lambda_candidates,
            n_folds=n_folds,
            eta=eta,
            seed=experiment_seed,
            verbose=verbose
        )
    else:
        final_lambda = manual_lambda

    gp = ProxyGaussianProcess(
        lengthscale_t=lengthscale_t,
        lengthscale_u=lengthscale_u,
        lengthscale_z=lengthscale_z,
        lambda_val=final_lambda,
        eta=eta
    )

    gp.set_data(
        z=Z_std,
        u=U_std,
        t=T_std,
        y=Y_std,
        t_mean=T_mean,
        t_std=T_std_val,
        y_mean=Y_mean,
        y_std=Y_std_val
    )

    t_test_orig = torch.linspace(t_test_start, t_test_end, n_test_points, dtype=torch.float64)
    h_true_orig = gp.compute_true_structural_function(t_test_orig, n_mc=10000, standardized=False)

    mse_orig, _, _, _ = gp.evaluate_predictions(t_test_orig, h_true_orig, standardized=False)

    return mse_orig, final_lambda

def run_repeated_experiments(N=25, base_seed=45, show_progress=True, **params):
    mse_vector = []
    lambda_vector = []

    for i in range(N):
        experiment_seed = base_seed + i
        if show_progress:
            print(f"Running experiment {i+1}/{N}...")

        mse_orig, final_lambda = run_single_experiment(experiment_seed, params)

        mse_vector.append(mse_orig)
        lambda_vector.append(final_lambda)

        if show_progress:
            print(f"  Experiment {i+1}: MSE = {mse_orig:.4f}, Lambda = {final_lambda:.4f}")

    mse_array = np.array(mse_vector)
    mse_mean = np.mean(mse_array)
    mse_std = np.std(mse_array, ddof=1)
    mse_se = mse_std / np.sqrt(N)

    results_dict = {
        'mse_vector': mse_vector,
        'mse_mean': mse_mean,
        'mse_std': mse_std,
        'mse_se': mse_se,
        'lambda_vector': lambda_vector,
        'lambda_mean': np.mean(lambda_vector) if params['use_cross_validation'] else params['manual_lambda'],
        'lambda_std': np.std(lambda_vector, ddof=1) if params['use_cross_validation'] else 0,
        'N': N,
        'params': params
    }

    return mse_vector, mse_mean, mse_se, results_dict

if __name__ == "__main__":
    n_samples = 200
    base_seed = 42
    N = 25
    use_median_heuristic = True
    lengthscale_t = 1.0
    lengthscale_u = 1.0
    lengthscale_z = 1.0
    eta = 0.1
    use_cross_validation = True
    n_folds = 5
    lambda_candidates = [0.001, 0.01, 0.1, 1.0]
    manual_lambda = 0.5
    t_test_start = 10
    t_test_end = 40
    n_test_points = 300
    verbose = False
    show_progress = True

    params = {
        'n_samples': n_samples,
        'use_median_heuristic': use_median_heuristic,
        'lengthscale_t': lengthscale_t,
        'lengthscale_u': lengthscale_u,
        'lengthscale_z': lengthscale_z,
        'eta': eta,
        'use_cross_validation': use_cross_validation,
        'lambda_candidates': lambda_candidates if use_cross_validation else None,
        'n_folds': n_folds if use_cross_validation else None,
        'manual_lambda': manual_lambda if not use_cross_validation else None,
        't_test_start': t_test_start,
        't_test_end': t_test_end,
        'n_test_points': n_test_points,
        'verbose': verbose
    }

    print("=" * 70)
    print("REPEATED EXPERIMENTS: PROXY GAUSSIAN PROCESS")
    print("=" * 70)
    print(f"Number of experiments: {N}")
    print(f"Data: n_samples={n_samples}, base_seed={base_seed}")
    print(f"Fixed parameters: eta={eta}")
    if use_median_heuristic:
        print(f"Lengthscales: median heuristic")
    else:
        print(f"Lengthscales: t={lengthscale_t}, u={lengthscale_u}, z={lengthscale_z}")

    if use_cross_validation:
        print(f"Lambda selection: {n_folds}-fold cross-validation")
        print(f"Lambda candidates: {lambda_candidates}")
    else:
        print(f"Lambda selection: manual input")
        print(f"Manual lambda: {manual_lambda}")

    print(f"Test data: [{t_test_start}, {t_test_end}] with {n_test_points} points")
    print("=" * 70)

    print(f"\nRunning {N} repeated experiments...")

    mse_vector, mse_mean, mse_se, results_dict = run_repeated_experiments(
        N=N,
        base_seed=base_seed,
        show_progress=show_progress,
        **params
    )

    print("\n" + "=" * 70)
    print("REPEATED EXPERIMENTS RESULTS SUMMARY")
    print("=" * 70)
    print(f"Number of experiments: {N}")
    print(f"MSE Mean: {results_dict['mse_mean']:.4f}")
    print(f"MSE Standard Error: {results_dict['mse_se']:.4f}")
    print(f"MSE Standard Deviation: {results_dict['mse_std']:.4f}")
    print(f"MSE Range: [{np.min(mse_vector):.4f}, {np.max(mse_vector):.4f}]")
    print(f"MSE 95% CI: [{results_dict['mse_mean'] - 1.96*results_dict['mse_se']:.4f}, "
          f"{results_dict['mse_mean'] + 1.96*results_dict['mse_se']:.4f}]")

    if results_dict['params']['use_cross_validation']:
        print(f"\nLambda selection (cross-validation):")
        print(f"  Lambda Mean: {results_dict['lambda_mean']:.4f}")
        print(f"  Lambda Std: {results_dict['lambda_std']:.4f}")
    else:
        print(f"\nLambda selection (manual):")
        print(f"  Lambda: {results_dict['lambda_mean']:.4f}")

    print("=" * 70)

    results = {
        'mse_vector': mse_vector,
        'mse_mean': mse_mean,
        'mse_se': mse_se,
        'results_dict': results_dict
    }