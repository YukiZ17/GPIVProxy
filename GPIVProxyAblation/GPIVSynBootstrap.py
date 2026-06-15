import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


def generate_data(n_samples=100, f_type='sine', seed=None):
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
    alpha = 0.2
    x = normal_dist.cdf((alpha * w + (1 - alpha) * v) / 2)
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

    dist_sq = torch.cdist(X1, X2, p=2) ** 2
    return torch.exp(-0.5 * dist_sq / (lengthscale ** 2))


class GPIV:
    def __init__(self, lengthscale_x=1.0, lengthscale_z=1.0, sigma=0.1, eta=0.1,
                 train_sigma=True, train_eta=False, lx=True, lz=True):
        self.lengthscale_x = torch.tensor(lengthscale_x, dtype=torch.float64, requires_grad=lx)
        self.lengthscale_z = torch.tensor(lengthscale_z, dtype=torch.float64, requires_grad=lz)
        self.sigma = torch.tensor(sigma, dtype=torch.float64, requires_grad=train_sigma)
        self.eta = torch.tensor(eta, dtype=torch.float64, requires_grad=train_eta)

        self.X_train = None
        self.Z_train = None
        self.y_train = None
        self.Kxx = None
        self.Lzz = None
        self.invLzz = None
        self.Lxz = None

    def set_data(self, X, Z, y):
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if len(Z.shape) == 1:
            Z = Z.reshape(-1, 1)
        self.X_train = X
        self.Z_train = Z
        self.y_train = y

    def compute_kernel_matrices(self):
        self.Kxx = rbf_kernel(self.X_train, self.X_train, self.lengthscale_x)
        self.Lzz = rbf_kernel(self.Z_train, self.Z_train, self.lengthscale_z)

        Lzz_noise = self.Lzz + self.eta * torch.eye(len(self.Z_train), dtype=torch.float64)
        try:
            self.invLzz = torch.inverse(Lzz_noise)
        except:
            self.invLzz = torch.pinverse(Lzz_noise)

        self.Lxz = torch.inverse(
            self.Lzz @ self.invLzz @ self.Kxx @ self.invLzz @ self.Lzz +
            (self.sigma ** 2) * torch.eye(len(self.Z_train), dtype=torch.float64)
        )

    def predict_batch(self, X_test):
        if len(X_test.shape) == 1:
            X_test = X_test.reshape(-1, 1)

        Kx = rbf_kernel(X_test, self.X_train, self.lengthscale_x)
        kxx_prime = torch.diag(rbf_kernel(X_test, X_test, self.lengthscale_x))

        means = Kx @ self.invLzz @ self.Lzz @ self.Lxz @ self.y_train.reshape(-1, 1)
        variances = kxx_prime - torch.diag(Kx @ self.invLzz @ self.Lzz @ self.Lxz @ self.Lzz @ self.invLzz @ Kx.T)

        return means.squeeze(), variances

    def negative_log_marginal_likelihood(self):
        n = len(self.y_train)
        K_full = (self.Lzz @ self.invLzz @ self.Kxx @ self.invLzz @ self.Lzz +
                  (self.sigma ** 2) * torch.eye(n, dtype=torch.float64))

        try:
            L = torch.linalg.cholesky(K_full)
            alpha = torch.cholesky_solve(self.y_train.reshape(-1, 1), L)
            data_fit = self.y_train @ alpha.squeeze()
            log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        except RuntimeError:
            K_inv = torch.pinverse(K_full)
            data_fit = self.y_train @ K_inv @ self.y_train
            sign, log_det_val = torch.slogdet(K_full)
            log_det = log_det_val

        return data_fit + log_det

    def optimize_hyperparameters(self, n_iterations=200, lr=0.01, verbose=False):
        params_to_optimize = []
        if self.lengthscale_x.requires_grad:
            params_to_optimize.append(self.lengthscale_x)
        if self.lengthscale_z.requires_grad:
            params_to_optimize.append(self.lengthscale_z)
        if self.sigma.requires_grad:
            params_to_optimize.append(self.sigma)
        if self.eta.requires_grad:
            params_to_optimize.append(self.eta)

        optimizer = Adam(params_to_optimize, lr=lr)

        for i in range(n_iterations):
            optimizer.zero_grad()
            self.compute_kernel_matrices()
            loss = self.negative_log_marginal_likelihood()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                if self.lengthscale_x.requires_grad:
                    self.lengthscale_x.data = torch.clamp(self.lengthscale_x, min=1e-3)
                if self.lengthscale_z.requires_grad:
                    self.lengthscale_z.data = torch.clamp(self.lengthscale_z, min=1e-5)
                if self.sigma.requires_grad:
                    self.sigma.data = torch.clamp(self.sigma, min=1e-6)
                if self.eta.requires_grad:
                    self.eta.data = torch.clamp(self.eta, min=1e-6)


def bootstrap_variance(gp_original, X_test, n_bootstrap=50,
                       train_lengthscale_x=False, train_lengthscale_z=False,
                       train_sigma=True, train_eta=False,
                       n_iterations=50, lr=0.02, verbose=True):
    n_samples = len(gp_original.X_train)
    n_test = len(X_test)

    if len(X_test.shape) == 1:
        X_test = X_test.reshape(-1, 1)

    all_predictions = []

    for b in range(n_bootstrap):
        if verbose and (b + 1) % 20 == 0:
            print(f"  Bootstrap sample {b + 1}/{n_bootstrap}")

        indices = torch.randint(0, n_samples, (n_samples,))
        X_boot = gp_original.X_train[indices]
        Z_boot = gp_original.Z_train[indices]
        y_boot = gp_original.y_train[indices]

        lengthscale_x_boot = median_heuristic(X_boot).item()
        lengthscale_z_boot = median_heuristic(Z_boot).item()

        gp_boot = GPIV(
            lengthscale_x=lengthscale_x_boot,
            lengthscale_z=lengthscale_z_boot,
            sigma=gp_original.sigma.item(),
            eta=gp_original.eta.item(),
            train_sigma=train_sigma,
            train_eta=train_eta,
            lx=train_lengthscale_x,
            lz=train_lengthscale_z
        )

        gp_boot.set_data(X_boot, Z_boot, y_boot)
        gp_boot.compute_kernel_matrices()

        if train_lengthscale_x or train_lengthscale_z or train_sigma or train_eta:
            gp_boot.optimize_hyperparameters(
                n_iterations=n_iterations,
                lr=lr,
                verbose=False
            )

        means_boot, _ = gp_boot.predict_batch(X_test)
        all_predictions.append(means_boot.detach().cpu().numpy())

    all_predictions = np.array(all_predictions)
    bootstrap_means = np.mean(all_predictions, axis=0)
    bootstrap_vars = np.var(all_predictions, axis=0, ddof=1)

    return torch.tensor(bootstrap_means, dtype=torch.float64), \
        torch.tensor(bootstrap_vars, dtype=torch.float64)


def run_bootstrap_gpiv_experiment(seed_head=42, n_experiments=10, n_samples=200,
                                  n_test_points=200, n_bootstrap=50, f_type='sine',
                                  quantile_thresholds=[0.65, 0.75, 0.85],
                                  train_params=['sigma'], eta=1.0,
                                  n_iterations=150, lr=0.02, verbose=True):
    train_lengthscale_x = 'lengthscale_x' in train_params
    train_lengthscale_z = 'lengthscale_z' in train_params
    train_sigma = 'sigma' in train_params
    train_eta = 'eta' in train_params

    all_mse = []
    all_coverage = []
    arc_data = {q: [] for q in quantile_thresholds}
    removal_ratios_list = []

    for exp in range(n_experiments):
        seed = seed_head + exp
        if verbose:
            print(f"Running experiment {exp + 1}/{n_experiments} (seed={seed})...")

        torch.manual_seed(seed)
        np.random.seed(seed)

        x, y, z, _ = generate_data(n_samples, f_type=f_type, seed=seed)

        lengthscale_x = median_heuristic(x).item()
        lengthscale_z = median_heuristic(z).item()

        gp_original = GPIV(
            lengthscale_x=lengthscale_x,
            lengthscale_z=lengthscale_z,
            sigma=1.2,
            eta=eta,
            train_sigma=train_sigma,
            train_eta=train_eta,
            lx=train_lengthscale_x,
            lz=train_lengthscale_z
        )

        gp_original.set_data(x.reshape(-1, 1), z.reshape(-1, 1), y)
        gp_original.compute_kernel_matrices()

        if train_params:
            gp_original.optimize_hyperparameters(
                n_iterations=n_iterations,
                lr=lr,
                verbose=False
            )

        x_test = torch.linspace(0.0, 1.0, n_test_points, dtype=torch.float64)

        if f_type == 'sine':
            h_x_true = 2 * torch.sin(2 * torch.pi * x_test)
        elif f_type == 'log':
            h_x_true = torch.log(torch.abs(16 * x_test - 8) + 1) * torch.sign(x_test - 0.5)
        elif f_type == 'linear':
            h_x_true = 4 * x_test - 2
        elif f_type == 'absolute':
            h_x_true = torch.abs(4 * x_test - 2) - 1

        if verbose:
            print(f"  Running bootstrap with {n_bootstrap} samples...")

        pred_means, pred_vars = bootstrap_variance(
            gp_original,
            x_test.reshape(-1, 1),
            n_bootstrap=n_bootstrap,
            train_lengthscale_x=train_lengthscale_x,
            train_lengthscale_z=train_lengthscale_z,
            train_sigma=train_sigma,
            train_eta=train_eta,
            n_iterations=n_iterations // 3,
            lr=lr,
            verbose=verbose
        )

        se = (h_x_true - pred_means) ** 2

        sorted_indices = torch.argsort(pred_vars, descending=True)
        se_sorted = se[sorted_indices]

        thresholds = {q: torch.quantile(se, q).item() for q in quantile_thresholds}

        n_per_step = max(1, n_test_points // 40)
        total_points = len(se_sorted)

        step_indices = []
        for i in range(0, total_points - n_per_step, n_per_step):
            step_indices.append(i)

        if len(step_indices) == 0 or step_indices[-1] < total_points - n_per_step:
            step_indices.append(total_points - n_per_step)

        accurate_ratios_dict = {}
        for quantile in quantile_thresholds:
            threshold = thresholds[quantile]
            accurate_ratios = []

            for i in step_indices:
                remaining_se = se_sorted[i:]
                accurate_count = torch.sum(remaining_se <= threshold).item()
                accurate_ratio = accurate_count / len(remaining_se)
                accurate_ratios.append(accurate_ratio)

            accurate_ratios_dict[quantile] = accurate_ratios

        if exp == 0:
            removal_ratios = [i / total_points for i in step_indices]
            removal_ratios_list = removal_ratios

        z_score = norm.ppf(0.975)
        pred_stds = torch.sqrt(torch.clamp(pred_vars, min=1e-10))
        lower = pred_means - z_score * pred_stds
        upper = pred_means + z_score * pred_stds
        coverage = torch.mean(((h_x_true >= lower) & (h_x_true <= upper)).float()).item()

        mse = torch.mean(se).item()
        all_mse.append(mse)
        all_coverage.append(coverage)

        for quantile in quantile_thresholds:
            arc_data[quantile].append(accurate_ratios_dict[quantile])

        if verbose:
            print(f"  MSE: {mse:.6f}, Coverage: {coverage:.4f}")

    mse_mean = np.mean(all_mse)
    mse_std = np.std(all_mse)
    coverage_mean = np.mean(all_coverage)
    coverage_std = np.std(all_coverage)

    mean_accuracies_dict = {}
    std_accuracies_dict = {}

    for quantile in quantile_thresholds:
        accuracies_array = np.array(arc_data[quantile])
        mean_accuracies_dict[quantile] = np.mean(accuracies_array, axis=0)
        std_accuracies_dict[quantile] = np.std(accuracies_array, axis=0)

    results = {
        'config': {
            'seed_head': seed_head,
            'n_experiments': n_experiments,
            'n_samples': n_samples,
            'n_test_points': n_test_points,
            'n_bootstrap': n_bootstrap,
            'f_type': f_type,
            'quantile_thresholds': quantile_thresholds,
            'train_params': train_params,
            'eta': eta,
            'n_iterations': n_iterations,
            'lr': lr
        },
        'removal_ratios': removal_ratios_list,
        'mean_accuracies_dict': mean_accuracies_dict,
        'std_accuracies_dict': std_accuracies_dict,
        'mse_mean': mse_mean,
        'mse_std': mse_std,
        'coverage_mean': coverage_mean,
        'coverage_std': coverage_std,
        'all_mse': all_mse,
        'all_coverage': all_coverage,
        'quantile_thresholds': quantile_thresholds,
        'n_experiments': n_experiments,
        'n_bootstrap': n_bootstrap
    }

    return results


def plot_bootstrap_results(results, save_path=None):
    config = results['config']
    removal_ratios = results['removal_ratios']
    quantile_thresholds = results['quantile_thresholds']

    plt.figure(figsize=(12, 8))

    colors = ['green', 'blue', 'red', 'orange', 'purple']

    for idx, quantile in enumerate(quantile_thresholds):
        if idx >= len(colors):
            color = 'gray'
        else:
            color = colors[idx]

        mean_accuracies = results['mean_accuracies_dict'][quantile]
        std_accuracies = results['std_accuracies_dict'][quantile]

        confidence_interval = 1.96 * std_accuracies / np.sqrt(results['n_experiments'])

        plt.plot(removal_ratios, mean_accuracies, '-', color=color,
                 linewidth=3, label=f'Q{quantile}')
        plt.fill_between(removal_ratios,
                         mean_accuracies - confidence_interval,
                         mean_accuracies + confidence_interval,
                         alpha=0.3, color=color)

    plt.xlabel('Proportion of High-Variance Data Removed', fontsize=12)
    plt.ylabel('Proportion of Accurate Predictions (SE ≤ Δ)', fontsize=12)

    title = f"GPIV Bootstrap ARC - {config['f_type']} function\n"
    title += f"n_samples={config['n_samples']}, n_bootstrap={config['n_bootstrap']}, "
    title += f"n_experiments={config['n_experiments']}"
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.45, 1.05)
    plt.legend(fontsize=10, loc='lower left')

    stats_text = f"MSE: {results['mse_mean']:.6f} ± {results['mse_std']:.6f}\n"
    stats_text += f"Coverage: {results['coverage_mean']:.4f} ± {results['coverage_std']:.4f}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


def main():
    seed_head = 42
    n_experiments = 25
    n_samples = 200
    n_test_points = 200
    n_bootstrap = 15

    f_type = 'linear'
    quantile_thresholds = [0.65, 0.75, 0.85]

    train_params = ['sigma']
    eta = 1.0
    n_iterations = 100
    lr = 0.02

    verbose = True

    print("=" * 60)
    print("GPIV Bootstrap Experiment")
    print("=" * 60)
    print(f"Seed head: {seed_head}")
    print(f"Number of experiments: {n_experiments}")
    print(f"Sample size: {n_samples}")
    print(f"Number of bootstrap samples: {n_bootstrap}")
    print(f"Function type: {f_type}")
    print(f"Quantile thresholds: {quantile_thresholds}")
    print(f"Parameters to optimize: {train_params}")
    print(f"Eta value: {eta}")
    print(f"Training iterations: {n_iterations}")
    print(f"Learning rate: {lr}")
    print("=" * 60)

    import time
    start_time = time.time()

    results = run_bootstrap_gpiv_experiment(
        seed_head=seed_head,
        n_experiments=n_experiments,
        n_samples=n_samples,
        n_test_points=n_test_points,
        n_bootstrap=n_bootstrap,
        f_type=f_type,
        quantile_thresholds=quantile_thresholds,
        train_params=train_params,
        eta=eta,
        n_iterations=n_iterations,
        lr=lr,
        verbose=verbose
    )

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Average time per experiment: {total_time / n_experiments:.2f} seconds")

    print("\n" + "=" * 60)
    print("EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"MSE: {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
    print(f"Coverage (95% CI): {results['coverage_mean']:.4f} ± {results['coverage_std']:.4f}")

    print("\n--- ARC VECTORS ---")
    print(f"Removal ratios (x-axis): {results['removal_ratios']}")
    print("\nMean accuracy vectors for each quantile:")
    for quantile in results['quantile_thresholds']:
        mean_acc = results['mean_accuracies_dict'][quantile]
        print(f"\nQ{quantile} (length {len(mean_acc)}):")
        print(f"  Values: {mean_acc}")
        print(f"  Initial (0% removed): {mean_acc[0]:.4f}")
        print(f"  Final ({results['removal_ratios'][-1] * 100:.1f}% removed): {mean_acc[-1]:.4f}")
        print(f"  Improvement: {(mean_acc[-1] - mean_acc[0]) * 100:.1f}%")

    print("\nStandard deviation vectors for each quantile:")
    for quantile in results['quantile_thresholds']:
        std_acc = results['std_accuracies_dict'][quantile]
        print(f"\nQ{quantile} (length {len(std_acc)}):")
        print(f"  Values: {std_acc}")
        print(f"  Mean std: {np.mean(std_acc):.4f}")
        print(f"  Max std: {np.max(std_acc):.4f}")

    print("\n--- DETAILED RESULTS FROM EACH EXPERIMENT ---")
    print("\nMSE values from all experiments:")
    for i, mse in enumerate(results['all_mse']):
        print(f"  Experiment {i + 1}: {mse:.6f}")

    print("\nCoverage values from all experiments:")
    for i, coverage in enumerate(results['all_coverage']):
        print(f"  Experiment {i + 1}: {coverage:.4f}")

    print("\n" + "=" * 60)
    print("PLOTTING ARC CURVE")
    print("=" * 60)
    plot_bootstrap_results(results, save_path="gpiv_bootstrap_results.png")

    print("\n" + "=" * 60)
    print("RESULTS DICTIONARY SUMMARY")
    print("=" * 60)
    print("Results dictionary contains the following keys:")
    print("  - 'config': Configuration parameters")
    print("  - 'removal_ratios': X-axis values for ARC plot (length: {})".format(len(results['removal_ratios'])))
    print("  - 'mean_accuracies_dict': Mean accuracy for each quantile")
    for quantile in results['quantile_thresholds']:
        print(f"      Q{quantile} vector length: {len(results['mean_accuracies_dict'][quantile])}")
    print("  - 'std_accuracies_dict': Std of accuracy for each quantile")
    for quantile in results['quantile_thresholds']:
        print(f"      Q{quantile} vector length: {len(results['std_accuracies_dict'][quantile])}")
    print("  - 'mse_mean', 'mse_std': MSE statistics")
    print("  - 'coverage_mean', 'coverage_std': Coverage statistics")
    print("  - 'all_mse': All MSE values (length: {})".format(len(results['all_mse'])))
    print("  - 'all_coverage': All coverage values (length: {})".format(len(results['all_coverage'])))
    print("=" * 60)

    print("\nVECTOR LENGTH CONSISTENCY CHECK:")
    for quantile in results['quantile_thresholds']:
        mean_len = len(results['mean_accuracies_dict'][quantile])
        std_len = len(results['std_accuracies_dict'][quantile])
        removal_len = len(results['removal_ratios'])
        print(f"  Q{quantile}: mean_acc length={mean_len}, std_acc length={std_len}, removal_ratios length={removal_len}")
        if mean_len == std_len == removal_len:
            print(f"    ✓ All vectors for Q{quantile} have consistent length")
        else:
            print(f"    ✗ Vector length mismatch for Q{quantile}!")

    return results


if __name__ == "__main__":
    results = main()