import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from scipy.stats import norm
from scipy.integrate import simpson


def generate_data(n_samples=100, f_type='absolute', alpha=0.5, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    covariance_matrix = torch.tensor([
        [1.0, 0.8, 0.5],
        [0.8, 1.0, 0.0],
        [0.5, 0.0, 1.0]
    ], dtype=torch.float64)

    mean = torch.zeros(3, dtype=torch.float64)
    multivariate_normal = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    samples = multivariate_normal.sample((n_samples,))

    e, v, w = samples[:, 0], samples[:, 1], samples[:, 2]

    normal_dist = torch.distributions.Normal(0, 1)
    x = normal_dist.cdf((alpha * w + (1 - alpha) * v) )
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
        self.X_train2 = None
        self.Z_train2 = None
        self.y_train2 = None

        self.Kxx = None
        self.Lzz2 = None
        self.Lzz = None
        self.invLzz = None
        self.Lxz2 = None

    def set_data(self, X, Z, y, split_ratio_1=0.5, split_ratio_2=0.5, shuffle=False, random_state=None):
        if shuffle:
            rng = np.random.default_rng(random_state)
            indices = rng.permutation(len(X))
            X = X[indices]
            Z = Z[indices]
            y = y[indices]

        n_total = len(X)
        n1 = int(n_total * split_ratio_1)
        n2 = int(n_total * split_ratio_2)
        if n1 + n2 > n_total:
            n2 = n_total - n1

        self.X_train = X[:n1]
        self.Z_train = Z[:n1]
        self.y_train = y[:n1]

        self.X_train2 = X[n2:]
        self.Z_train2 = Z[n2:]
        self.y_train2 = y[n2:]

    def compute_kernel_matrices(self):
        self.Kxx = rbf_kernel(self.X_train, self.X_train, self.lengthscale_x)
        self.Lzz = rbf_kernel(self.Z_train, self.Z_train, self.lengthscale_z)
        self.Lzz2 = rbf_kernel(self.Z_train, self.Z_train2, self.lengthscale_z)

        Lzz_noise = self.Lzz + self.eta * torch.eye(len(self.Z_train), dtype=torch.float64)

        try:
            self.invLzz = torch.inverse(Lzz_noise)
        except:
            self.invLzz = torch.pinverse(Lzz_noise)

        self.Lxz2 = torch.inverse(
            self.Lzz2.t() @ self.invLzz @ self.Kxx @ self.invLzz @ self.Lzz2 +
            (self.sigma**2) * torch.eye(len(self.Z_train2), dtype=torch.float64)
        )

    def posterior_mean(self, x_new):
        Kx = rbf_kernel(x_new.reshape(1, -1), self.X_train, self.lengthscale_x)
        mean = Kx @ self.invLzz @ self.Lzz2 @ self.Lxz2 @ self.y_train2.reshape(-1, 1)
        return mean.squeeze()

    def posterior_variance(self, x_new):
        Kx = rbf_kernel(x_new.reshape(1, -1), self.X_train, self.lengthscale_x)
        kxx_prime = rbf_kernel(x_new.reshape(1, -1), x_new.reshape(1, -1), self.lengthscale_x)

        variance = kxx_prime - Kx @ self.invLzz @ self.Lzz2 @ self.Lxz2 @ self.Lzz2.t() @ self.invLzz @ Kx.T
        return variance.squeeze()

    def predict(self, x_new):
        mean = self.posterior_mean(x_new)
        variance = self.posterior_variance(x_new)
        return mean, variance

    def negative_log_marginal_likelihood(self):
        n = len(self.y_train2)

        K_full = (self.Lzz2.t() @ self.invLzz @ self.Kxx @ self.invLzz @ self.Lzz2 +
                 (self.sigma**2) * torch.eye(n, dtype=torch.float64))

        try:
            L = torch.linalg.cholesky(K_full)
            alpha = torch.cholesky_solve(self.y_train2.reshape(-1, 1), L)
            data_fit = self.y_train2 @ alpha.squeeze()
            log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        except RuntimeError:
            K_inv = torch.pinverse(K_full)
            data_fit = self.y_train2 @ K_inv @ self.y_train2
            sign, log_det_val = torch.slogdet(K_full)
            log_det = log_det_val

        return data_fit + log_det

    def optimize_hyperparameters(self, n_iterations=200, lr=0.01, verbose=True):
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

        history = {
            'losses': [], 'lengthscales_x': [], 'lengthscales_z': [],
            'sigmas': [], 'etas': []
        }

        if verbose:
            print("Hyperparameter optimization starting...")
            print(f"Initial: lengthscale_x={self.lengthscale_x.item():.3f}, "
                  f"lengthscale_z={self.lengthscale_z.item():.3f}, "
                  f"sigma={self.sigma.item():.3f}, eta={self.eta.item():.3f}")

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

            history['losses'].append(loss.item())
            history['lengthscales_x'].append(self.lengthscale_x.item())
            history['lengthscales_z'].append(self.lengthscale_z.item())
            history['sigmas'].append(self.sigma.item())
            history['etas'].append(self.eta.item())

            if verbose and (i + 1) % 50 == 0:
                print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.3f}, "
                      f"lengthscale_x={self.lengthscale_x.item():.3f}, "
                      f"lengthscale_z={self.lengthscale_z.item():.3f}, "
                      f"sigma={self.sigma.item():.3f}")

        if verbose:
            print("Optimization completed.")
            print(f"Final: lengthscale_x={self.lengthscale_x.item():.3f}, "
                  f"lengthscale_z={self.lengthscale_z.item():.3f}, "
                  f"sigma={self.sigma.item():.3f}, eta={self.eta.item():.3f}")

        return history

    def compute_posterior_mse(self, f_type='absolute', n_points=200, x_range=(0,1)):
        x_start, x_end = x_range
        x_test = torch.linspace(x_start, x_end, n_points, dtype=torch.float64)

        if f_type == 'sine':
            h_x = 2 * torch.sin(2 * torch.pi * x_test)
        elif f_type == 'log':
            h_x = torch.log(torch.abs(16 * x_test - 8) + 1) * torch.sign(x_test - 0.5)
        elif f_type == 'linear':
            h_x = 4 * x_test - 2
        elif f_type == 'absolute':
            h_x = torch.abs(4 * x_test - 2) - 1
        else:
            raise ValueError(f"Unknown function type: {f_type}")

        means = []
        for x in x_test:
            mean, _ = self.predict(x)
            means.append(mean.item())

        means = torch.tensor(means, dtype=torch.float64)
        mse = torch.mean((h_x - means) ** 2).item()

        return mse, x_test, h_x, means

    def compute_coverage(self, x_test, h_x_true, confidence=0.95):
        pred_means = []
        pred_vars = []

        for x in x_test:
            mean, var = self.predict(x)
            pred_means.append(mean.item())
            pred_vars.append(var.item())

        pred_means = torch.tensor(pred_means, dtype=torch.float64)
        pred_vars = torch.tensor(pred_vars, dtype=torch.float64)

        z_score = norm.ppf((1 + confidence) / 2)
        pred_stds = torch.sqrt(torch.clamp(pred_vars, min=1e-10))

        lower = pred_means - z_score * pred_stds
        upper = pred_means + z_score * pred_stds

        coverage = torch.mean(((h_x_true >= lower) & (h_x_true <= upper)).float()).item()

        return coverage

    def visualize_posterior(self, f_type='absolute', n_points=200, x_range=(0,1)):
        mse, x_test, h_x, means = self.compute_posterior_mse(f_type, n_points, x_range)

        print(f"MSE on {n_points} points: {mse:.4f}")

        variances = []
        for x in x_test:
            _, var = self.predict(x)
            variances.append(var.item())

        variances = torch.tensor(variances, dtype=torch.float64)
        stds = torch.sqrt(variances)

        plt.figure(figsize=(12, 8))

        plt.plot(x_test.numpy(), means.numpy(), 'b-', linewidth=2, label='Posterior Mean')

        plt.fill_between(x_test.numpy(),
                        (means - 2 * stds).numpy(),
                        (means + 2 * stds).numpy(),
                        alpha=0.2, color='blue', label='95% CI')

        plt.plot(x_test.numpy(), h_x.numpy(), 'orange', linewidth=2, label=f'True h(x): {f_type}')

        if self.X_train is not None:
            plt.scatter(self.X_train2.numpy(), self.y_train2.numpy(),
                       color='red', s=20, alpha=0.5, label='Training Data')

        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title(f'GPIV Posterior Distribution: {f_type} function\n'
                 f'MSE: {mse:.4f}, lengthscale_x={self.lengthscale_x.item():.3f}, '
                 f'lengthscale_z={self.lengthscale_z.item():.3f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(x_range[0], x_range[1])
        plt.tight_layout()
        plt.show()

        return x_test, variances, (h_x - means) ** 2

    def visualize_optimization(self, history):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(history['losses'])
        axes[0, 0].set_xlabel('Iterations')
        axes[0, 0].set_ylabel('Negative Log Likelihood')
        axes[0, 0].set_title('Loss Curve')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(history['lengthscales_x'])
        axes[0, 1].set_xlabel('Iterations')
        axes[0, 1].set_ylabel('Lengthscale')
        axes[0, 1].set_title('Lengthscale X')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(history['lengthscales_z'])
        axes[1, 0].set_xlabel('Iterations')
        axes[1, 0].set_ylabel('Lengthscale')
        axes[1, 0].set_title('Lengthscale Z')
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(history['sigmas'])
        axes[1, 1].set_xlabel('Iterations')
        axes[1, 1].set_ylabel('Sigma')
        axes[1, 1].set_title('Noise Sigma')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('GPIV Hyperparameter Optimization History', fontsize=14)
        plt.tight_layout()
        plt.show()


def bootstrap_gp_predictions(gp, X_train1, Z_train1, y_train1,
                             X_train2, Z_train2, y_train2,
                             x_test, n_bootstrap=100, random_seed=None):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    n_test = len(x_test)
    bootstrap_means = np.zeros((n_test, n_bootstrap))

    lengthscale_x = gp.lengthscale_x.item()
    lengthscale_z = gp.lengthscale_z.item()
    sigma = gp.sigma.item()
    eta = gp.eta.item()

    X_train2_t = torch.tensor(X_train2, dtype=torch.float64).reshape(-1, 1)
    Z_train2_t = torch.tensor(Z_train2, dtype=torch.float64).reshape(-1, 1)
    y_train2_t = torch.tensor(y_train2, dtype=torch.float64)

    for b in range(n_bootstrap):
        indices = np.random.choice(len(X_train1), len(X_train1), replace=True)
        X1_boot = X_train1[indices]
        Z1_boot = Z_train1[indices]
        y1_boot = y_train1[indices]

        X1_boot_t = torch.tensor(X1_boot, dtype=torch.float64).reshape(-1, 1)
        Z1_boot_t = torch.tensor(Z1_boot, dtype=torch.float64).reshape(-1, 1)
        y1_boot_t = torch.tensor(y1_boot, dtype=torch.float64)

        gp_boot = GPIV(
            lengthscale_x=lengthscale_x,
            lengthscale_z=lengthscale_z,
            sigma=sigma,
            eta=eta,
            train_sigma=True,
            train_eta=False,
            lx=False,
            lz=False
        )
        gp_boot.X_train = X1_boot_t
        gp_boot.Z_train = Z1_boot_t
        gp_boot.y_train = y1_boot_t
        gp_boot.X_train2 = X_train2_t
        gp_boot.Z_train2 = Z_train2_t
        gp_boot.y_train2 = y_train2_t

        gp_boot.compute_kernel_matrices()

        for i, xv in enumerate(x_test):
            mean, _ = gp_boot.predict(xv)
            bootstrap_means[i, b] = mean.item()

    return bootstrap_means


def run_gpiv_experiment_with_rejection(f_type='absolute', n_samples=300,
                                       n_test_points=200, seed=42, quantile_threshold=0.75,
                                       n_iterations=150, lr=0.02,
                                       train_lengthscale_x=True, train_lengthscale_z=True,
                                       train_sigma=True, eta=1.0, alpha=0.5,
                                       split_ratio_1=0.5, split_ratio_2=0.5,
                                       x_range=(0,1), n_bootstrap=0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    x, y, z, h_true = generate_data(n_samples, f_type=f_type, alpha=alpha, seed=seed)

    lengthscale_x = median_heuristic(x).item()
    lengthscale_z = median_heuristic(z).item()

    gp = GPIV(
        lengthscale_x=lengthscale_x,
        lengthscale_z=lengthscale_z,
        sigma=1.2,
        eta=eta,
        train_sigma=train_sigma,
        train_eta=False,
        lx=train_lengthscale_x,
        lz=train_lengthscale_z
    )

    gp.set_data(x.reshape(-1, 1), z.reshape(-1, 1), y,
                split_ratio_1=split_ratio_1, split_ratio_2=split_ratio_2,
                shuffle=True, random_state=seed)
    gp.compute_kernel_matrices()

    history = gp.optimize_hyperparameters(n_iterations=n_iterations, lr=lr, verbose=False)

    x_start, x_end = x_range
    x_test = torch.linspace(x_start, x_end, n_test_points, dtype=torch.float64)

    if f_type == 'sine':
        h_x_true = 2 * torch.sin(2 * torch.pi * x_test)
    elif f_type == 'log':
        h_x_true = torch.log(torch.abs(16 * x_test - 8) + 1) * torch.sign(x_test - 0.5)
    elif f_type == 'linear':
        h_x_true = 4 * x_test - 2
    elif f_type == 'absolute':
        h_x_true = torch.abs(4 * x_test - 2) - 1

    pred_means = []
    se = []
    var = []

    for x_val in x_test:
        mean, variance = gp.predict(x_val)
        pred_means.append(mean.item())
        se.append((h_x_true[x_test == x_val] - mean.item()) ** 2)
        var.append(variance.item())

    pred_means = torch.tensor(pred_means, dtype=torch.float64)
    se = torch.tensor(se, dtype=torch.float64)
    var = torch.tensor(var, dtype=torch.float64)

    sorted_indices = torch.argsort(var, descending=True)
    se_sorted = se[sorted_indices]
    var_sorted = var[sorted_indices]

    n_per_step = max(1, n_test_points // 40)
    total_points = len(se_sorted)

    threshold = torch.quantile(se, quantile_threshold).item()

    accurate_ratios = []
    removed_ratios = []

    for i in range(0, total_points - n_per_step, n_per_step):
        remaining_se = se_sorted[i:]
        accurate_count = torch.sum(remaining_se <= threshold).item()
        accurate_ratio = accurate_count / len(remaining_se)
        accurate_ratios.append(accurate_ratio)
        removed_ratios.append(i / total_points)

    coverage = gp.compute_coverage(x_test, h_x_true, confidence=0.95)

    bootstrap_variance = None
    total_variance = var.clone()
    accurate_ratios_total = accurate_ratios[:]
    removed_ratios_total = removed_ratios[:]
    coverage_total = coverage
    auc = 0.0

    if n_bootstrap > 0:
        print(f"  Performing bootstrap with {n_bootstrap} resamples...")
        X1 = gp.X_train.squeeze().numpy()
        Z1 = gp.Z_train.squeeze().numpy()
        y1 = gp.y_train.numpy()
        X2 = gp.X_train2.squeeze().numpy()
        Z2 = gp.Z_train2.squeeze().numpy()
        y2 = gp.y_train2.numpy()
        x_test_np = x_test.numpy()

        boot_means = bootstrap_gp_predictions(
            gp, X1, Z1, y1, X2, Z2, y2,
            x_test, n_bootstrap=n_bootstrap, random_seed=seed
        )
        bootstrap_variance = np.var(boot_means, axis=1, ddof=1)
        bootstrap_variance = torch.tensor(bootstrap_variance, dtype=torch.float64)

        total_variance = var + bootstrap_variance

        print(var)
        print(bootstrap_variance)

        sorted_indices_total = torch.argsort(total_variance, descending=True)
        se_sorted_total = se[sorted_indices_total]
        var_sorted_total = total_variance[sorted_indices_total]

        accurate_ratios_total = []
        removed_ratios_total = []
        for i in range(0, total_points - n_per_step, n_per_step):
            remaining_se = se_sorted_total[i:]
            accurate_count = torch.sum(remaining_se <= threshold).item()
            accurate_ratio = accurate_count / len(remaining_se)
            accurate_ratios_total.append(accurate_ratio)
            removed_ratios_total.append(i / total_points)

        auc = simpson(accurate_ratios_total, removed_ratios_total)

        z_score = norm.ppf(0.975)
        pred_stds_total = torch.sqrt(torch.clamp(total_variance, min=1e-10))
        lower_total = pred_means - z_score * pred_stds_total
        upper_total = pred_means + z_score * pred_stds_total
        coverage_total = torch.mean(((h_x_true >= lower_total) & (h_x_true <= upper_total)).float()).item()

    mse = torch.mean(se).item()

    rejection_curve = {
        'accurate_ratios': accurate_ratios,
        'removed_ratios': removed_ratios,
        'threshold': threshold,
        'mse': mse,
        'coverage': coverage,
        'sorted_variances': var_sorted.numpy(),
        'sorted_errors': se_sorted.numpy(),
        'bootstrap_variance': bootstrap_variance.numpy() if bootstrap_variance is not None else None,
        'total_variance': total_variance.numpy(),
        'accurate_ratios_total': accurate_ratios_total,
        'removed_ratios_total': removed_ratios_total,
        'auc': auc,
        'coverage_total': coverage_total,
        'n_bootstrap': n_bootstrap
    }

    return mse, coverage, rejection_curve, gp


def run_gpiv_multiple_experiments(n_experiments=25, f_type='absolute', n_samples=300,
                                  n_test_points=200, quantile_threshold=0.75,
                                  n_iterations=150, lr=0.02,
                                  train_lengthscale_x=True, train_lengthscale_z=True,
                                  train_sigma=True, eta=1.0, alpha=0.5,
                                  split_ratio_1=0.5, split_ratio_2=0.5,
                                  x_range=(0,1), seed_head=500, n_bootstrap=0):
    print(f"\n{'='*60}")
    print(f"RUNNING {n_experiments} GPIV EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Function type: {f_type}")
    print(f"Sample size per experiment: {n_samples}")
    print(f"Test points: {n_test_points}")
    print(f"Quantile threshold for accuracy: {quantile_threshold}")
    print(f"Training lengthscale_x: {train_lengthscale_x}")
    print(f"Training lengthscale_z: {train_lengthscale_z}")
    print(f"Alpha: {alpha}")
    print(f"Split ratio 1: {split_ratio_1}, Split ratio 2: {split_ratio_2}")
    print(f"X range: {x_range}")
    print(f"Bootstrap resamples: {n_bootstrap}")

    all_accurate_ratios = []
    all_removed_ratios = []
    all_mse = []
    all_coverage = []
    all_accurate_ratios_total = []
    all_removed_ratios_total = []
    all_auc = []
    all_coverage_total = []

    for exp in range(n_experiments):
        print(f"\nGPIV Experiment {exp+1}/{n_experiments}")

        seed = seed_head + exp

        mse, coverage, rejection_curve, _ = run_gpiv_experiment_with_rejection(
            f_type=f_type,
            n_samples=n_samples,
            n_test_points=n_test_points,
            seed=seed,
            quantile_threshold=quantile_threshold,
            n_iterations=n_iterations,
            lr=lr,
            train_lengthscale_x=train_lengthscale_x,
            train_lengthscale_z=train_lengthscale_z,
            train_sigma=train_sigma,
            eta=eta,
            alpha=alpha,
            split_ratio_1=split_ratio_1,
            split_ratio_2=split_ratio_2,
            x_range=x_range,
            n_bootstrap=n_bootstrap
        )

        all_accurate_ratios.append(rejection_curve['accurate_ratios'])
        all_removed_ratios.append(rejection_curve['removed_ratios'])
        all_mse.append(mse)
        all_coverage.append(coverage)
        if n_bootstrap > 0:
            all_accurate_ratios_total.append(rejection_curve['accurate_ratios_total'])
            all_removed_ratios_total.append(rejection_curve['removed_ratios_total'])
            all_auc.append(rejection_curve['auc'])
            all_coverage_total.append(rejection_curve['coverage_total'])

        print(f"  MSE: {mse:.6f}, Coverage: {coverage*100:.2f}%")
        if n_bootstrap > 0:
            print(f"  AUC: {rejection_curve['auc']:.4f}, Cov_total: {rejection_curve['coverage_total']*100:.2f}%")

    all_accurate_ratios = np.array(all_accurate_ratios)
    all_removed_ratios = np.array(all_removed_ratios)
    all_mse = np.array(all_mse)
    all_coverage = np.array(all_coverage)

    common_removed_ratios = all_removed_ratios[0]

    mean_accurate_ratios = np.mean(all_accurate_ratios, axis=0)
    std_accurate_ratios = np.std(all_accurate_ratios, axis=0)
    confidence_interval = 1.96 * std_accurate_ratios / np.sqrt(n_experiments)

    mean_mse = np.mean(all_mse)
    std_mse = np.std(all_mse)
    mean_coverage = np.mean(all_coverage)
    std_coverage = np.std(all_coverage)

    results = {
        'all_accurate_ratios': all_accurate_ratios,
        'all_removed_ratios': all_removed_ratios,
        'all_mse': all_mse,
        'all_coverage': all_coverage,
        'mean_accurate_ratios': mean_accurate_ratios,
        'std_accurate_ratios': std_accurate_ratios,
        'confidence_interval': confidence_interval,
        'common_removed_ratios': common_removed_ratios,
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'mean_coverage': mean_coverage,
        'std_coverage': std_coverage,
        'n_bootstrap': n_bootstrap
    }

    if n_bootstrap > 0:
        all_accurate_ratios_total = np.array(all_accurate_ratios_total)
        all_removed_ratios_total = np.array(all_removed_ratios_total)
        all_auc = np.array(all_auc)
        all_coverage_total = np.array(all_coverage_total)

        common_removed_total = all_removed_ratios_total[0]
        mean_accurate_total = np.mean(all_accurate_ratios_total, axis=0)
        std_accurate_total = np.std(all_accurate_ratios_total, axis=0)
        ci_total = 1.96 * std_accurate_total / np.sqrt(n_experiments)

        results.update({
            'all_accurate_ratios_total': all_accurate_ratios_total,
            'all_removed_ratios_total': all_removed_ratios_total,
            'mean_accurate_ratios_total': mean_accurate_total,
            'std_accurate_ratios_total': std_accurate_total,
            'confidence_interval_total': ci_total,
            'common_removed_ratios_total': common_removed_total,
            'all_auc': all_auc,
            'mean_auc': np.mean(all_auc),
            'std_auc': np.std(all_auc),
            'all_coverage_total': all_coverage_total,
            'mean_coverage_total': np.mean(all_coverage_total),
            'std_coverage_total': np.std(all_coverage_total)
        })

    return results


def plot_gpiv_results(results, f_type='absolute', quantile_threshold=0.75,
                      train_lengthscale_x=True, train_lengthscale_z=True):
    n_experiments = len(results['all_mse'])
    common_removed = results['common_removed_ratios']
    mean_accurate = results['mean_accurate_ratios']
    ci = results['confidence_interval']

    has_bootstrap = (results['n_bootstrap'] > 0)

    if has_bootstrap:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax1.plot(common_removed, mean_accurate, 'b-', linewidth=3, label='Mean Accuracy')
    ax1.fill_between(common_removed,
                     mean_accurate - ci,
                     mean_accurate + ci,
                     alpha=0.3, color='blue', label='95% CI')
    ax1.set_xlabel('Proportion of High-Variance Data Removed')
    ax1.set_ylabel(f'Proportion Accurate (SE ≤ Q{quantile_threshold})')
    lx_str = "Optimized" if train_lengthscale_x else "Fixed"
    lz_str = "Optimized" if train_lengthscale_z else "Fixed"
    ax1.set_title(f'GPIV ARC (Original Variance)\n{f_type}, {n_experiments} exps, lx={lx_str}, lz={lz_str}')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(quantile_threshold - 0.05, 1.0)
    ax1.legend()

    mse_vals = results['all_mse']
    ax2.hist(mse_vals, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
    ax2.axvline(results['mean_mse'], color='red', linestyle='--', linewidth=2,
                label=f'Mean: {results["mean_mse"]:.4f}')
    ax2.set_xlabel('MSE')
    ax2.set_ylabel('Frequency')
    ax2.set_title('MSE Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    cov_vals = results['all_coverage']
    ax3.hist(cov_vals, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.axvline(results['mean_coverage'], color='red', linestyle='--', linewidth=2,
                label=f'Mean: {results["mean_coverage"]*100:.2f}%')
    ax3.axvline(0.95, color='blue', linestyle='-', linewidth=2, label='Target 95%')
    ax3.set_xlabel('Coverage Rate')
    ax3.set_ylabel('Frequency')
    ax3.set_title('95% Coverage (Original Variance)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    for i in range(min(25, n_experiments)):
        ax4.plot(common_removed, results['all_accurate_ratios'][i],
                 'gray', alpha=0.2, linewidth=1)
    ax4.plot(common_removed, mean_accurate, 'b-', linewidth=3, label='Mean')
    ax4.fill_between(common_removed, mean_accurate - ci, mean_accurate + ci,
                     alpha=0.3, color='blue')
    ax4.set_xlabel('Proportion Removed')
    ax4.set_ylabel('Accuracy Proportion')
    ax4.set_title('All Individual ARC Traces')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    if has_bootstrap:
        common_removed_total = results['common_removed_ratios_total']
        mean_accurate_total = results['mean_accurate_ratios_total']
        ci_total = results['confidence_interval_total']

        ax5.plot(common_removed, mean_accurate, 'b-', linewidth=2, label='Original Variance')
        ax5.fill_between(common_removed, mean_accurate - ci, mean_accurate + ci,
                         alpha=0.2, color='blue')
        ax5.plot(common_removed_total, mean_accurate_total, 'r-', linewidth=2, label='Total Variance (GP + Bootstrap)')
        ax5.fill_between(common_removed_total, mean_accurate_total - ci_total,
                         mean_accurate_total + ci_total, alpha=0.2, color='red')
        ax5.set_xlabel('Proportion Removed')
        ax5.set_ylabel('Accuracy Proportion')
        ax5.set_title('ARC Comparison: Original vs Total Variance')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        auc_vals = results['all_auc']
        ax6.hist(auc_vals, bins=15, alpha=0.7, color='coral', edgecolor='black')
        ax6.axvline(results['mean_auc'], color='red', linestyle='--', linewidth=2,
                    label=f'Mean AUC: {results["mean_auc"]:.4f}')
        ax6.set_xlabel('AUC')
        ax6.set_ylabel('Frequency')
        ax6.set_title('AUC Distribution (Total Variance)')
        ax6.grid(True, alpha=0.3)
        ax6.legend()

    plt.suptitle(f'GPIV Results: {f_type} function, {n_experiments} experiments', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    print(f"\n{'='*60}")
    print("GPIV SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Lengthscale optimization: x={train_lengthscale_x}, z={train_lengthscale_z}")
    print(f"MSE: Mean={results['mean_mse']:.6f} ± {results['std_mse']:.6f}")
    print(f"Coverage (original): Mean={results['mean_coverage']*100:.2f}% ± {results['std_coverage']*100:.2f}%")
    if has_bootstrap:
        print(f"\nBootstrap results (n={results['n_bootstrap']}):")
        print(f"  Coverage (total variance): Mean={results['mean_coverage_total']*100:.2f}% ± {results['std_coverage_total']*100:.2f}%")
        print(f"  AUC (total variance): Mean={results['mean_auc']:.4f} ± {results['std_auc']:.4f}")

    print(f"\nARC at key removal proportions (original variance):")
    key_points = [0, 0.1, 0.2, 0.5, 0.8]
    for point in key_points:
        idx = np.argmin(np.abs(common_removed - point))
        if idx < len(mean_accurate):
            print(f"  {point*100:.0f}% removed: {mean_accurate[idx]:.4f} ± {results['std_accurate_ratios'][idx]:.4f}")


def run_single_gpiv_demo(f_type='absolute', n_samples=300, seed=91,
                         train_lengthscale_x=True, train_lengthscale_z=True,
                         train_sigma=True, eta=1.0, alpha=0.5,
                         split_ratio_1=0.5, split_ratio_2=0.5,
                         x_range=(0,1), n_bootstrap=50):
    print(f"\n{'='*60}")
    print(f"SINGLE GPIV DEMO: {f_type.upper()} FUNCTION")
    print(f"{'='*60}")
    print(f"Training lengthscale_x: {train_lengthscale_x}")
    print(f"Training lengthscale_z: {train_lengthscale_z}")
    print(f"Alpha: {alpha}")
    print(f"Split ratios: {split_ratio_1}, {split_ratio_2}")
    print(f"X range: {x_range}")
    print(f"Bootstrap resamples: {n_bootstrap}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    x, y, z, h_true = generate_data(n_samples, f_type=f_type, alpha=alpha, seed=seed)

    lengthscale_x = median_heuristic(x).item()
    lengthscale_z = median_heuristic(z).item()
    print(f"Median heuristic lengthscales: x={lengthscale_x:.3f}, z={lengthscale_z:.3f}")

    gp = GPIV(
        lengthscale_x=lengthscale_x,
        lengthscale_z=lengthscale_z,
        sigma=1.2,
        eta=eta,
        train_sigma=train_sigma,
        train_eta=False,
        lx=train_lengthscale_x,
        lz=train_lengthscale_z
    )

    gp.set_data(x.reshape(-1, 1), z.reshape(-1, 1), y,
                split_ratio_1=split_ratio_1, split_ratio_2=split_ratio_2,
                shuffle=True, random_state=seed)
    gp.compute_kernel_matrices()

    print(f"\nInitial NLL: {gp.negative_log_marginal_likelihood().item():.3f}")

    history = gp.optimize_hyperparameters(n_iterations=50, lr=0.01, verbose=True)
    print(f"\nFinal NLL: {gp.negative_log_marginal_likelihood().item():.3f}")

    gp.visualize_optimization(history)

    x_start, x_end = x_range
    x_test = torch.linspace(x_start, x_end, 200, dtype=torch.float64)

    if f_type == 'sine':
        h_true_test = 2 * torch.sin(2 * torch.pi * x_test)
    elif f_type == 'log':
        h_true_test = torch.log(torch.abs(16 * x_test - 8) + 1) * torch.sign(x_test - 0.5)
    elif f_type == 'linear':
        h_true_test = 4 * x_test - 2
    elif f_type == 'absolute':
        h_true_test = torch.abs(4 * x_test - 2) - 1

    pred_means = []
    pred_vars = []
    for xv in x_test:
        m, v = gp.predict(xv)
        pred_means.append(m.item())
        pred_vars.append(v.item())
    pred_means = torch.tensor(pred_means)
    pred_vars = torch.tensor(pred_vars)

    if n_bootstrap > 0:
        X1 = gp.X_train.squeeze().numpy()
        Z1 = gp.Z_train.squeeze().numpy()
        y1 = gp.y_train.numpy()
        X2 = gp.X_train2.squeeze().numpy()
        Z2 = gp.Z_train2.squeeze().numpy()
        y2 = gp.y_train2.numpy()

        boot_means = bootstrap_gp_predictions(
            gp, X1, Z1, y1, X2, Z2, y2, x_test,
            n_bootstrap=n_bootstrap, random_seed=100*seed
        )
        boot_var = np.var(boot_means, axis=1, ddof=1)
        total_var = pred_vars.numpy() + boot_var

        print(np.mean(boot_var))
        print(np.mean(total_var))
    else:
        boot_var = None
        total_var = pred_vars.numpy()

    plt.figure(figsize=(12, 8))
    plt.plot(x_test.numpy(), pred_means.numpy(), 'b-', linewidth=2, label='Posterior Mean')
    plt.fill_between(x_test.numpy(),
                     pred_means.numpy() - 2 * np.sqrt(pred_vars.numpy()),
                     pred_means.numpy() + 2 * np.sqrt(pred_vars.numpy()),
                     alpha=0.2, color='blue', label='GP 95% CI')
    if n_bootstrap > 0:
        plt.fill_between(x_test.numpy(),
                         pred_means.numpy() - 2 * np.sqrt(total_var),
                         pred_means.numpy() + 2 * np.sqrt(total_var),
                         alpha=0.2, color='red', label='Total 95% CI (GP+Bootstrap)')
    plt.plot(x_test.numpy(), h_true_test.numpy(), 'orange', linewidth=2, label='True h(x)')
    plt.scatter(gp.X_train2.numpy(), gp.y_train2.numpy(), color='red', s=20, alpha=0.5, label='Training Data')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'GPIV Posterior with Bootstrap (n={n_bootstrap})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(x_range)
    plt.show()

    return gp


if __name__ == "__main__":
    FUNCTION_TYPE = 'log'
    EXPERIMENT_MODE = 'single'
    ALPHA = 0.5
    SPLIT_RATIO_1 = 0.5
    SPLIT_RATIO_2 = 0.5
    X_RANGE = (-0.00, 1.00)
    N_BOOTSTRAP = 200

    if EXPERIMENT_MODE == 'single':
        gp = run_single_gpiv_demo(
            f_type=FUNCTION_TYPE,
            n_samples=1000,
            seed=22,
            train_lengthscale_x=False,
            train_lengthscale_z=False,
            train_sigma=True,
            eta=0.1,
            alpha=ALPHA,
            split_ratio_1=SPLIT_RATIO_1,
            split_ratio_2=SPLIT_RATIO_2,
            x_range=X_RANGE,
            n_bootstrap=N_BOOTSTRAP
        )
    elif EXPERIMENT_MODE == 'multiple':
        N_EXPERIMENTS = 25
        N_SAMPLES = 400
        N_TEST_POINTS = 200
        QUANTILE_THRESHOLD = 0.75

        print("Gaussian Process Instrumental Variables (GPIV) Experiment")
        print("=" * 70)

        results = run_gpiv_multiple_experiments(
            n_experiments=N_EXPERIMENTS,
            f_type=FUNCTION_TYPE,
            n_samples=N_SAMPLES,
            n_test_points=N_TEST_POINTS,
            quantile_threshold=QUANTILE_THRESHOLD,
            n_iterations=50,
            lr=0.02,
            train_lengthscale_x=False,
            train_lengthscale_z=False,
            train_sigma=True,
            eta=0.1,
            alpha=ALPHA,
            split_ratio_1=SPLIT_RATIO_1,
            split_ratio_2=SPLIT_RATIO_2,
            x_range=X_RANGE,
            seed_head=1,
            n_bootstrap=N_BOOTSTRAP
        )

        plot_gpiv_results(results, f_type=FUNCTION_TYPE,
                         quantile_threshold=QUANTILE_THRESHOLD,
                         train_lengthscale_x=True, train_lengthscale_z=True)
    else:
        print(f"Unknown experiment mode: {EXPERIMENT_MODE}")
        print("Please choose from: 'single', 'multiple'")

    print("\nExperiment completed!")