import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve
import math
import warnings

def h_function(t):
    return 2 * ((t-5)**4 / 600 + torch.exp(-4*(t-5)**2) + t/10 - 2)

def generate_demand_data(n_samples=500, rho=0.5, seed=None):
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

class GPIV:
    def __init__(self, lengthscale_p=1.0, lengthscale_c=1.0, lengthscale_t=1.0,
                 lengthscale_s=1.0, amplitude_p=1.0, amplitude_c=1.0,
                 amplitude_t=1.0, amplitude_s=1.0, sigma=0.1, eta=0.5,
                 optimize_lengthscale_p=True, optimize_lengthscale_c=True,
                 optimize_lengthscale_t=True, optimize_lengthscale_s=True,
                 optimize_amplitude_p=False, optimize_amplitude_c=False,
                 optimize_amplitude_t=False, optimize_amplitude_s=False,
                 optimize_sigma=True, optimize_eta=False):
        self.lengthscale_p = torch.tensor(lengthscale_p, dtype=torch.float64, requires_grad=optimize_lengthscale_p)
        self.lengthscale_c = torch.tensor(lengthscale_c, dtype=torch.float64, requires_grad=optimize_lengthscale_c)
        self.lengthscale_t = torch.tensor(lengthscale_t, dtype=torch.float64, requires_grad=optimize_lengthscale_t)
        self.lengthscale_s = torch.tensor(lengthscale_s, dtype=torch.float64, requires_grad=optimize_lengthscale_s)

        self.amplitude_p = torch.tensor(amplitude_p, dtype=torch.float64, requires_grad=optimize_amplitude_p)
        self.amplitude_c = torch.tensor(amplitude_c, dtype=torch.float64, requires_grad=optimize_amplitude_c)
        self.amplitude_t = torch.tensor(amplitude_t, dtype=torch.float64, requires_grad=optimize_amplitude_t)
        self.amplitude_s = torch.tensor(amplitude_s, dtype=torch.float64, requires_grad=optimize_amplitude_s)

        self.sigma = torch.tensor(sigma, dtype=torch.float64, requires_grad=optimize_sigma)
        self.eta = torch.tensor(eta, dtype=torch.float64, requires_grad=optimize_eta)

        self.scalers = None

    def set_data(self, P, C, T, S, y, scalers=None):
        self.P_train = P
        self.C_train = C
        self.T_train = T
        self.S_train = S
        self.y_train = y

        if scalers is not None:
            self.scalers = scalers

    def compute_kernel_matrices(self):
        self.Kxx = (rbf_kernel(self.P_train, self.P_train, self.lengthscale_p, self.amplitude_p) *
                   rbf_kernel(self.T_train, self.T_train, self.lengthscale_t, self.amplitude_t) *
                   rbf_kernel(self.S_train, self.S_train, self.lengthscale_s, self.amplitude_s))

        self.Lzz = (rbf_kernel(self.C_train, self.C_train, self.lengthscale_c, self.amplitude_c) *
                   rbf_kernel(self.T_train, self.T_train, self.lengthscale_t, self.amplitude_t) *
                   rbf_kernel(self.S_train, self.S_train, self.lengthscale_s, self.amplitude_s))

        Lzz_noise = self.Lzz + self.eta * torch.eye(len(self.C_train), dtype=torch.float64)

        try:
            self.invLzz = torch.inverse(Lzz_noise)
        except:
            self.invLzz = torch.pinverse(Lzz_noise)

        self.Lxz = torch.inverse(
            self.Lzz @ self.invLzz @ self.Kxx @ self.invLzz @ self.Lzz +
            (self.sigma**2) * torch.eye(len(self.C_train), dtype=torch.float64)
        )

    def posterior_mean(self, p_new, t_new, s_new):
        Kp = rbf_kernel(p_new.reshape(1, -1), self.P_train, self.lengthscale_p, self.amplitude_p)
        Kt = rbf_kernel(t_new.reshape(1, -1), self.T_train, self.lengthscale_t, self.amplitude_t)
        Ks = rbf_kernel(s_new.reshape(1, -1), self.S_train, self.lengthscale_s, self.amplitude_s)
        Kx = Kp * Kt * Ks

        mean = Kx @ self.invLzz @ self.Lzz @ self.Lxz @ self.y_train.reshape(-1, 1)
        return mean.squeeze()

    def posterior_covariance(self, p_new, t_new, s_new, p_prime, t_prime, s_prime):
        Kp = rbf_kernel(p_new.reshape(1, -1), self.P_train, self.lengthscale_p, self.amplitude_p)
        Kt = rbf_kernel(t_new.reshape(1, -1), self.T_train, self.lengthscale_t, self.amplitude_t)
        Ks = rbf_kernel(s_new.reshape(1, -1), self.S_train, self.lengthscale_s, self.amplitude_s)
        Kx = Kp * Kt * Ks

        Kp_p = rbf_kernel(p_prime.reshape(1, -1), self.P_train, self.lengthscale_p, self.amplitude_p)
        Kt_p = rbf_kernel(t_prime.reshape(1, -1), self.T_train, self.lengthscale_t, self.amplitude_t)
        Ks_p = rbf_kernel(s_prime.reshape(1, -1), self.S_train, self.lengthscale_s, self.amplitude_s)
        Kx_prime = Kp_p * Kt_p * Ks_p

        kpp_p = rbf_kernel(p_prime.reshape(1, -1), p_new.reshape(1, -1), self.lengthscale_p, self.amplitude_p)
        ktt_p = rbf_kernel(t_prime.reshape(1, -1), t_new.reshape(1, -1), self.lengthscale_t, self.amplitude_t)
        kss_p = rbf_kernel(s_prime.reshape(1, -1), s_new.reshape(1, -1), self.lengthscale_s, self.amplitude_s)
        kxx_prime = kpp_p * ktt_p * kss_p

        covariance = kxx_prime - Kx @ self.invLzz @ self.Lzz @ self.Lxz @ self.Lzz @ self.invLzz @ Kx_prime.T
        return covariance.squeeze()

    def posterior_variance(self, p_new, t_new, s_new):
        return self.posterior_covariance(p_new, t_new, s_new, p_new, t_new, s_new)

    def predict(self, p_new, t_new, s_new, standardize_input=True):
        if standardize_input and self.scalers is not None:
            p_scaled = (p_new - self.scalers['P']['mean']) / self.scalers['P']['std']
            t_scaled = (t_new - self.scalers['T']['mean']) / self.scalers['T']['std']
            s_scaled = (s_new - self.scalers['S']['mean']) / self.scalers['S']['std']
        else:
            p_scaled, t_scaled, s_scaled = p_new, t_new, s_new

        mean = self.posterior_mean(p_scaled, t_scaled, s_scaled)
        variance = self.posterior_variance(p_scaled, t_scaled, s_scaled)

        return mean, variance

    def negative_log_marginal_likelihood(self):
        n = len(self.y_train)

        K_full = (self.Lzz @ self.invLzz @ self.Kxx @ self.invLzz @ self.Lzz +
                 (self.sigma**2) * torch.eye(n, dtype=torch.float64))

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

        nll = data_fit + log_det
        return nll

    def optimize_hyperparameters(self, n_iterations=200, lr=0.01, verbose=True):
        params = []
        param_names = []

        if self.lengthscale_p.requires_grad:
            params.append(self.lengthscale_p)
            param_names.append('lengthscale_p')
        if self.lengthscale_c.requires_grad:
            params.append(self.lengthscale_c)
            param_names.append('lengthscale_c')
        if self.lengthscale_t.requires_grad:
            params.append(self.lengthscale_t)
            param_names.append('lengthscale_t')
        if self.lengthscale_s.requires_grad:
            params.append(self.lengthscale_s)
            param_names.append('lengthscale_s')
        if self.amplitude_p.requires_grad:
            params.append(self.amplitude_p)
            param_names.append('amplitude_p')
        if self.amplitude_c.requires_grad:
            params.append(self.amplitude_c)
            param_names.append('amplitude_c')
        if self.amplitude_t.requires_grad:
            params.append(self.amplitude_t)
            param_names.append('amplitude_t')
        if self.amplitude_s.requires_grad:
            params.append(self.amplitude_s)
            param_names.append('amplitude_s')
        if self.sigma.requires_grad:
            params.append(self.sigma)
            param_names.append('sigma')
        if self.eta.requires_grad:
            params.append(self.eta)
            param_names.append('eta')

        if len(params) == 0:
            if verbose:
                print("No hyperparameters to optimize (all requires_grad=False)")
            return {'losses': [], 'parameters': {}}

        optimizer = torch.optim.Adam(params, lr=lr)

        losses = []
        history = {
            'losses': [],
            'parameters': {name: [] for name in param_names}
        }

        if verbose:
            print("Starting hyperparameter optimization...")
            print(f"Parameters to optimize: {', '.join(param_names)}")
            print(f"Initial values:")
            for name in param_names:
                value = getattr(self, name).item()
                print(f"  {name}: {value:.3f}")

        for i in range(n_iterations):
            optimizer.zero_grad()

            self.compute_kernel_matrices()

            loss = self.negative_log_marginal_likelihood()

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                if hasattr(self, 'lengthscale_p'):
                    self.lengthscale_p.data = torch.clamp(self.lengthscale_p, min=1e-3)
                if hasattr(self, 'lengthscale_c'):
                    self.lengthscale_c.data = torch.clamp(self.lengthscale_c, min=1e-5)
                if hasattr(self, 'lengthscale_t'):
                    self.lengthscale_t.data = torch.clamp(self.lengthscale_t, min=1e-3)
                if hasattr(self, 'lengthscale_s'):
                    self.lengthscale_s.data = torch.clamp(self.lengthscale_s, min=1e-5)
                if hasattr(self, 'sigma'):
                    self.sigma.data = torch.clamp(self.sigma, min=1e-6)

                if hasattr(self, 'amplitude_p'):
                    self.amplitude_p.data = torch.clamp(self.amplitude_p, min=1e-3)
                if hasattr(self, 'amplitude_c'):
                    self.amplitude_c.data = torch.clamp(self.amplitude_c, min=1e-3)
                if hasattr(self, 'amplitude_t'):
                    self.amplitude_t.data = torch.clamp(self.amplitude_t, min=1e-3)
                if hasattr(self, 'amplitude_s'):
                    self.amplitude_s.data = torch.clamp(self.amplitude_s, min=1e-3)

            losses.append(loss.item())
            for name in param_names:
                history['parameters'][name].append(getattr(self, name).item())

            if verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.3f}")
                for name in param_names:
                    value = getattr(self, name).item()
                    print(f"  {name}: {value:.3f}", end=' ')
                print()

        history['losses'] = losses

        if verbose:
            print(f"Optimization completed. Final values:")
            for name in param_names:
                value = getattr(self, name).item()
                print(f"  {name}: {value:.3f}")

        return history

    def visualize_optimization(self, history):
        if len(history['losses']) == 0:
            print("No optimization history to visualize")
            return

        num_params = len(history['parameters'])
        if num_params == 0:
            plt.figure(figsize=(8, 5))
            plt.plot(history['losses'])
            plt.xlabel('Iterations')
            plt.ylabel('Negative Log Marginal Likelihood')
            plt.title('Loss Curve')
            plt.grid(True, alpha=0.3)
            plt.show()
            return

        rows = 2
        cols = max(2, (num_params + 1) // 2 + 1)

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = axes.flatten()

        axes[0].plot(history['losses'])
        axes[0].set_xlabel('Iterations')
        axes[0].set_ylabel('NLL')
        axes[0].set_title('Loss Curve')
        axes[0].grid(True, alpha=0.3)

        for idx, (param_name, param_values) in enumerate(history['parameters'].items()):
            ax_idx = idx + 1
            if ax_idx < len(axes):
                axes[ax_idx].plot(param_values)
                axes[ax_idx].set_xlabel('Iterations')
                axes[ax_idx].set_ylabel(param_name)
                axes[ax_idx].set_title(f'{param_name} Optimization')
                axes[ax_idx].grid(True, alpha=0.3)

        for i in range(num_params + 1, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

    def compute_test_predictions(self, standardize_input=True):
        p_test = torch.linspace(2.5, 27.5, 30, dtype=torch.float64)
        t_test = torch.linspace(0, 10, 20, dtype=torch.float64)
        s_test = torch.arange(1, 8, dtype=torch.float64)

        predictions = []
        variances = []
        true_values = []

        for p in p_test:
            for t in t_test:
                for s in s_test:
                    h_t = h_function(t)
                    true_f = (100 + (10 + p) * s * h_t - 2 * p).item()

                    if self.scalers is not None and 'Y' in self.scalers:
                        true_f_scaled = (true_f - self.scalers['Y']['mean'].item()) / self.scalers['Y']['std'].item()
                    else:
                        true_f_scaled = true_f

                    pred_f, var_f = self.predict(p, t, s, standardize_input=standardize_input)

                    predictions.append(pred_f.item())
                    variances.append(var_f.item())
                    true_values.append(true_f_scaled)

        predictions = np.array(predictions)
        variances = np.array(variances)
        true_values = np.array(true_values)

        return predictions, variances, true_values

    def compute_mse(self, standardize_input=True):
        predictions, _, true_values = self.compute_test_predictions(standardize_input)
        mse = np.mean((predictions - true_values)**2)
        return mse

    def compute_accuracy_rejection_curve(self, delta=None, quantile=0.75, standardize_input=True):
        predictions, variances, true_values = self.compute_test_predictions(standardize_input)

        errors = np.abs(predictions - true_values)

        sorted_indices = np.argsort(variances)[::-1]
        sorted_errors = errors[sorted_indices]

        if delta is None:
            delta = np.quantile(errors, quantile)

        n_points = len(predictions)
        rejection_rates = np.linspace(0, 0.99, 100)
        accuracies = []

        for rejection_rate in rejection_rates:
            n_remove = int(n_points * rejection_rate)

            if n_remove < n_points:
                kept_errors = sorted_errors[n_remove:]
                accuracy = np.mean(kept_errors <= delta)
            else:
                accuracy = 0.0

            accuracies.append(accuracy)

        return rejection_rates, np.array(accuracies), delta

    def compute_coverage_rate(self, confidence_level=0.95, standardize_input=True):
        predictions, variances, true_values = self.compute_test_predictions(standardize_input)

        z_score = norm.ppf((1 + confidence_level) / 2)

        stds = np.sqrt(variances)
        lower_bound = predictions - z_score * stds
        upper_bound = predictions + z_score * stds
        in_ci = (true_values >= lower_bound) & (true_values <= upper_bound)
        coverage = np.mean(in_ci)

        return coverage

    def visualize_demand_curve_p(self, t_fixed=5.0, s_fixed=4, standardize_input=True):
        p_range = torch.linspace(2.5, 27.5, 100, dtype=torch.float64)

        true_demand = []
        pred_demand = []
        var_demand = []

        for p in p_range:
            h_t = h_function(torch.tensor(t_fixed))
            true_f = 100 + (10 + p) * s_fixed * h_t - 2 * p

            if self.scalers is not None and 'Y' in self.scalers:
                true_f_scaled = (true_f - self.scalers['Y']['mean']) / self.scalers['Y']['std']
            else:
                true_f_scaled = true_f

            true_demand.append(true_f_scaled.item())

            pred_f, var_f = self.predict(p, torch.tensor(t_fixed, dtype=torch.float64),
                                         torch.tensor(s_fixed, dtype=torch.float64),
                                         standardize_input=standardize_input)
            pred_demand.append(pred_f.item())
            var_demand.append(var_f.item())

        var_tensor = torch.tensor(var_demand)
        stds = torch.sqrt(var_tensor)

        plt.figure(figsize=(10, 6))
        plt.plot(p_range.numpy(), true_demand, 'b-', linewidth=2, label='True (Standardized)')
        plt.plot(p_range.numpy(), pred_demand, 'r--', linewidth=2, label='Predicted (Standardized)')
        plt.fill_between(p_range.numpy(),
                        pred_demand - 1.96 * stds.numpy(),
                        pred_demand + 1.96 * stds.numpy(),
                        alpha=0.2, color='blue', label='95% CI')
        plt.xlabel('Price (P) - Original Scale')
        plt.ylabel('Demand (Y) - Standardized Scale')
        plt.title(f'Demand Curve (T={t_fixed}, S={s_fixed}) - Standardized')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return p_range, true_demand, pred_demand

    def visualize_demand_curve_t(self, p_fixed=25.0, s_fixed=4, standardize_input=True):
        t_range = torch.linspace(0, 10, 100, dtype=torch.float64)

        true_demand = []
        pred_demand = []
        var_demand = []

        for t in t_range:
            h_t = h_function(torch.tensor(t))
            true_f = 100 + (10 + p_fixed) * s_fixed * h_t - 2 * p_fixed

            if self.scalers is not None and 'Y' in self.scalers:
                true_f_scaled = (true_f - self.scalers['Y']['mean']) / self.scalers['Y']['std']
            else:
                true_f_scaled = true_f

            true_demand.append(true_f_scaled.item())

            pred_f, var_f = self.predict(torch.tensor(p_fixed, dtype=torch.float64),
                                        t, torch.tensor(s_fixed, dtype=torch.float64),
                                        standardize_input=standardize_input)
            pred_demand.append(pred_f.item())
            var_demand.append(var_f.item())

        var_tensor = torch.tensor(var_demand)
        stds = torch.sqrt(var_tensor)

        plt.figure(figsize=(10, 6))
        plt.plot(t_range.numpy(), true_demand, 'b-', linewidth=2, label='True (Standardized)')
        plt.plot(t_range.numpy(), pred_demand, 'r--', linewidth=2, label='Predicted (Standardized)')
        plt.fill_between(t_range.numpy(),
                        pred_demand - 1.96 * stds.numpy(),
                        pred_demand + 1.96 * stds.numpy(),
                        alpha=0.2, color='blue', label='95% CI')
        plt.xlabel('Temperature (T) - Original Scale')
        plt.ylabel('Demand (Y) - Standardized Scale')
        plt.title(f'Demand Curve (P={p_fixed}, S={s_fixed}) - Standardized')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return t_range, true_demand, pred_demand

class BootstrapGPIV:
    def __init__(self, lengthscale_p=1.0, lengthscale_c=1.0, lengthscale_t=1.0,
                 lengthscale_s=1.0, amplitude_p=1.0, amplitude_c=1.0,
                 amplitude_t=1.0, amplitude_s=1.0, sigma=0.1, eta=0.5,
                 optimize_lengthscale_p=True, optimize_lengthscale_c=True,
                 optimize_lengthscale_t=True, optimize_lengthscale_s=True,
                 optimize_amplitude_p=False, optimize_amplitude_c=False,
                 optimize_amplitude_t=False, optimize_amplitude_s=False,
                 optimize_sigma=True, optimize_eta=False,
                 n_bootstrap=200):
        self.lengthscale_p = torch.tensor(lengthscale_p, dtype=torch.float64, requires_grad=optimize_lengthscale_p)
        self.lengthscale_c = torch.tensor(lengthscale_c, dtype=torch.float64, requires_grad=optimize_lengthscale_c)
        self.lengthscale_t = torch.tensor(lengthscale_t, dtype=torch.float64, requires_grad=optimize_lengthscale_t)
        self.lengthscale_s = torch.tensor(lengthscale_s, dtype=torch.float64, requires_grad=optimize_lengthscale_s)

        self.amplitude_p = torch.tensor(amplitude_p, dtype=torch.float64, requires_grad=optimize_amplitude_p)
        self.amplitude_c = torch.tensor(amplitude_c, dtype=torch.float64, requires_grad=optimize_amplitude_c)
        self.amplitude_t = torch.tensor(amplitude_t, dtype=torch.float64, requires_grad=optimize_amplitude_t)
        self.amplitude_s = torch.tensor(amplitude_s, dtype=torch.float64, requires_grad=optimize_amplitude_s)

        self.sigma = torch.tensor(sigma, dtype=torch.float64, requires_grad=optimize_sigma)
        self.eta = torch.tensor(eta, dtype=torch.float64, requires_grad=optimize_eta)

        self.n_bootstrap = n_bootstrap
        self.bootstrap_models = None
        self.bootstrap_predictions = None

        self.scalers = None
        self.original_data = None

    def set_data(self, P, C, T, S, y, scalers=None, original_data=None):
        self.P_train = P
        self.C_train = C
        self.T_train = T
        self.S_train = S
        self.y_train = y

        if scalers is not None:
            self.scalers = scalers

        if original_data is not None:
            self.original_data = original_data
        else:
            self.original_data = {
                'P': P.clone(),
                'C': C.clone(),
                'T': T.clone(),
                'S': S.clone(),
                'y': y.clone()
            }

    def compute_kernel_matrices(self):
        self.Kxx = (rbf_kernel(self.P_train, self.P_train, self.lengthscale_p, self.amplitude_p) *
                   rbf_kernel(self.T_train, self.T_train, self.lengthscale_t, self.amplitude_t) *
                   rbf_kernel(self.S_train, self.S_train, self.lengthscale_s, self.amplitude_s))

        self.Lzz = (rbf_kernel(self.C_train, self.C_train, self.lengthscale_c, self.amplitude_c) *
                   rbf_kernel(self.T_train, self.T_train, self.lengthscale_t, self.amplitude_t) *
                   rbf_kernel(self.S_train, self.S_train, self.lengthscale_s, self.amplitude_s))

        Lzz_noise = self.Lzz + self.eta * torch.eye(len(self.C_train), dtype=torch.float64)

        try:
            self.invLzz = torch.inverse(Lzz_noise)
        except:
            self.invLzz = torch.pinverse(Lzz_noise)

        self.Lxz = torch.inverse(
            self.Lzz @ self.invLzz @ self.Kxx @ self.invLzz @ self.Lzz +
            (self.sigma**2) * torch.eye(len(self.C_train), dtype=torch.float64)
        )

    def posterior_mean(self, p_new, t_new, s_new):
        Kp = rbf_kernel(p_new.reshape(1, -1), self.P_train, self.lengthscale_p, self.amplitude_p)
        Kt = rbf_kernel(t_new.reshape(1, -1), self.T_train, self.lengthscale_t, self.amplitude_t)
        Ks = rbf_kernel(s_new.reshape(1, -1), self.S_train, self.lengthscale_s, self.amplitude_s)
        Kx = Kp * Kt * Ks

        mean = Kx @ self.invLzz @ self.Lzz @ self.Lxz @ self.y_train.reshape(-1, 1)
        return mean.squeeze()

    def bootstrap_resample(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        n = len(self.original_data['y'])

        indices = torch.randint(0, n, (n,))

        bootstrap_data = {
            'P': self.original_data['P'][indices],
            'C': self.original_data['C'][indices],
            'T': self.original_data['T'][indices],
            'S': self.original_data['S'][indices],
            'y': self.original_data['y'][indices]
        }

        return bootstrap_data

    def train_bootstrap_model(self, bootstrap_data, n_iterations=100, lr=0.1, verbose=False):
        model = BootstrapGPIV(
            lengthscale_p=self.lengthscale_p.item(),
            lengthscale_c=self.lengthscale_c.item(),
            lengthscale_t=self.lengthscale_t.item(),
            lengthscale_s=self.lengthscale_s.item(),
            amplitude_p=self.amplitude_p.item(),
            amplitude_c=self.amplitude_c.item(),
            amplitude_t=self.amplitude_t.item(),
            amplitude_s=self.amplitude_s.item(),
            sigma=self.sigma.item(),
            eta=self.eta.item(),
            optimize_lengthscale_p=self.lengthscale_p.requires_grad,
            optimize_lengthscale_c=self.lengthscale_c.requires_grad,
            optimize_lengthscale_t=self.lengthscale_t.requires_grad,
            optimize_lengthscale_s=self.lengthscale_s.requires_grad,
            optimize_amplitude_p=self.amplitude_p.requires_grad,
            optimize_amplitude_c=self.amplitude_c.requires_grad,
            optimize_amplitude_t=self.amplitude_t.requires_grad,
            optimize_amplitude_s=self.amplitude_s.requires_grad,
            optimize_sigma=self.sigma.requires_grad,
            optimize_eta=self.eta.requires_grad,
            n_bootstrap=self.n_bootstrap
        )

        model.set_data(
            bootstrap_data['P'], bootstrap_data['C'], bootstrap_data['T'],
            bootstrap_data['S'], bootstrap_data['y'],
            scalers=self.scalers, original_data=bootstrap_data
        )

        model.optimize_hyperparameters(n_iterations=n_iterations, lr=lr, verbose=verbose)

        return model

    def run_bootstrap(self, n_iterations=None, lr=0.1, verbose=True):
        if n_iterations is None:
            n_iterations = 50

        if verbose:
            print(f"Running bootstrap with {self.n_bootstrap} iterations...")

        self.bootstrap_models = []

        for i in range(self.n_bootstrap):
            if verbose and (i + 1) % 20 == 0:
                print(f"  Bootstrap iteration {i + 1}/{self.n_bootstrap}")

            bootstrap_data = self.bootstrap_resample(seed=i)

            bootstrap_model = self.train_bootstrap_model(
                bootstrap_data, n_iterations=n_iterations, lr=lr, verbose=False
            )

            self.bootstrap_models.append(bootstrap_model)

        if verbose:
            print("Bootstrap completed.")

    def predict_with_bootstrap_variance(self, p_new, t_new, s_new, standardize_input=True):
        if self.bootstrap_models is None:
            raise ValueError("Bootstrap models not trained. Call run_bootstrap() first.")

        if standardize_input and self.scalers is not None:
            p_scaled = (p_new - self.scalers['P']['mean']) / self.scalers['P']['std']
            t_scaled = (t_new - self.scalers['T']['mean']) / self.scalers['T']['std']
            s_scaled = (s_new - self.scalers['S']['mean']) / self.scalers['S']['std']
        else:
            p_scaled, t_scaled, s_scaled = p_new, t_new, s_new

        mean = self.posterior_mean(p_scaled, t_scaled, s_scaled).item()

        bootstrap_predictions = []
        for model in self.bootstrap_models:
            pred = model.posterior_mean(p_scaled, t_scaled, s_scaled).item()
            bootstrap_predictions.append(pred)

        variance = np.var(bootstrap_predictions)

        return mean, variance

    def compute_test_predictions_with_bootstrap(self, standardize_input=True):
        p_test = torch.linspace(0, 30, 30, dtype=torch.float64)
        t_test = torch.linspace(0, 10, 20, dtype=torch.float64)
        s_test = torch.arange(1, 8, dtype=torch.float64)

        predictions = []
        variances = []
        true_values = []

        for p in p_test:
            for t in t_test:
                for s in s_test:
                    h_t = h_function(t)
                    true_f = (100 + (10 + p) * s * h_t - 2 * p).item()

                    if self.scalers is not None and 'Y' in self.scalers:
                        true_f_scaled = (true_f - self.scalers['Y']['mean'].item()) / self.scalers['Y']['std'].item()
                    else:
                        true_f_scaled = true_f

                    pred_f, var_f = self.predict_with_bootstrap_variance(p, t, s, standardize_input=standardize_input)

                    predictions.append(pred_f)
                    variances.append(var_f)
                    true_values.append(true_f_scaled)

        predictions = np.array(predictions)
        variances = np.array(variances)
        true_values = np.array(true_values)

        return predictions, variances, true_values

    def compute_mse_with_bootstrap(self, standardize_input=True):
        predictions, _, true_values = self.compute_test_predictions_with_bootstrap(standardize_input)
        mse = np.mean((predictions - true_values)**2)
        return mse

    def compute_accuracy_rejection_curve_with_bootstrap(self, delta=None, quantile=0.75, standardize_input=True):
        predictions, variances, true_values = self.compute_test_predictions_with_bootstrap(standardize_input)

        errors = np.abs(predictions - true_values)

        sorted_indices = np.argsort(variances)[::-1]
        sorted_errors = errors[sorted_indices]

        if delta is None:
            delta = np.quantile(errors, quantile)

        n_points = len(predictions)
        rejection_rates = np.linspace(0, 0.99, 100)
        accuracies = []

        for rejection_rate in rejection_rates:
            n_remove = int(n_points * rejection_rate)

            if n_remove < n_points:
                kept_errors = sorted_errors[n_remove:]
                accuracy = np.mean(kept_errors <= delta)
            else:
                accuracy = 0.0

            accuracies.append(accuracy)

        return rejection_rates, np.array(accuracies), delta

    def compute_coverage_rate_with_bootstrap(self, confidence_level=0.95, standardize_input=True):
        predictions, variances, true_values = self.compute_test_predictions_with_bootstrap(standardize_input)

        z_score = norm.ppf((1 + confidence_level) / 2)

        stds = np.sqrt(variances)
        lower_bound = predictions - z_score * stds
        upper_bound = predictions + z_score * stds
        in_ci = (true_values >= lower_bound) & (true_values <= upper_bound)
        coverage = np.mean(in_ci)

        return coverage

    def negative_log_marginal_likelihood(self):
        n = len(self.y_train)

        K_full = (self.Lzz @ self.invLzz @ self.Kxx @ self.invLzz @ self.Lzz +
                 (self.sigma**2) * torch.eye(n, dtype=torch.float64))

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

        nll = data_fit + log_det
        return nll

    def optimize_hyperparameters(self, n_iterations=200, lr=0.01, verbose=True):
        params = []
        param_names = []

        if self.lengthscale_p.requires_grad:
            params.append(self.lengthscale_p)
            param_names.append('lengthscale_p')
        if self.lengthscale_c.requires_grad:
            params.append(self.lengthscale_c)
            param_names.append('lengthscale_c')
        if self.lengthscale_t.requires_grad:
            params.append(self.lengthscale_t)
            param_names.append('lengthscale_t')
        if self.lengthscale_s.requires_grad:
            params.append(self.lengthscale_s)
            param_names.append('lengthscale_s')
        if self.amplitude_p.requires_grad:
            params.append(self.amplitude_p)
            param_names.append('amplitude_p')
        if self.amplitude_c.requires_grad:
            params.append(self.amplitude_c)
            param_names.append('amplitude_c')
        if self.amplitude_t.requires_grad:
            params.append(self.amplitude_t)
            param_names.append('amplitude_t')
        if self.amplitude_s.requires_grad:
            params.append(self.amplitude_s)
            param_names.append('amplitude_s')
        if self.sigma.requires_grad:
            params.append(self.sigma)
            param_names.append('sigma')
        if self.eta.requires_grad:
            params.append(self.eta)
            param_names.append('eta')

        if len(params) == 0:
            if verbose:
                print("No hyperparameters to optimize (all requires_grad=False)")
            return {'losses': [], 'parameters': {}}

        optimizer = torch.optim.Adam(params, lr=lr)

        losses = []
        history = {
            'losses': [],
            'parameters': {name: [] for name in param_names}
        }

        if verbose:
            print("Starting hyperparameter optimization...")
            print(f"Parameters to optimize: {', '.join(param_names)}")
            print(f"Initial values:")
            for name in param_names:
                value = getattr(self, name).item()
                print(f"  {name}: {value:.3f}")

        for i in range(n_iterations):
            optimizer.zero_grad()

            self.compute_kernel_matrices()

            loss = self.negative_log_marginal_likelihood()

            loss.backward()

            optimizer.step()

            with torch.no_grad():
                if hasattr(self, 'lengthscale_p'):
                    self.lengthscale_p.data = torch.clamp(self.lengthscale_p, min=1e-3)
                if hasattr(self, 'lengthscale_c'):
                    self.lengthscale_c.data = torch.clamp(self.lengthscale_c, min=1e-5)
                if hasattr(self, 'lengthscale_t'):
                    self.lengthscale_t.data = torch.clamp(self.lengthscale_t, min=1e-3)
                if hasattr(self, 'lengthscale_s'):
                    self.lengthscale_s.data = torch.clamp(self.lengthscale_s, min=1e-5)
                if hasattr(self, 'sigma'):
                    self.sigma.data = torch.clamp(self.sigma, min=1e-6)

                if hasattr(self, 'amplitude_p'):
                    self.amplitude_p.data = torch.clamp(self.amplitude_p, min=1e-3)
                if hasattr(self, 'amplitude_c'):
                    self.amplitude_c.data = torch.clamp(self.amplitude_c, min=1e-3)
                if hasattr(self, 'amplitude_t'):
                    self.amplitude_t.data = torch.clamp(self.amplitude_t, min=1e-3)
                if hasattr(self, 'amplitude_s'):
                    self.amplitude_s.data = torch.clamp(self.amplitude_s, min=1e-3)

            losses.append(loss.item())
            for name in param_names:
                history['parameters'][name].append(getattr(self, name).item())

            if verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.3f}")
                for name in param_names:
                    value = getattr(self, name).item()
                    print(f"  {name}: {value:.3f}", end=' ')
                print()

        history['losses'] = losses

        if verbose:
            print(f"Optimization completed. Final values:")
            for name in param_names:
                value = getattr(self, name).item()
                print(f"  {name}: {value:.3f}")

        return history

    def visualize_optimization(self, history):
        if len(history['losses']) == 0:
            print("No optimization history to visualize")
            return

        num_params = len(history['parameters'])
        if num_params == 0:
            plt.figure(figsize=(8, 5))
            plt.plot(history['losses'])
            plt.xlabel('Iterations')
            plt.ylabel('Negative Log Marginal Likelihood')
            plt.title('Loss Curve')
            plt.grid(True, alpha=0.3)
            plt.show()
            return

        rows = 2
        cols = max(2, (num_params + 1) // 2 + 1)

        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        axes = axes.flatten()

        axes[0].plot(history['losses'])
        axes[0].set_xlabel('Iterations')
        axes[0].set_ylabel('NLL')
        axes[0].set_title('Loss Curve')
        axes[0].grid(True, alpha=0.3)

        for idx, (param_name, param_values) in enumerate(history['parameters'].items()):
            ax_idx = idx + 1
            if ax_idx < len(axes):
                axes[ax_idx].plot(param_values)
                axes[ax_idx].set_xlabel('Iterations')
                axes[ax_idx].set_ylabel(param_name)
                axes[ax_idx].set_title(f'{param_name} Optimization')
                axes[ax_idx].grid(True, alpha=0.3)

        for i in range(num_params + 1, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.show()

def set_seed(seed=442):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_single_experiment(seed=59, n_train=500, quantile=0.75,
                         optimize_lengthscale_p=True, optimize_lengthscale_c=True,
                         optimize_lengthscale_t=True, optimize_lengthscale_s=True,
                         optimize_amplitude_p=False, optimize_amplitude_c=False,
                         optimize_amplitude_t=False, optimize_amplitude_s=False,
                         optimize_sigma=True, optimize_eta=False,
                         standardize_input=True, verbose=True):
    if verbose:
        print(f"Running single experiment with seed {seed}...")

    Y_train, P_train, T_train, S_train, C_train = generate_demand_data(
        n_samples=n_train, rho=0.5, seed=seed
    )

    data_dict = {'Y': Y_train, 'P': P_train, 'T': T_train, 'S': S_train, 'C': C_train}
    scaled_data, scalers = standardize_data(data_dict)

    Y_scaled = scaled_data['Y']
    P_scaled = scaled_data['P']
    T_scaled = scaled_data['T']
    S_scaled = scaled_data['S']
    C_scaled = scaled_data['C']

    lengthscale_p = median_heuristic(P_scaled).item()
    lengthscale_c = median_heuristic(C_scaled).item()
    lengthscale_t = median_heuristic(T_scaled).item()
    lengthscale_s = median_heuristic(S_scaled).item()

    gp = GPIV(
        lengthscale_p=lengthscale_p,
        lengthscale_c=lengthscale_c,
        lengthscale_t=lengthscale_t,
        lengthscale_s=lengthscale_s,
        amplitude_p=1.0, amplitude_c=1.0, amplitude_t=1.0, amplitude_s=1.0,
        sigma=0.5, eta=0.5,
        optimize_lengthscale_p=optimize_lengthscale_p,
        optimize_lengthscale_c=optimize_lengthscale_c,
        optimize_lengthscale_t=optimize_lengthscale_t,
        optimize_lengthscale_s=optimize_lengthscale_s,
        optimize_amplitude_p=optimize_amplitude_p,
        optimize_amplitude_c=optimize_amplitude_c,
        optimize_amplitude_t=optimize_amplitude_t,
        optimize_amplitude_s=optimize_amplitude_s,
        optimize_sigma=optimize_sigma,
        optimize_eta=optimize_eta
    )

    gp.set_data(P_scaled, C_scaled, T_scaled, S_scaled, Y_scaled, scalers)

    gp.compute_kernel_matrices()

    any_optimize = (optimize_lengthscale_p or optimize_lengthscale_c or
                   optimize_lengthscale_t or optimize_lengthscale_s or
                   optimize_amplitude_p or optimize_amplitude_c or
                   optimize_amplitude_t or optimize_amplitude_s or
                   optimize_sigma or optimize_eta)

    if any_optimize:
        if verbose:
            print("Optimizing hyperparameters...")
        history = gp.optimize_hyperparameters(n_iterations=150, lr=0.1, verbose=verbose)
    else:
        history = {'losses': [], 'parameters': {}}

    test_mse = gp.compute_mse(standardize_input=standardize_input)

    if verbose:
        print("Computing ARC and coverage...")
    rejection_rates, accuracies, delta = gp.compute_accuracy_rejection_curve(
        delta=None, quantile=quantile, standardize_input=standardize_input
    )

    arc_results = {
        'rejection_rates': rejection_rates,
        'accuracies': accuracies,
        'delta': delta
    }

    initial_coverage = gp.compute_coverage_rate(
        confidence_level=0.95, standardize_input=standardize_input
    )

    return gp, test_mse, arc_results, initial_coverage, history

def run_multiple_experiments(n_repeats=25, n_train=500, quantile=0.75, base_seed=42,
                            optimize_lengthscale_p=True, optimize_lengthscale_c=True,
                            optimize_lengthscale_t=True, optimize_lengthscale_s=True,
                            optimize_amplitude_p=False, optimize_amplitude_c=False,
                            optimize_amplitude_t=False, optimize_amplitude_s=False,
                            optimize_sigma=True, optimize_eta=False,
                            standardize_input=True):
    print(f"Running {n_repeats} experiments...")

    all_mse = []
    all_coverage = []
    all_arc_results = []

    for i in range(n_repeats):
        seed = base_seed + i

        if (i + 1) % 5 == 0:
            print(f"  Running experiment {i + 1}/{n_repeats}...")

        _, test_mse, arc_results, initial_coverage, _ = run_single_experiment(
            seed=seed,
            n_train=n_train,
            quantile=quantile,
            optimize_lengthscale_p=optimize_lengthscale_p,
            optimize_lengthscale_c=optimize_lengthscale_c,
            optimize_lengthscale_t=optimize_lengthscale_t,
            optimize_lengthscale_s=optimize_lengthscale_s,
            optimize_amplitude_p=optimize_amplitude_p,
            optimize_amplitude_c=optimize_amplitude_c,
            optimize_amplitude_t=optimize_amplitude_t,
            optimize_amplitude_s=optimize_amplitude_s,
            optimize_sigma=optimize_sigma,
            optimize_eta=optimize_eta,
            standardize_input=standardize_input,
            verbose=False
        )

        all_mse.append(test_mse)
        all_coverage.append(initial_coverage)
        all_arc_results.append(arc_results)

    all_mse = np.array(all_mse)
    all_coverage = np.array(all_coverage)

    all_accuracies = np.array([arc['accuracies'] for arc in all_arc_results])
    all_deltas = np.array([arc['delta'] for arc in all_arc_results])

    rejection_rates = all_arc_results[0]['rejection_rates'] if all_arc_results else None

    summary_stats = {
        'mse': {
            'mean': np.mean(all_mse),
            'std': np.std(all_mse),
            'median': np.median(all_mse),
            'min': np.min(all_mse),
            'max': np.max(all_mse)
        },
        'coverage': {
            'mean': np.mean(all_coverage),
            'std': np.std(all_coverage),
            'median': np.median(all_coverage),
            'min': np.min(all_coverage),
            'max': np.max(all_coverage)
        },
        'arc': {
            'mean_accuracies': np.mean(all_accuracies, axis=0),
            'std_accuracies': np.std(all_accuracies, axis=0),
            'mean_delta': np.mean(all_deltas),
            'std_delta': np.std(all_deltas)
        },
        'rejection_rates': rejection_rates
    }

    all_results = {
        'all_mse': all_mse,
        'all_coverage': all_coverage,
        'all_arc_results': all_arc_results,
        'summary_stats': summary_stats,
        'n_repeats': n_repeats,
        'n_train': n_train,
        'quantile': quantile,
        'base_seed': base_seed
    }

    return all_results

def visualize_single_experiment(gp, test_mse, arc_results, initial_coverage, history,
                               quantile=0.75, show_optimization=True, show_demand_curves=True,
                               show_arc=True):
    if show_optimization and len(history['losses']) > 0:
        print("\nVisualizing optimization history...")
        gp.visualize_optimization(history)

    if show_demand_curves:
        print("\nVisualizing demand curves...")
        gp.visualize_demand_curve_p(t_fixed=2.0, s_fixed=4)
        gp.visualize_demand_curve_t(p_fixed=20.0, s_fixed=4)

    if show_arc:
        print("\nVisualizing Accuracy Rejection Curve...")
        plt.figure(figsize=(10, 6))

        plt.plot(arc_results['rejection_rates'] * 100, arc_results['accuracies'] * 100,
                'g-', linewidth=3, label=f'ARC (Δ={arc_results["delta"]:.3f})')

        key_rejection_points = [0, 10, 20, 50, 80]
        for point in key_rejection_points:
            idx = np.argmin(np.abs(arc_results['rejection_rates'] * 100 - point))
            if idx < len(arc_results['accuracies']):
                plt.plot(arc_results['rejection_rates'][idx] * 100,
                        arc_results['accuracies'][idx] * 100, 'ro', markersize=8)
                plt.annotate(f'{arc_results["accuracies"][idx]*100:.1f}%',
                            (arc_results['rejection_rates'][idx] * 100,
                             arc_results['accuracies'][idx] * 100),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=9)

        plt.xlabel('Proportion of High-Variance Data Removed (%)')
        plt.ylabel(f'Proportion of Accurate Predictions (SE ≤ Q{quantile}) (%)')
        plt.title(f'GPIV Accuracy Rejection Curve (Single Experiment, Q{quantile})')
        plt.ylim(max(0, (quantile - 0.05) * 100), 100)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    print(f"\n=== Single Experiment Results ===")
    print(f"Test MSE (standardized scale): {test_mse:.6f}")
    print(f"Accuracy threshold (Δ, Q{quantile}): {arc_results['delta']:.4f}")
    print(f"Baseline Accuracy (0% removal): {arc_results['accuracies'][0]*100:.1f}%")
    print(f"Initial Coverage (0% removal): {initial_coverage*100:.1f}%")

def visualize_multiple_experiments(all_results):
    summary_stats = all_results['summary_stats']
    n_experiments = all_results['n_repeats']
    quantile = all_results['quantile']
    n_train = all_results['n_train']

    if summary_stats['rejection_rates'] is None:
        print("No ARC results to visualize")
        return

    rejection_rates = summary_stats['rejection_rates']
    mean_accuracies = summary_stats['arc']['mean_accuracies']
    std_accuracies = summary_stats['arc']['std_accuracies']
    mean_coverage = summary_stats['coverage']['mean']
    mean_mse = summary_stats['mse']['mean']

    confidence_interval = 1.96 * std_accuracies / np.sqrt(n_experiments)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    ax1.plot(rejection_rates * 100, mean_accuracies * 100, 'g-', linewidth=3, label='Mean Accuracy')
    ax1.fill_between(rejection_rates * 100,
                     (mean_accuracies - confidence_interval) * 100,
                     (mean_accuracies + confidence_interval) * 100,
                     alpha=0.3, color='green', label='95% CI')

    ax1.set_xlabel('Proportion of High-Variance Data Removed (%)')
    ax1.set_ylabel(f'Proportion of Accurate Predictions (SE ≤ Q{quantile}) (%)')
    ax1.set_title(f'Average ARC for GPIV Demand Design\n(ρ=0.5, n={n_train}, {n_experiments} Experiments)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(quantile * 100 - 5, 100)
    ax1.legend()

    key_points = [0, 10, 20, 50, 80]
    for point in key_points:
        idx = np.argmin(np.abs(rejection_rates * 100 - point))
        if idx < len(mean_accuracies):
            ax1.plot(rejection_rates[idx] * 100, mean_accuracies[idx] * 100, 'ro', markersize=8)
            ax1.annotate(f'{mean_accuracies[idx]*100:.1f}%',
                        (rejection_rates[idx] * 100, mean_accuracies[idx] * 100),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=9)

    ax2 = axes[0, 1]
    all_arc_results = all_results['all_arc_results']
    for i in range(min(25, n_experiments)):
        ax2.plot(all_arc_results[i]['rejection_rates'] * 100,
                all_arc_results[i]['accuracies'] * 100,
                'gray', alpha=0.2, linewidth=1)

    ax2.plot(rejection_rates * 100, mean_accuracies * 100, 'g-', linewidth=3, label='Mean')
    ax2.fill_between(rejection_rates * 100,
                     (mean_accuracies - confidence_interval) * 100,
                     (mean_accuracies + confidence_interval) * 100,
                     alpha=0.3, color='green', label='95% CI')

    ax2.set_xlabel('Proportion of High-Variance Data Removed (%)')
    ax2.set_ylabel(f'Proportion of Accurate Predictions (SE ≤ Q{quantile}) (%)')
    ax2.set_title(f'All ARC Traces for GPIV Demand Design\n({n_experiments} Experiments)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(quantile * 100 - 5, 100)
    ax2.legend()

    ax3 = axes[1, 0]
    mse_values = all_results['all_mse']
    ax3.hist(mse_values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(mean_mse, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_mse:.4f}')
    ax3.axvline(summary_stats['mse']['mean'] - summary_stats['mse']['std'], color='orange',
                linestyle=':', linewidth=1.5)
    ax3.axvline(summary_stats['mse']['mean'] + summary_stats['mse']['std'], color='orange',
                linestyle=':', linewidth=1.5, label=f'±1 std')

    ax3.set_xlabel('MSE (Standardized Scale)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'MSE Distribution ({n_experiments} Experiments)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = axes[1, 1]
    coverage_values = all_results['all_coverage']
    ax4.hist(coverage_values, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.axvline(mean_coverage, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_coverage*100:.1f}%')
    ax4.axvline(0.95, color='blue', linestyle='-', linewidth=2,
                label='Target: 95%', alpha=0.7)

    ax4.set_xlabel('Initial Coverage Rate (0% Removal)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'95% Coverage Distribution ({n_experiments} Experiments)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.suptitle(f'GPIV Demand Design Results\nρ=0.5, n={n_train}, {n_experiments} Experiments, Q{quantile}',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"MSE Statistics (Standardized Scale):")
    print(f"  Mean: {summary_stats['mse']['mean']:.6f}")
    print(f"  Std: {summary_stats['mse']['std']:.6f}")
    print(f"  Min: {summary_stats['mse']['min']:.6f}")
    print(f"  Max: {summary_stats['mse']['max']:.6f}")
    print(f"  95% CI for MSE: [{summary_stats['mse']['mean'] - 1.96*summary_stats['mse']['std']/np.sqrt(n_experiments):.6f}, "
          f"{summary_stats['mse']['mean'] + 1.96*summary_stats['mse']['std']/np.sqrt(n_experiments):.6f}]")

    print(f"\nCoverage Statistics (95% CI, 0% removal):")
    print(f"  Mean: {summary_stats['coverage']['mean']*100:.2f}%")
    print(f"  Std: {summary_stats['coverage']['std']*100:.2f}%")
    print(f"  Min: {summary_stats['coverage']['min']*100:.2f}%")
    print(f"  Max: {summary_stats['coverage']['max']*100:.2f}%")
    print(f"  95% CI for Coverage: [{summary_stats['coverage']['mean'] - 1.96*summary_stats['coverage']['std']/np.sqrt(n_experiments):.4f}, "
          f"{summary_stats['coverage']['mean'] + 1.96*summary_stats['coverage']['std']/np.sqrt(n_experiments):.4f}]")

    print(f"\nAccuracy Threshold (Δ, Q{quantile}):")
    print(f"  Mean: {summary_stats['arc']['mean_delta']:.4f}")
    print(f"  Std: {summary_stats['arc']['std_delta']:.4f}")

    print(f"\nARC Statistics at Key Points:")
    key_points = [0, 0.1, 0.2, 0.5, 0.8]
    for point in key_points:
        idx = np.argmin(np.abs(rejection_rates - point))
        if idx < len(mean_accuracies):
            accuracy_at_point = mean_accuracies[idx]
            std_at_point = summary_stats['arc']['std_accuracies'][idx]
            ci_width = 1.96 * std_at_point / np.sqrt(n_experiments)
            print(f"  At {point*100:.0f}% removed: Accuracy = {accuracy_at_point:.4f} ± {ci_width:.4f}")

    return fig

def run_single_experiment_with_bootstrap(seed=59, n_train=500, quantile=0.75,
                                        optimize_lengthscale_p=True, optimize_lengthscale_c=True,
                                        optimize_lengthscale_t=True, optimize_lengthscale_s=True,
                                        optimize_amplitude_p=False, optimize_amplitude_c=False,
                                        optimize_amplitude_t=False, optimize_amplitude_s=False,
                                        optimize_sigma=True, optimize_eta=False,
                                        standardize_input=True, verbose=True,
                                        n_bootstrap=200):
    if verbose:
        print(f"Running single experiment with bootstrap (seed {seed})...")
        print(f"Bootstrap iterations: {n_bootstrap}")

    Y_train, P_train, T_train, S_train, C_train = generate_demand_data(
        n_samples=n_train, rho=0.5, seed=seed
    )

    data_dict = {'Y': Y_train, 'P': P_train, 'T': T_train, 'S': S_train, 'C': C_train}
    scaled_data, scalers = standardize_data(data_dict)

    Y_scaled = scaled_data['Y']
    P_scaled = scaled_data['P']
    T_scaled = scaled_data['T']
    S_scaled = scaled_data['S']
    C_scaled = scaled_data['C']

    lengthscale_p = median_heuristic(P_scaled).item()
    lengthscale_c = median_heuristic(C_scaled).item()
    lengthscale_t = median_heuristic(T_scaled).item()
    lengthscale_s = median_heuristic(S_scaled).item()

    gp = BootstrapGPIV(
        lengthscale_p=lengthscale_p,
        lengthscale_c=lengthscale_c,
        lengthscale_t=lengthscale_t,
        lengthscale_s=lengthscale_s,
        amplitude_p=1.0, amplitude_c=1.0, amplitude_t=1.0, amplitude_s=1.0,
        sigma=0.5, eta=0.5,
        optimize_lengthscale_p=optimize_lengthscale_p,
        optimize_lengthscale_c=optimize_lengthscale_c,
        optimize_lengthscale_t=optimize_lengthscale_t,
        optimize_lengthscale_s=optimize_lengthscale_s,
        optimize_amplitude_p=optimize_amplitude_p,
        optimize_amplitude_c=optimize_amplitude_c,
        optimize_amplitude_t=optimize_amplitude_t,
        optimize_amplitude_s=optimize_amplitude_s,
        optimize_sigma=optimize_sigma,
        optimize_eta=optimize_eta,
        n_bootstrap=n_bootstrap
    )

    gp.set_data(P_scaled, C_scaled, T_scaled, S_scaled, Y_scaled, scalers)

    gp.compute_kernel_matrices()

    any_optimize = (optimize_lengthscale_p or optimize_lengthscale_c or
                   optimize_lengthscale_t or optimize_lengthscale_s or
                   optimize_amplitude_p or optimize_amplitude_c or
                   optimize_amplitude_t or optimize_amplitude_s or
                   optimize_sigma or optimize_eta)

    if any_optimize:
        if verbose:
            print("Optimizing hyperparameters for main model...")
        history = gp.optimize_hyperparameters(n_iterations=150, lr=0.1, verbose=verbose)
    else:
        history = {'losses': [], 'parameters': {}}

    if verbose:
        print("Running bootstrap to estimate variance...")
    gp.run_bootstrap(n_iterations=50, lr=0.1, verbose=verbose)

    test_mse = gp.compute_mse_with_bootstrap(standardize_input=standardize_input)

    if verbose:
        print("Computing ARC and coverage with bootstrap variance...")
    rejection_rates, accuracies, delta = gp.compute_accuracy_rejection_curve_with_bootstrap(
        delta=None, quantile=quantile, standardize_input=standardize_input
    )

    arc_results = {
        'rejection_rates': rejection_rates,
        'accuracies': accuracies,
        'delta': delta
    }

    initial_coverage = gp.compute_coverage_rate_with_bootstrap(
        confidence_level=0.95, standardize_input=standardize_input
    )

    return gp, test_mse, arc_results, initial_coverage, history

def run_multiple_experiments_with_bootstrap(n_repeats=25, n_train=500, quantile=0.75, base_seed=42,
                                           optimize_lengthscale_p=True, optimize_lengthscale_c=True,
                                           optimize_lengthscale_t=True, optimize_lengthscale_s=True,
                                           optimize_amplitude_p=False, optimize_amplitude_c=False,
                                           optimize_amplitude_t=False, optimize_amplitude_s=False,
                                           optimize_sigma=True, optimize_eta=False,
                                           standardize_input=True, n_bootstrap=200):
    print(f"Running {n_repeats} experiments with bootstrap variance...")
    print(f"Bootstrap iterations per experiment: {n_bootstrap}")

    all_mse = []
    all_coverage = []
    all_arc_results = []

    for i in range(n_repeats):
        seed = base_seed + i

        if (i + 1) % 5 == 0:
            print(f"  Running experiment {i + 1}/{n_repeats}...")

        _, test_mse, arc_results, initial_coverage, _ = run_single_experiment_with_bootstrap(
            seed=seed,
            n_train=n_train,
            quantile=quantile,
            optimize_lengthscale_p=optimize_lengthscale_p,
            optimize_lengthscale_c=optimize_lengthscale_c,
            optimize_lengthscale_t=optimize_lengthscale_t,
            optimize_lengthscale_s=optimize_lengthscale_s,
            optimize_amplitude_p=optimize_amplitude_p,
            optimize_amplitude_c=optimize_amplitude_c,
            optimize_amplitude_t=optimize_amplitude_t,
            optimize_amplitude_s=optimize_amplitude_s,
            optimize_sigma=optimize_sigma,
            optimize_eta=optimize_eta,
            standardize_input=standardize_input,
            verbose=False,
            n_bootstrap=n_bootstrap
        )

        all_mse.append(test_mse)
        all_coverage.append(initial_coverage)
        all_arc_results.append(arc_results)

    all_mse = np.array(all_mse)
    all_coverage = np.array(all_coverage)

    all_accuracies = np.array([arc['accuracies'] for arc in all_arc_results])
    all_deltas = np.array([arc['delta'] for arc in all_arc_results])

    rejection_rates = all_arc_results[0]['rejection_rates'] if all_arc_results else None

    summary_stats = {
        'mse': {
            'mean': np.mean(all_mse),
            'std': np.std(all_mse),
            'median': np.median(all_mse),
            'min': np.min(all_mse),
            'max': np.max(all_mse)
        },
        'coverage': {
            'mean': np.mean(all_coverage),
            'std': np.std(all_coverage),
            'median': np.median(all_coverage),
            'min': np.min(all_coverage),
            'max': np.max(all_coverage)
        },
        'arc': {
            'mean_accuracies': np.mean(all_accuracies, axis=0),
            'std_accuracies': np.std(all_accuracies, axis=0),
            'mean_delta': np.mean(all_deltas),
            'std_delta': np.std(all_deltas)
        },
        'rejection_rates': rejection_rates
    }

    all_results = {
        'all_mse': all_mse,
        'all_coverage': all_coverage,
        'all_arc_results': all_arc_results,
        'summary_stats': summary_stats,
        'n_repeats': n_repeats,
        'n_train': n_train,
        'quantile': quantile,
        'base_seed': base_seed,
        'n_bootstrap': n_bootstrap
    }

    return all_results

def visualize_single_experiment_with_bootstrap(gp, test_mse, arc_results, initial_coverage, history,
                                              quantile=0.75, show_optimization=True, show_arc=True):
    if show_optimization and len(history['losses']) > 0:
        print("\nVisualizing optimization history...")
        gp.visualize_optimization(history)

    if show_arc:
        print("\nVisualizing Accuracy Rejection Curve with Bootstrap Variance...")
        plt.figure(figsize=(10, 6))

        plt.plot(arc_results['rejection_rates'] * 100, arc_results['accuracies'] * 100,
                'g-', linewidth=3, label=f'ARC (Bootstrap, Δ={arc_results["delta"]:.3f})')

        key_rejection_points = [0, 10, 20, 50, 80]
        for point in key_rejection_points:
            idx = np.argmin(np.abs(arc_results['rejection_rates'] * 100 - point))
            if idx < len(arc_results['accuracies']):
                plt.plot(arc_results['rejection_rates'][idx] * 100,
                        arc_results['accuracies'][idx] * 100, 'ro', markersize=8)
                plt.annotate(f'{arc_results["accuracies"][idx]*100:.1f}%',
                            (arc_results['rejection_rates'][idx] * 100,
                             arc_results['accuracies'][idx] * 100),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha='center',
                            fontsize=9)

        plt.xlabel('Proportion of High-Variance Data Removed (%)')
        plt.ylabel(f'Proportion of Accurate Predictions (SE ≤ Q{quantile}) (%)')
        plt.title(f'GPIV Accuracy Rejection Curve with Bootstrap Variance (Single Experiment, Q{quantile})')
        plt.ylim(max(0, (quantile - 0.05) * 100), 100)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    print(f"\n=== Single Experiment Results with Bootstrap Variance ===")
    print(f"Test MSE (standardized scale): {test_mse:.6f}")
    print(f"Accuracy threshold (Δ, Q{quantile}): {arc_results['delta']:.4f}")
    print(f"Baseline Accuracy (0% removal): {arc_results['accuracies'][0]*100:.1f}%")
    print(f"Initial Coverage (0% removal): {initial_coverage*100:.1f}%")

def visualize_multiple_experiments_with_bootstrap(all_results):
    summary_stats = all_results['summary_stats']
    n_experiments = all_results['n_repeats']
    quantile = all_results['quantile']
    n_train = all_results['n_train']
    n_bootstrap = all_results.get('n_bootstrap', 200)

    if summary_stats['rejection_rates'] is None:
        print("No ARC results to visualize")
        return

    rejection_rates = summary_stats['rejection_rates']
    mean_accuracies = summary_stats['arc']['mean_accuracies']
    std_accuracies = summary_stats['arc']['std_accuracies']
    mean_coverage = summary_stats['coverage']['mean']
    mean_mse = summary_stats['mse']['mean']

    confidence_interval = 1.96 * std_accuracies / np.sqrt(n_experiments)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    ax1 = axes[0, 0]
    ax1.plot(rejection_rates * 100, mean_accuracies * 100, 'g-', linewidth=3, label='Mean Accuracy')
    ax1.fill_between(rejection_rates * 100,
                     (mean_accuracies - confidence_interval) * 100,
                     (mean_accuracies + confidence_interval) * 100,
                     alpha=0.3, color='green', label='95% CI')

    ax1.set_xlabel('Proportion of High-Variance Data Removed (%)')
    ax1.set_ylabel(f'Proportion of Accurate Predictions (SE ≤ Q{quantile}) (%)')
    ax1.set_title(f'Average ARC for GPIV with Bootstrap Variance\n(ρ=0.5, n={n_train}, {n_experiments} Experiments, {n_bootstrap} Bootstrap)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(quantile * 100 - 5, 100)
    ax1.legend()

    key_points = [0, 10, 20, 50, 80]
    for point in key_points:
        idx = np.argmin(np.abs(rejection_rates * 100 - point))
        if idx < len(mean_accuracies):
            ax1.plot(rejection_rates[idx] * 100, mean_accuracies[idx] * 100, 'ro', markersize=8)
            ax1.annotate(f'{mean_accuracies[idx]*100:.1f}%',
                        (rejection_rates[idx] * 100, mean_accuracies[idx] * 100),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=9)

    ax2 = axes[0, 1]
    all_arc_results = all_results['all_arc_results']
    for i in range(min(25, n_experiments)):
        ax2.plot(all_arc_results[i]['rejection_rates'] * 100,
                all_arc_results[i]['accuracies'] * 100,
                'gray', alpha=0.2, linewidth=1)

    ax2.plot(rejection_rates * 100, mean_accuracies * 100, 'g-', linewidth=3, label='Mean')
    ax2.fill_between(rejection_rates * 100,
                     (mean_accuracies - confidence_interval) * 100,
                     (mean_accuracies + confidence_interval) * 100,
                     alpha=0.3, color='green', label='95% CI')

    ax2.set_xlabel('Proportion of High-Variance Data Removed (%)')
    ax2.set_ylabel(f'Proportion of Accurate Predictions (SE ≤ Q{quantile}) (%)')
    ax2.set_title(f'All ARC Traces for GPIV with Bootstrap Variance\n({n_experiments} Experiments)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(quantile * 100 - 5, 100)
    ax2.legend()

    ax3 = axes[1, 0]
    mse_values = all_results['all_mse']
    ax3.hist(mse_values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(mean_mse, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_mse:.4f}')
    ax3.axvline(summary_stats['mse']['mean'] - summary_stats['mse']['std'], color='orange',
                linestyle=':', linewidth=1.5)
    ax3.axvline(summary_stats['mse']['mean'] + summary_stats['mse']['std'], color='orange',
                linestyle=':', linewidth=1.5, label=f'±1 std')

    ax3.set_xlabel('MSE (Standardized Scale)')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'MSE Distribution with Bootstrap Variance ({n_experiments} Experiments)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    ax4 = axes[1, 1]
    coverage_values = all_results['all_coverage']
    ax4.hist(coverage_values, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.axvline(mean_coverage, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_coverage*100:.1f}%')
    ax4.axvline(0.95, color='blue', linestyle='-', linewidth=2,
                label='Target: 95%', alpha=0.7)

    ax4.set_xlabel('Initial Coverage Rate (0% Removal)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'95% Coverage Distribution with Bootstrap Variance ({n_experiments} Experiments)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.suptitle(f'GPIV Demand Design Results with Bootstrap Variance\nρ=0.5, n={n_train}, {n_experiments} Experiments, Q{quantile}, {n_bootstrap} Bootstrap',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()

    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS WITH BOOTSTRAP VARIANCE")
    print(f"{'='*60}")
    print(f"MSE Statistics (Standardized Scale):")
    print(f"  Mean: {summary_stats['mse']['mean']:.6f}")
    print(f"  Std: {summary_stats['mse']['std']:.6f}")
    print(f"  Min: {summary_stats['mse']['min']:.6f}")
    print(f"  Max: {summary_stats['mse']['max']:.6f}")
    print(f"  95% CI for MSE: [{summary_stats['mse']['mean'] - 1.96*summary_stats['mse']['std']/np.sqrt(n_experiments):.6f}, "
          f"{summary_stats['mse']['mean'] + 1.96*summary_stats['mse']['std']/np.sqrt(n_experiments):.6f}]")

    print(f"\nCoverage Statistics (95% CI, 0% removal):")
    print(f"  Mean: {summary_stats['coverage']['mean']*100:.2f}%")
    print(f"  Std: {summary_stats['coverage']['std']*100:.2f}%")
    print(f"  Min: {summary_stats['coverage']['min']*100:.2f}%")
    print(f"  Max: {summary_stats['coverage']['max']*100:.2f}%")
    print(f"  95% CI for Coverage: [{summary_stats['coverage']['mean'] - 1.96*summary_stats['coverage']['std']/np.sqrt(n_experiments):.4f}, "
          f"{summary_stats['coverage']['mean'] + 1.96*summary_stats['coverage']['std']/np.sqrt(n_experiments):.4f}]")

    print(f"\nAccuracy Threshold (Δ, Q{quantile}):")
    print(f"  Mean: {summary_stats['arc']['mean_delta']:.4f}")
    print(f"  Std: {summary_stats['arc']['std_delta']:.4f}")

    print(f"\nARC Statistics at Key Points:")
    key_points = [0, 0.1, 0.2, 0.5, 0.8]
    for point in key_points:
        idx = np.argmin(np.abs(rejection_rates - point))
        if idx < len(mean_accuracies):
            accuracy_at_point = mean_accuracies[idx]
            std_at_point = summary_stats['arc']['std_accuracies'][idx]
            ci_width = 1.96 * std_at_point / np.sqrt(n_experiments)
            print(f"  At {point*100:.0f}% removed: Accuracy = {accuracy_at_point:.4f} ± {ci_width:.4f}")

    return fig

def main():
    print("=" * 70)
    print("Gaussian Process Instrumental Variable (GPIV) Demand Design Experiment")
    print("=" * 70)

    MODE = 'multiple_bootstrap'

    N_REPEATS = 20
    N_TRAIN = 200
    BASE_SEED = 42

    QUANTILE = 0.85

    N_BOOTSTRAP = 25

    OPTIMIZE_CONFIG = {
        'optimize_lengthscale_p': True,
        'optimize_lengthscale_c': False,
        'optimize_lengthscale_t': True,
        'optimize_lengthscale_s': True,
        'optimize_amplitude_p': False,
        'optimize_amplitude_c': False,
        'optimize_amplitude_t': False,
        'optimize_amplitude_s': False,
        'optimize_sigma': True,
        'optimize_eta': False,
        'standardize_input': True
    }

    if MODE == 'single':
        print("Running SINGLE experiment mode (original method)...")
        print(f"Parameters: n={N_TRAIN}, quantile={QUANTILE}, seed={BASE_SEED}")

        gp, mse, arc_results, coverage, history = run_single_experiment(
            seed=BASE_SEED,
            n_train=N_TRAIN,
            quantile=QUANTILE,
            **OPTIMIZE_CONFIG,
            verbose=True
        )

        visualize_single_experiment(
            gp, mse, arc_results, coverage, history,
            quantile=QUANTILE,
            show_optimization=True,
            show_demand_curves=True,
            show_arc=True
        )

        print(f"\nFinal Results:")
        print(f"  MSE (standardized): {mse:.6f}")
        print(f"  Initial Coverage: {coverage*100:.2f}%")

        return {
            'mse': mse,
            'coverage': coverage,
            'arc_results': arc_results
        }

    elif MODE == 'multiple':
        print("Running MULTIPLE experiments mode (original method)...")
        print(f"Parameters: {N_REPEATS} experiments, n={N_TRAIN}, quantile={QUANTILE}")
        print(f"Base seed: {BASE_SEED}")

        all_results = run_multiple_experiments(
            n_repeats=N_REPEATS,
            n_train=N_TRAIN,
            quantile=QUANTILE,
            base_seed=BASE_SEED,
            **OPTIMIZE_CONFIG
        )

        visualize_multiple_experiments(all_results)

        print(f"\nReturning results for {N_REPEATS} experiments.")
        print(f"  all_mse: {len(all_results['all_mse'])} values")
        print(f"  all_coverage: {len(all_results['all_coverage'])} values")
        print(f"  all_arc_results: {len(all_results['all_arc_results'])} ARC results")

        return all_results

    elif MODE == 'single_bootstrap':
        print("Running SINGLE experiment mode with BOOTSTRAP variance...")
        print(f"Parameters: n={N_TRAIN}, quantile={QUANTILE}, seed={BASE_SEED}")
        print(f"Bootstrap iterations: {N_BOOTSTRAP}")

        gp, mse, arc_results, coverage, history = run_single_experiment_with_bootstrap(
            seed=BASE_SEED,
            n_train=N_TRAIN,
            quantile=QUANTILE,
            **OPTIMIZE_CONFIG,
            verbose=True,
            n_bootstrap=N_BOOTSTRAP
        )

        visualize_single_experiment_with_bootstrap(
            gp, mse, arc_results, coverage, history,
            quantile=QUANTILE,
            show_optimization=True,
            show_arc=True
        )

        print(f"\nFinal Results with Bootstrap Variance:")
        print(f"  MSE (standardized): {mse:.6f}")
        print(f"  Initial Coverage: {coverage*100:.2f}%")

        return {
            'mse': mse,
            'coverage': coverage,
            'arc_results': arc_results
        }

    elif MODE == 'multiple_bootstrap':
        print("Running MULTIPLE experiments mode with BOOTSTRAP variance...")
        print(f"Parameters: {N_REPEATS} experiments, n={N_TRAIN}, quantile={QUANTILE}")
        print(f"Base seed: {BASE_SEED}")
        print(f"Bootstrap iterations per experiment: {N_BOOTSTRAP}")

        all_results = run_multiple_experiments_with_bootstrap(
            n_repeats=N_REPEATS,
            n_train=N_TRAIN,
            quantile=QUANTILE,
            base_seed=BASE_SEED,
            **OPTIMIZE_CONFIG,
            n_bootstrap=N_BOOTSTRAP
        )

        visualize_multiple_experiments_with_bootstrap(all_results)

        print(f"\nReturning results for {N_REPEATS} experiments with bootstrap variance.")
        print(f"  all_mse: {len(all_results['all_mse'])} values")
        print(f"  all_coverage: {len(all_results['all_coverage'])} values")
        print(f"  all_arc_results: {len(all_results['all_arc_results'])} ARC results")

        return all_results

    else:
        print(f"Unknown mode: {MODE}")
        print("Available modes: 'single', 'multiple', 'single_bootstrap', 'multiple_bootstrap'")

    print("\nExperiment completed!")

if __name__ == "__main__":
    results = main()