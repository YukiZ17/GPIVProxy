import torch
import numpy as np
from scipy.stats import norm
from torch.optim import Adam
from data_generation import rbf_kernel, h_function

class GPIV:
    def __init__(self, lengthscale_p=1.0, lengthscale_c=1.0, lengthscale_t=1.0,
                 lengthscale_s=1.0, amplitude_p=1.0, amplitude_c=1.0,
                 amplitude_t=1.0, amplitude_s=1.0, sigma=0.5, eta=0.1,
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
        return data_fit + log_det

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
        optimizer = Adam(params, lr=lr)
        history = {'losses': [], 'parameters': {name: [] for name in param_names}}
        if verbose:
            print("Starting hyperparameter optimization...")
            print(f"Parameters to optimize: {', '.join(param_names)}")
            print("Initial values:")
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
            history['losses'].append(loss.item())
            for name in param_names:
                history['parameters'][name].append(getattr(self, name).item())
            if verbose and (i + 1) % 10 == 0:
                print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.3f}")
                for name in param_names:
                    value = getattr(self, name).item()
                    print(f"  {name}: {value:.3f}", end=' ')
                print()
        if verbose:
            print(f"Optimization completed. Final values:")
            for name in param_names:
                value = getattr(self, name).item()
                print(f"  {name}: {value:.3f}")
        return history

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
                    true_f = 100 + (10 + p) * s * h_t - 2 * p
                    if self.scalers is not None and 'Y' in self.scalers:
                        true_f_scaled = (true_f - self.scalers['Y']['mean'].item()) / self.scalers['Y']['std'].item()
                    else:
                        true_f_scaled = true_f
                    pred_f, var_f = self.predict(p, t, s, standardize_input=standardize_input)
                    predictions.append(pred_f.item())
                    variances.append(var_f.item())
                    true_values.append(true_f_scaled)
        return np.array(predictions), np.array(variances), np.array(true_values)

    def compute_mse(self, standardize_input=True):
        predictions, _, true_values = self.compute_test_predictions(standardize_input)
        return np.mean((predictions - true_values)**2)

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
        return np.mean(in_ci)

def set_seed(seed=442):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False