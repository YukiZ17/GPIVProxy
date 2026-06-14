import torch
import numpy as np
from torch.optim import Adam
from data_generation import rbf_kernel, unstandardize_data, compute_true_structural_function

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

    def set_data(self, z, u, t, y, t_mean=None, t_std=None, y_mean=None, y_std=None):
        self.u = u.double()
        self.z = z.double()
        self.y = y.double().reshape(-1, 1)
        self.t = t.double()
        self.t_mean = t_mean if t_mean is not None else torch.tensor(0.0, dtype=torch.float64)
        self.t_std = t_std if t_std is not None else torch.tensor(1.0, dtype=torch.float64)
        self.y_mean = y_mean if y_mean is not None else torch.tensor(0.0, dtype=torch.float64)
        self.y_std = y_std if y_std is not None else torch.tensor(1.0, dtype=torch.float64)
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
        self.C = self.Ktt * (self.Ktz @ self.invKtz @ self.Kuu @ self.invKtz @ self.Ktz) + (self.sigma)**2 * torch.eye(len(self.z), dtype=torch.float64)
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

    def posterior_variance(self, t_new, standardized=True):
        return self.posterior_covariance(t_new, t_new, standardized)

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
        return data_fit + log_det

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
            if verbose:
                print("Warning: No parameters to optimize!")
            return {'losses': [], 'lengthscales_t': [], 'lengthscales_u': [], 'lengthscales_z': [], 'sigmas': []}
        optimizer = Adam(params, lr=lr)
        losses = []
        lengthscales_t = []
        lengthscales_u = []
        lengthscales_z = []
        sigmas = []
        if verbose:
            print("Hyperparameter optimization starts...")
            print(f"Initial Value: lengthscale_t={self.lengthscale_t.item():.3f}, lengthscale_u={self.lengthscale_u.item():.3f}, lengthscale_z={self.lengthscale_z.item():.3f}, sigma={self.sigma.item():.3f}, eta={self.eta.item():.3f} (fixed)")
            print(f"Optimizing: lengthscale_t={self.optimize_t}, lengthscale_u={self.optimize_u}, lengthscale_z={self.optimize_z}, sigma={self.optimize_sigma}")
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
                print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.3f}, lengthscale_t={self.lengthscale_t.item():.3f}, lengthscale_u={self.lengthscale_u.item():.3f}, lengthscale_z={self.lengthscale_z.item():.3f}, sigma={self.sigma.item():.3f}")
        if verbose:
            print(f"optimization finished: lengthscale_t={self.lengthscale_t.item():.3f}, lengthscale_u={self.lengthscale_u.item():.3f}, lengthscale_z={self.lengthscale_z.item():.3f}, sigma={self.sigma.item():.3f}, eta={self.eta.item():.3f} (fixed)")
        return {'losses': losses, 'lengthscales_t': lengthscales_t, 'lengthscales_u': lengthscales_u, 'lengthscales_z': lengthscales_z, 'sigmas': sigmas}

    def compute_true_structural_function(self, t_values, n_mc=10000, standardized=True):
        return compute_true_structural_function(t_values, self.t_mean, self.t_std, self.y_mean, self.y_std, n_mc, standardized)

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
        lower = predictions - 1.96 * stds
        upper = predictions + 1.96 * stds
        coverage_mask = (h_true >= lower) & (h_true <= upper)
        coverage = torch.mean(coverage_mask.float()).item()
        if standardized:
            mse_std = mse_orig
        else:
            predictions_std = (predictions - self.y_mean) / self.y_std
            h_true_std = (h_true - self.y_mean) / self.y_std
            mse_std = torch.mean((predictions_std - h_true_std)**2)
        return mae.item(), mse_orig.item(), mse_std.item(), predictions, variances, errors, coverage

    def compute_arc_analysis(self, t_test_start=-2, t_test_end=4, n_points=400, quantile_p=0.75, remove_step=10):
        t_test_orig = torch.linspace(t_test_start, t_test_end, n_points, dtype=torch.float64)
        h_true_orig = self.compute_true_structural_function(t_test_orig, n_mc=10000, standardized=False)
        mae, mse_orig, mse_std, predictions, variances, errors, coverage = self.evaluate_predictions(t_test_orig, h_true_orig, standardized=False)
        quantile_threshold = torch.quantile(errors, quantile_p).item()
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
        return {'proportions_below_threshold': proportions_below_threshold, 'removed_counts': removed_counts, 'quantile_threshold': quantile_threshold, 'final_auc': final_auc, 'auc_vector': auc_vector, 'coverage': coverage, 'mae': mae, 'mse_orig': mse_orig, 'mse_std': mse_std, 'errors': errors.numpy(), 'variances': variances.numpy(), 'predictions': predictions.numpy(), 'true_values': h_true_orig.numpy()}