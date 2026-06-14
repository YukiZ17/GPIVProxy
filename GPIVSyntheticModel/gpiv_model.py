import torch
from torch.optim import Adam
from scipy.stats import norm
from data_generation import rbf_kernel

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
            (self.sigma**2) * torch.eye(len(self.Z_train), dtype=torch.float64)
        )

    def posterior_mean(self, x_new):
        Kx = rbf_kernel(x_new.reshape(1, -1), self.X_train, self.lengthscale_x)
        mean = Kx @ self.invLzz @ self.Lzz @ self.Lxz @ self.y_train.reshape(-1, 1)
        return mean.squeeze()

    def posterior_variance(self, x_new):
        Kx = rbf_kernel(x_new.reshape(1, -1), self.X_train, self.lengthscale_x)
        kxx_prime = rbf_kernel(x_new.reshape(1, -1), x_new.reshape(1, -1), self.lengthscale_x)
        variance = kxx_prime - Kx @ self.invLzz @ self.Lzz @ self.Lxz @ self.Lzz @ self.invLzz @ Kx.T
        return variance.squeeze()

    def predict(self, x_new):
        mean = self.posterior_mean(x_new)
        variance = self.posterior_variance(x_new)
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
        history = {'losses': [], 'lengthscales_x': [], 'lengthscales_z': [],
                   'sigmas': [], 'etas': []}
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
            if verbose and (i+1) % 50 == 0:
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