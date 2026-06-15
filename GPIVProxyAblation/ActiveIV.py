import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm
from torch.optim import Adam
import warnings

warnings.filterwarnings('ignore')


# ==================== DATA GENERATION ====================

def generate_data(n_samples=100, f_type='absolute', seed=None, mask_interval=[0, 0.20]):
    """
    Generate IV data with mask option
    mask_interval: [min, max] - mask data points where x falls in this interval
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Covariance Matrix
    covariance_matrix = torch.tensor([
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float64)

    # Generate Multivariate Gaussian
    mean = torch.zeros(3, dtype=torch.float64)
    multivariate_normal = torch.distributions.MultivariateNormal(mean, covariance_matrix)
    samples = multivariate_normal.sample((n_samples,))

    e, v, w = samples[:, 0], samples[:, 1], samples[:, 2]

    # Normal CDF
    normal_dist = torch.distributions.Normal(0, 1)
    x = normal_dist.cdf((w + v) / 2)
    z = normal_dist.cdf(w)

    # Generate true function h(x)
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

    # Add noise
    y = h_x + e

    # Apply mask if interval is specified
    if mask_interval is not None:
        mask_min, mask_max = mask_interval
        keep_mask = (x < mask_min) | (x > mask_max)
        x = x[keep_mask]
        y = y[keep_mask]
        z = z[keep_mask]
        h_x = h_x[keep_mask]

    return x, y, z, h_x


# ==================== HELPER FUNCTIONS ====================

def median_heuristic(X):
    """Compute median distance for RBF lengthscale initialization"""
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    dist = torch.cdist(X, X).triu(diagonal=1)
    return torch.median(dist[dist > 0])


def rbf_kernel(X1, X2, lengthscale=1.0):
    """RBF kernel calculation"""
    if len(X1.shape) == 1:
        X1 = X1.reshape(-1, 1)
    if len(X2.shape) == 1:
        X2 = X2.reshape(-1, 1)

    dist_sq = torch.cdist(X1, X2, p=2) ** 2
    K = torch.exp(-0.5 * dist_sq / (lengthscale ** 2))
    return K


# ==================== GPIV MODEL ====================

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

        # Cache for faster predictions
        self.cache_valid = False
        self.cached_invLzz = None
        self.cached_Lzz = None
        self.cached_Lxz = None

    def set_data(self, X, Z, y):
        self.X_train = X
        self.Z_train = Z
        self.y_train = y
        self.cache_valid = False

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

        # Update cache
        self.cache_valid = True
        self.cached_invLzz = self.invLzz.clone()
        self.cached_Lzz = self.Lzz.clone()
        self.cached_Lxz = self.Lxz.clone()

    def predict_batch(self, x_new, z_new=None):
        """Predict for multiple points"""
        if len(x_new.shape) == 1:
            x_new = x_new.reshape(-1, 1)

        if not self.cache_valid:
            self.compute_kernel_matrices()

        Kx = rbf_kernel(x_new, self.X_train, self.lengthscale_x)
        kxx_prime = rbf_kernel(x_new, x_new, self.lengthscale_x)

        mean = Kx @ self.invLzz @ self.Lzz @ self.Lxz @ self.y_train.reshape(-1, 1)
        variance = torch.diag(kxx_prime - Kx @ self.invLzz @ self.Lzz @ self.Lxz @ self.Lzz @ self.invLzz @ Kx.T)

        return mean.squeeze(), variance.squeeze()

    def compute_total_posterior_variance(self, x_points):
        """Compute total posterior variance for given points"""
        _, variances = self.predict_batch(x_points)
        return torch.sum(variances).item()

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
            _, log_det_val = torch.slogdet(K_full)
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

        if not params_to_optimize:
            return

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

            if verbose and i % 50 == 0:
                print(f"  Iteration {i}, Loss: {loss.item():.4f}")

    def compute_mse(self, x_test, true_function):
        """Compute MSE on test points"""
        pred_means, _ = self.predict_batch(x_test)
        mse = torch.mean((true_function - pred_means) ** 2).item()
        return mse


# ==================== ACTIVE LEARNING ====================

def run_active_learning_experiment(
        f_type='absolute',
        n_initial=200,
        n_pool=200,
        n_test=50,
        test_interval=[0.0, 1.0],
        mask_interval=[0, 0.20],
        n_select_per_iter=10,
        max_iterations=20,
        seed=42,
        n_iterations_train=100,
        lr=0.02,
        train_lengthscale_x=True,
        train_lengthscale_z=False,
        train_sigma=True,
        train_eta=False
):
    """
    Run active learning experiment for GPIV

    Steps:
    1. Generate initial data (with optional mask)
    2. Generate pool data
    3. Generate test data
    4. Train initial model on initial data
    5. Repeat until pool is empty or max_iterations reached:
        a. For each data point in pool, add it to training temporarily and compute total posterior variance on pool
        b. Sort by total variance (loss) and select top n_select_per_iter points with smallest loss
        c. Add selected points to training data, remove from pool
        d. Retrain model on new training data
        e. Compute MSE on test data
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"Running Active Learning Experiment:")
    print(f"  Design: {f_type}")
    print(f"  Initial data: {n_initial} points")
    print(f"  Pool data: {n_pool} points")
    print(f"  Test data: {n_test} points on interval {test_interval}")
    print(f"  Mask interval: {mask_interval}")
    print(f"  Active learning: Select {n_select_per_iter} points per iteration")

    # ========== Step 1-3: Generate data ==========
    print("\n1. Generating data...")

    # Generate initial data
    x_initial, y_initial, z_initial, _ = generate_data(
        n_initial, f_type=f_type, seed=seed, mask_interval=mask_interval
    )

    # Generate pool data (different seed)
    x_pool, y_pool, z_pool, _ = generate_data(
        n_pool, f_type=f_type, seed=seed + 1000, mask_interval=mask_interval
    )

    # Generate test data
    test_min, test_max = test_interval
    x_test = torch.linspace(test_min, test_max, n_test, dtype=torch.float64)

    # True function on test points
    if f_type == 'sine':
        true_function = 2 * torch.sin(2 * torch.pi * x_test)
    elif f_type == 'log':
        true_function = torch.log(torch.abs(16 * x_test - 8) + 1) * torch.sign(x_test - 0.5)
    elif f_type == 'linear':
        true_function = 4 * x_test - 2
    elif f_type == 'absolute':
        true_function = torch.abs(4 * x_test - 2) - 1

    # Convert pool data to list of points for easier manipulation
    pool_points = []
    for i in range(len(x_pool)):
        pool_points.append({
            'x': x_pool[i].reshape(1, -1),
            'z': z_pool[i].reshape(1, -1),
            'y': y_pool[i].reshape(1)
        })

    # ========== Step 4: Train initial model ==========
    print("\n2. Training initial model...")

    # Initialize GPIV
    lengthscale_x = median_heuristic(x_initial).item()
    lengthscale_z = median_heuristic(z_initial).item()

    model = GPIV(
        lengthscale_x=lengthscale_x,
        lengthscale_z=lengthscale_z,
        sigma=1.2,
        eta=1.0,
        train_sigma=train_sigma,
        train_eta=train_eta,
        lx=train_lengthscale_x,
        lz=train_lengthscale_z
    )

    # Set initial data
    model.set_data(x_initial.reshape(-1, 1), z_initial.reshape(-1, 1), y_initial)
    model.optimize_hyperparameters(n_iterations=n_iterations_train, lr=lr, verbose=True)

    # Compute initial MSE
    initial_mse = model.compute_mse(x_test, true_function)
    print(f"  Initial MSE: {initial_mse:.6f}")

    # ========== Step 5: Active Learning Loop ==========
    print("\n3. Starting active learning loop...")

    # Results tracking
    mse_history = [initial_mse]
    training_size_history = [len(x_initial)]
    selected_indices_history = []

    # Prepare training data as lists for easy appending
    X_train = [x_initial.reshape(-1, 1)]
    Z_train = [z_initial.reshape(-1, 1)]
    y_train = [y_initial]

    iteration = 0
    while len(pool_points) > 0 and iteration < max_iterations:
        iteration += 1
        print(f"\n  Iteration {iteration}: Pool size = {len(pool_points)}")

        # If remaining pool is smaller than selection size, adjust
        current_select = min(n_select_per_iter, len(pool_points))

        # ========== Step 5a: Compute loss for each pool point ==========
        print(f"    Computing losses for {len(pool_points)} pool points...")

        losses = []

        # Pre-compute current model's predictions on pool (for variance computation)
        # We'll compute the effect of adding each point individually
        for i, point in enumerate(pool_points):
            # Create temporary training set with this point added
            X_temp = torch.cat([torch.cat(X_train, dim=0), point['x']], dim=0)
            Z_temp = torch.cat([torch.cat(Z_train, dim=0), point['z']], dim=0)
            y_temp = torch.cat([torch.cat(y_train, dim=0), point['y']], dim=0)

            # Create temporary model with same hyperparameters
            temp_model = GPIV(
                lengthscale_x=model.lengthscale_x.item(),
                lengthscale_z=model.lengthscale_z.item(),
                sigma=model.sigma.item(),
                eta=model.eta.item(),
                train_sigma=True,  # Don't retrain hyperparameters
                train_eta=False,
                lx=False,
                lz=False
            )

            temp_model.set_data(X_temp, Z_temp, y_temp)
            temp_model.compute_kernel_matrices()

            # Compute loss = total posterior variance on all pool points
            # We'll use a subset of pool points to speed up computation
            # Use all pool points' x values
            pool_x = torch.cat([p['x'] for p in pool_points], dim=0)
            loss = temp_model.compute_total_posterior_variance(pool_x)
            losses.append(loss)

            if i % 20 == 0:
                print(f"      Processed {i}/{len(pool_points)} points...")

        # ========== Step 5b: Select points with smallest loss ==========
        # Sort by loss (ascending - smaller loss is better)
        sorted_indices = np.argsort(losses)[:current_select]
        selected_points = [pool_points[i] for i in sorted_indices]

        print(f"    Selected {current_select} points with losses: {[losses[i] for i in sorted_indices]}")

        # ========== Step 5c: Update training data and pool ==========
        # Add selected points to training data
        for point in selected_points:
            X_train.append(point['x'])
            Z_train.append(point['z'])
            y_train.append(point['y'])

        # Remove selected points from pool
        pool_points = [p for i, p in enumerate(pool_points) if i not in sorted_indices]

        # ========== Step 5d: Retrain model ==========
        print(f"    Retraining model with {sum([x.shape[0] for x in X_train])} training points...")

        # Create new training tensors
        X_train_tensor = torch.cat(X_train, dim=0)
        Z_train_tensor = torch.cat(Z_train, dim=0)
        y_train_tensor = torch.cat(y_train, dim=0)

        # Reinitialize model with updated data
        model = GPIV(
            lengthscale_x=model.lengthscale_x.item(),
            lengthscale_z=model.lengthscale_z.item(),
            sigma=model.sigma.item(),
            eta=model.eta.item(),
            train_sigma=train_sigma,
            train_eta=train_eta,
            lx=train_lengthscale_x,
            lz=train_lengthscale_z
        )

        model.set_data(X_train_tensor, Z_train_tensor, y_train_tensor)
        model.optimize_hyperparameters(n_iterations=n_iterations_train, lr=lr, verbose=False)

        # ========== Step 5e: Compute MSE ==========
        current_mse = model.compute_mse(x_test, true_function)
        mse_history.append(current_mse)
        training_size_history.append(len(X_train_tensor))
        selected_indices_history.append(sorted_indices.tolist())

        print(f"    MSE after iteration {iteration}: {current_mse:.6f}")
        print(f"    Training size: {len(X_train_tensor)}")

    print(f"\nActive learning completed after {iteration} iterations")

    return {
        'mse_history': mse_history,
        'training_size_history': training_size_history,
        'selected_indices_history': selected_indices_history,
        'initial_mse': initial_mse,
        'final_mse': mse_history[-1],
        'final_training_size': training_size_history[-1],
        'f_type': f_type,
        'n_initial': n_initial,
        'n_pool': n_pool,
        'n_test': n_test,
        'test_interval': test_interval,
        'mask_interval': mask_interval,
        'n_select_per_iter': n_select_per_iter,
        'n_iterations': iteration,
        'model': model,
        'x_test': x_test,
        'true_function': true_function
    }


# ==================== PLOTTING FUNCTIONS ====================

def plot_active_learning_results(results, plot_predictions=True):
    """
    Plot results from active learning experiment
    """
    mse_history = results['mse_history']
    training_size_history = results['training_size_history']

    # Create figure with 2 or 3 subplots
    if plot_predictions:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: MSE vs Training Size
    ax1 = axes[0]
    ax1.plot(training_size_history, mse_history, 'bo-', linewidth=2, markersize=8)
    ax1.scatter(training_size_history[0], mse_history[0], color='green', s=150,
                zorder=5, label=f'Initial: {mse_history[0]:.4f}')
    ax1.scatter(training_size_history[-1], mse_history[-1], color='red', s=150,
                zorder=5, label=f'Final: {mse_history[-1]:.4f}')

    ax1.set_xlabel('Training Set Size', fontsize=12)
    ax1.set_ylabel('MSE', fontsize=12)
    ax1.set_title('MSE vs Training Set Size', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: MSE vs Iteration
    ax2 = axes[1]
    iterations = list(range(len(mse_history)))
    ax2.plot(iterations, mse_history, 'ro-', linewidth=2, markersize=8)

    ax2.set_xlabel('Active Learning Iteration', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_title('MSE vs Active Learning Iteration', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(iterations)

    # Add percentage improvement
    improvement = 100 * (mse_history[0] - mse_history[-1]) / mse_history[0]
    ax2.text(0.05, 0.95, f'Improvement: {improvement:.1f}%',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Final predictions if requested
    if plot_predictions:
        ax3 = axes[2]

        model = results['model']
        x_test = results['x_test']
        true_function = results['true_function']

        # Get predictions
        pred_means, pred_vars = model.predict_batch(x_test.reshape(-1, 1))
        pred_means = pred_means.detach().numpy()
        pred_vars = pred_vars.detach().numpy()
        x_test_np = x_test.detach().numpy()
        true_function_np = true_function.detach().numpy()

        # Calculate confidence intervals
        pred_stds = np.sqrt(np.maximum(pred_vars, 1e-10))
        z_score = norm.ppf(0.975)
        lower = pred_means - z_score * pred_stds
        upper = pred_means + z_score * pred_stds

        # Plot
        ax3.plot(x_test_np, true_function_np, 'k-', linewidth=3, label='True Function')
        ax3.plot(x_test_np, pred_means, 'r-', linewidth=2, label='GPIV Prediction')
        ax3.fill_between(x_test_np, lower, upper, alpha=0.3, color='red', label='95% CI')

        # Highlight test interval
        test_min, test_max = results['test_interval']
        ax3.axvspan(test_min, test_max, alpha=0.1, color='blue', label='Test Interval')

        ax3.set_xlabel('X', fontsize=12)
        ax3.set_ylabel('Y', fontsize=12)
        ax3.set_title(f'Final Predictions (MSE: {results["final_mse"]:.4f})', fontsize=14)
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

    # Overall title
    title_parts = [
        f"Active Learning for GPIV (Design: {results['f_type']})",
        f"Initial: {results['n_initial']} points, Pool: {results['n_pool']} points, Test: {results['n_test']} points on {results['test_interval']}"
    ]

    if results.get('mask_interval'):
        title_parts.append(f"Mask: {results['mask_interval']}")

    plt.suptitle("\n".join(title_parts), fontsize=16, y=1.05)
    plt.tight_layout()

    return fig


# ==================== MAIN FUNCTION ====================

def run_active_learning_analysis():
    """
    Main function to run active learning experiment with user-defined configuration
    """
    # ===== USER CONFIGURATION =====
    # Choose design: 'absolute', 'linear', 'sine', or 'log'
    DESIGN = 'sine'

    # Data sizes
    N_INITIAL = 200  # Initial training data
    N_POOL = 200  # Pool data for active learning
    N_TEST = 40  # Test points for evaluation

    # Data intervals
    MASK_INTERVAL = [0, 0.0]  # Mask for training data (set to None for no mask)
    TEST_INTERVAL = [0.0, 1]  # Interval for test points

    # Active learning settings
    N_SELECT_PER_ITER = 10  # Number of points to select each iteration
    MAX_ITERATIONS = 10  # Maximum number of active learning iterations

    # Training settings
    N_ITERATIONS_TRAIN = 50  # Number of iterations for hyperparameter optimization
    LEARNING_RATE = 0.02

    # Which hyperparameters to tune
    TUNE_LENGTHSCALE_X = False
    TUNE_LENGTHSCALE_Z = False
    TUNE_SIGMA = True
    TUNE_ETA = False

    # Random seed
    SEED = 42

    # Plot settings
    PLOT_PREDICTIONS = True  # Whether to show final predictions plot
    # ================================

    print("=" * 60)
    print("ACTIVE LEARNING FOR GPIV")
    print("=" * 60)

    # Run the active learning experiment
    results = run_active_learning_experiment(
        f_type=DESIGN,
        n_initial=N_INITIAL,
        n_pool=N_POOL,
        n_test=N_TEST,
        test_interval=TEST_INTERVAL,
        mask_interval=MASK_INTERVAL,
        n_select_per_iter=N_SELECT_PER_ITER,
        max_iterations=MAX_ITERATIONS,
        seed=SEED,
        n_iterations_train=N_ITERATIONS_TRAIN,
        lr=LEARNING_RATE,
        train_lengthscale_x=TUNE_LENGTHSCALE_X,
        train_lengthscale_z=TUNE_LENGTHSCALE_Z,
        train_sigma=TUNE_SIGMA,
        train_eta=TUNE_ETA
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Initial MSE (with {N_INITIAL} points): {results['initial_mse']:.6f}")
    print(f"Final MSE (with {results['final_training_size']} points): {results['final_mse']:.6f}")
    print(f"Improvement: {100 * (results['initial_mse'] - results['final_mse']) / results['initial_mse']:.1f}%")
    print(f"Active learning iterations: {results['n_iterations']}")
    print(f"Total points added: {results['final_training_size'] - N_INITIAL}")

    # Plot results
    print("\nGenerating plots...")
    fig = plot_active_learning_results(results, plot_predictions=PLOT_PREDICTIONS)
    plt.show()

    return results


# Run the analysis
if __name__ == "__main__":
    results = run_active_learning_analysis()