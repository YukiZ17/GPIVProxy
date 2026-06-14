import torch
import numpy as np
import matplotlib.pyplot as plt
from data_generation import generate_data, median_heuristic
from gpiv_model import GPIV
from metrics import compute_auc_arc, compute_mean_ci_width

def run_single_gpiv_demo(f_type='absolute', n_samples=300, seed=91,
                         train_lengthscale_x=True, train_lengthscale_z=True,
                         train_sigma=True, eta=1.0, n_iterations=150, lr=0.02,
                         n_test_points=200, quantile_threshold=0.75):
    print(f"\n{'='*60}")
    print(f"SINGLE GPIV DEMO: {f_type.upper()} FUNCTION")
    print(f"{'='*60}")
    print(f"Training lengthscale_x: {train_lengthscale_x}")
    print(f"Training lengthscale_z: {train_lengthscale_z}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    x, y, z, h_true = generate_data(n_samples, f_type=f_type, seed=seed)

    lengthscale_x = median_heuristic(x).item()
    lengthscale_z = median_heuristic(z).item()
    print(f"Median heuristic lengthscales: x={lengthscale_x:.3f}, z={lengthscale_z:.3f}")

    gp = GPIV(lengthscale_x=lengthscale_x, lengthscale_z=lengthscale_z,
              sigma=1.2, eta=eta, train_sigma=train_sigma, train_eta=False,
              lx=train_lengthscale_x, lz=train_lengthscale_z)
    gp.set_data(x.reshape(-1,1), z.reshape(-1,1), y)
    gp.compute_kernel_matrices()
    print(f"\nInitial NLL: {gp.negative_log_marginal_likelihood().item():.3f}")

    history = gp.optimize_hyperparameters(n_iterations=n_iterations, lr=lr, verbose=True)
    print(f"\nFinal NLL: {gp.negative_log_marginal_likelihood().item():.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].plot(history['losses'])
    axes[0,0].set_xlabel('Iterations')
    axes[0,0].set_ylabel('Negative Log Likelihood')
    axes[0,0].set_title('Loss Curve')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,1].plot(history['lengthscales_x'])
    axes[0,1].set_xlabel('Iterations')
    axes[0,1].set_ylabel('Lengthscale')
    axes[0,1].set_title('Lengthscale X')
    axes[0,1].grid(True, alpha=0.3)
    axes[1,0].plot(history['lengthscales_z'])
    axes[1,0].set_xlabel('Iterations')
    axes[1,0].set_ylabel('Lengthscale')
    axes[1,0].set_title('Lengthscale Z')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(history['sigmas'])
    axes[1,1].set_xlabel('Iterations')
    axes[1,1].set_ylabel('Sigma')
    axes[1,1].set_title('Noise Sigma')
    axes[1,1].grid(True, alpha=0.3)
    plt.suptitle('GPIV Hyperparameter Optimization History', fontsize=14)
    plt.tight_layout()
    plt.show()

    x_test = torch.linspace(0, 1, n_test_points, dtype=torch.float64)
    if f_type == 'sine':
        h_x_true = 2 * torch.sin(2 * torch.pi * x_test)
    elif f_type == 'log':
        h_x_true = torch.log(torch.abs(16 * x_test - 8) + 1) * torch.sign(x_test - 0.5)
    elif f_type == 'linear':
        h_x_true = 4 * x_test - 2
    elif f_type == 'absolute':
        h_x_true = torch.abs(4 * x_test - 2) - 1

    means = []
    variances = []
    for x_val in x_test:
        mean, var = gp.predict(x_val)
        means.append(mean.item())
        variances.append(var.item())
    means = torch.tensor(means, dtype=torch.float64)
    variances = torch.tensor(variances, dtype=torch.float64)
    stds = torch.sqrt(variances)

    plt.figure(figsize=(12, 8))
    plt.plot(x_test.numpy(), means.numpy(), 'b-', linewidth=2, label='Posterior Mean')
    plt.fill_between(x_test.numpy(),
                     (means - 2 * stds).numpy(),
                     (means + 2 * stds).numpy(),
                     alpha=0.2, color='blue', label='95% CI')
    plt.plot(x_test.numpy(), h_x_true.numpy(), 'orange', linewidth=2, label=f'True h(x): {f_type}')
    plt.scatter(x.numpy(), y.numpy(), color='red', s=20, alpha=0.5, label='Training Data')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'GPIV Posterior Distribution: {f_type} function\n'
              f'MSE: {torch.mean((h_x_true - means)**2).item():.4f}, '
              f'lengthscale_x={gp.lengthscale_x.item():.3f}, lengthscale_z={gp.lengthscale_z.item():.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.show()

    se = []
    var = []
    for x_val in x_test:
        mean, variance = gp.predict(x_val)
        se.append((h_x_true[x_test == x_val] - mean.item())**2)
        var.append(variance.item())
    se = torch.tensor(se, dtype=torch.float64)
    var = torch.tensor(var, dtype=torch.float64)

    sorted_indices = torch.argsort(var, descending=True)
    se_sorted = se[sorted_indices]
    n_per_step = max(1, n_test_points // 50)
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

    auc_arc = compute_auc_arc(removed_ratios, accurate_ratios)
    mean_ci_width = compute_mean_ci_width(gp, x_test, confidence=0.95)
    mse = torch.mean(se).item()
    coverage = gp.compute_coverage(x_test, h_x_true, confidence=0.95)

    plt.figure(figsize=(10, 6))
    plt.plot(removed_ratios, accurate_ratios, 'b-', linewidth=2, label='ARC')
    plt.fill_between(removed_ratios, 0, accurate_ratios, alpha=0.2, color='blue')
    plt.xlabel('Proportion of High-Variance Data Removed')
    plt.ylabel(f'Proportion of Accurate Predictions (SE ≤ Q{quantile_threshold})')
    plt.title(f'ARC Curve for {f_type} function (AUC = {auc_arc:.4f})')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nGPIV Results for Single Experiment:")
    print(f"MSE: {mse:.6f}")
    print(f"Coverage (95% CI): {coverage*100:.2f}%")
    print(f"AUC of ARC: {auc_arc:.4f}")
    print(f"Mean 95% CI Width: {mean_ci_width:.4f}")

    return gp