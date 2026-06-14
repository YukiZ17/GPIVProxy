import torch
import numpy as np
import matplotlib.pyplot as plt
from data_generation import generate_data, median_heuristic
from gpiv_model import GPIV
from metrics import compute_auc_arc

def run_gpiv_multiple_experiments(n_experiments=25, f_type='absolute', n_samples=300,
                                  n_test_points=200, quantile_threshold=0.75,
                                  n_iterations=150, lr=0.02,
                                  train_lengthscale_x=True, train_lengthscale_z=True,
                                  train_sigma=True, eta=1.0, seed_head=500):
    print(f"\n{'='*60}")
    print(f"RUNNING {n_experiments} GPIV EXPERIMENTS")
    print(f"{'='*60}")
    print(f"Function type: {f_type}")
    print(f"Sample size per experiment: {n_samples}")
    print(f"Test points: {n_test_points}")
    print(f"Quantile threshold for accuracy: {quantile_threshold}")
    print(f"Training lengthscale_x: {train_lengthscale_x}")
    print(f"Training lengthscale_z: {train_lengthscale_z}")

    all_accurate_ratios = []
    all_removed_ratios = []
    all_mse = []
    all_coverage = []
    all_auc = []

    for exp in range(n_experiments):
        print(f"\nGPIV Experiment {exp+1}/{n_experiments}")
        seed = seed_head + exp
        torch.manual_seed(seed)
        np.random.seed(seed)
        x, y, z, _ = generate_data(n_samples, f_type=f_type, seed=seed)

        lengthscale_x = median_heuristic(x).item()
        lengthscale_z = median_heuristic(z).item()
        gp = GPIV(lengthscale_x=lengthscale_x, lengthscale_z=lengthscale_z,
                  sigma=1.2, eta=eta, train_sigma=train_sigma, train_eta=False,
                  lx=train_lengthscale_x, lz=train_lengthscale_z)
        gp.set_data(x.reshape(-1,1), z.reshape(-1,1), y)
        gp.compute_kernel_matrices()
        gp.optimize_hyperparameters(n_iterations=n_iterations, lr=lr, verbose=False)

        x_test = torch.linspace(0, 1, n_test_points, dtype=torch.float64)
        if f_type == 'sine':
            h_x_true = 2 * torch.sin(2 * torch.pi * x_test)
        elif f_type == 'log':
            h_x_true = torch.log(torch.abs(16 * x_test - 8) + 1) * torch.sign(x_test - 0.5)
        elif f_type == 'linear':
            h_x_true = 4 * x_test - 2
        elif f_type == 'absolute':
            h_x_true = torch.abs(4 * x_test - 2) - 1

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
        n_per_step = max(1, n_test_points // 100)
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
        mse = torch.mean(se).item()
        coverage = gp.compute_coverage(x_test, h_x_true, confidence=0.95)

        all_accurate_ratios.append(accurate_ratios)
        all_removed_ratios.append(removed_ratios)
        all_mse.append(mse)
        all_coverage.append(coverage)
        all_auc.append(auc_arc)

        print(f"  MSE: {mse:.6f}, AUC: {auc_arc:.4f}, Coverage: {coverage*100:.2f}%")

    all_accurate_ratios = np.array(all_accurate_ratios)
    all_removed_ratios = np.array(all_removed_ratios)
    all_mse = np.array(all_mse)
    all_coverage = np.array(all_coverage)
    all_auc = np.array(all_auc)

    common_removed_ratios = all_removed_ratios[0]
    mean_accurate_ratios = np.mean(all_accurate_ratios, axis=0)
    std_accurate_ratios = np.std(all_accurate_ratios, axis=0)
    confidence_interval = 1.96 * std_accurate_ratios / np.sqrt(n_experiments)

    mean_mse = np.mean(all_mse)
    se_mse = np.std(all_mse) / np.sqrt(n_experiments)
    mean_coverage = np.mean(all_coverage)
    se_coverage = np.std(all_coverage) / np.sqrt(n_experiments)
    mean_auc = np.mean(all_auc)
    se_auc = np.std(all_auc) / np.sqrt(n_experiments)

    print(f"\n{'='*60}")
    print("GPIV MULTIPLE EXPERIMENTS SUMMARY")
    print(f"{'='*60}")
    print(f"Lengthscale optimization: x={train_lengthscale_x}, z={train_lengthscale_z}")
    print(f"MSE: Mean = {mean_mse:.6f}, Standard Error = {se_mse:.6f}")
    print(f"Coverage (95% CI): Mean = {mean_coverage*100:.2f}%, Standard Error = {se_coverage*100:.2f}%")
    print(f"AUC of ARC: Mean = {mean_auc:.4f}, Standard Error = {se_auc:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(common_removed_ratios, mean_accurate_ratios, 'b-', linewidth=3, label='Mean Accuracy')
    plt.fill_between(common_removed_ratios,
                     mean_accurate_ratios - confidence_interval,
                     mean_accurate_ratios + confidence_interval,
                     alpha=0.3, color='blue', label='95% CI')
    plt.xlabel('Proportion of High-Variance Data Removed')
    plt.ylabel(f'Proportion of Accurate Predictions (SE ≤ Q{quantile_threshold})')
    plt.title(f'GPIV Average ARC for {f_type} function\n({n_experiments} Experiments)')
    plt.grid(True, alpha=0.3)
    plt.ylim(quantile_threshold - 0.05, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return {
        'mean_mse': mean_mse, 'se_mse': se_mse,
        'mean_coverage': mean_coverage, 'se_coverage': se_coverage,
        'mean_auc': mean_auc, 'se_auc': se_auc
    }