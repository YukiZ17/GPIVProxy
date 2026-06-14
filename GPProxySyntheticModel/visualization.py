import matplotlib.pyplot as plt
import torch
import numpy as np
from data_generation import unstandardize_data

def visualize_optimization(self, history):
    if self.optimize_t or self.optimize_u or self.optimize_z or self.optimize_sigma:
        n_params = sum([self.optimize_t, self.optimize_u, self.optimize_z, self.optimize_sigma])
        fig, axes = plt.subplots(1, n_params + 1, figsize=(3*(n_params+1), 3))
        if n_params == 1:
            axes = [axes] if not isinstance(axes, np.ndarray) else axes
        axes[0].plot(history['losses'])
        axes[0].set_xlabel('iterations')
        axes[0].set_ylabel('-loglik')
        axes[0].set_title('Loss Curve')
        axes[0].grid(True, alpha=0.3)
        idx = 1
        if self.optimize_t:
            axes[idx].plot(history['lengthscales_t'])
            axes[idx].set_xlabel('iterations')
            axes[idx].set_ylabel('lengthscale_t')
            axes[idx].set_title('lengthscale_t optimization')
            axes[idx].grid(True, alpha=0.3)
            idx += 1
        if self.optimize_u:
            axes[idx].plot(history['lengthscales_u'])
            axes[idx].set_xlabel('iterations')
            axes[idx].set_ylabel('lengthscale_u')
            axes[idx].set_title('lengthscale_u optimization')
            axes[idx].grid(True, alpha=0.3)
            idx += 1
        if self.optimize_z:
            axes[idx].plot(history['lengthscales_z'])
            axes[idx].set_xlabel('iterations')
            axes[idx].set_ylabel('lengthscale_z')
            axes[idx].set_title('lengthscale_z optimization')
            axes[idx].grid(True, alpha=0.3)
            idx += 1
        if self.optimize_sigma:
            axes[idx].plot(history['sigmas'])
            axes[idx].set_xlabel('iterations')
            axes[idx].set_ylabel('sigma')
            axes[idx].set_title('sigma optimization')
            axes[idx].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def visualize_posterior(self, t_test_start=-2, t_test_end=4, n_points=400):
    t_test_orig = torch.linspace(t_test_start, t_test_end, n_points, dtype=torch.float64)
    h_true_orig = self.compute_true_structural_function(t_test_orig, n_mc=10000, standardized=False)
    mae, mse_orig, mse_std, predictions, variances, errors, coverage = self.evaluate_predictions(t_test_orig, h_true_orig, standardized=False)
    stds = torch.sqrt(variances)
    plt.figure(figsize=(12, 8))
    plt.plot(t_test_orig.numpy(), predictions.numpy(), linewidth=2, label='Proxy GP Posterior Mean', color='blue')
    plt.fill_between(t_test_orig.numpy(), (predictions - 2 * stds).numpy(), (predictions + 2 * stds).numpy(), alpha=0.2, color='blue', label='95% Confidence Interval')
    plt.plot(t_test_orig.numpy(), h_true_orig.numpy(), 'orange', linewidth=3, label='True E[Y|do(x)] (Structural Function)', linestyle='--')
    t_train_orig = unstandardize_data(self.t, self.t_mean, self.t_std)
    y_train_orig = unstandardize_data(self.y, self.y_mean, self.y_std)
    plt.scatter(t_train_orig.numpy(), y_train_orig.numpy(), alpha=0.5, label='Training Data', color='green', s=20)
    plt.xlabel('Treatment X', fontsize=12)
    plt.ylabel('Outcome Y', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(-2, 4)
    plt.show()
    print(f"Mean Squared Error (MSE) - original scale: {mse_orig:.4f}")
    print(f"Mean Squared Error (MSE) - standardized scale: {mse_std:.4f}")
    print(f"Coverage (95% CI): {coverage:.4f}")
    return predictions, variances, errors, mae, mse_orig, mse_std, coverage

def plot_arc_curve(arc_results):
    plt.figure(figsize=(8,6))
    plt.plot(arc_results['removed_counts'], arc_results['proportions_below_threshold'], marker='o', linewidth=2, color='blue')
    plt.xlabel('Number of High-Variance Points Removed')
    plt.ylabel('Proportion of Errors Below Threshold')
    plt.title(f'ARC Curve (AUC = {arc_results["final_auc"]:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()