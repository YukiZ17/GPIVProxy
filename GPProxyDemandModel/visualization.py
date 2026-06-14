import matplotlib.pyplot as plt
import numpy as np
import torch
from data_generation import unstandardize_data

def plot_optimization_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(history['losses'])
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Negative Log Marginal Likelihood')
    axes[0, 0].set_title('Loss History')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].plot(history['lengthscales_t'], label='lengthscale_t')
    axes[0, 1].plot(history['lengthscales_u'], label='lengthscale_u')
    axes[0, 1].plot(history['lengthscales_z'], label='lengthscale_z')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Lengthscale')
    axes[0, 1].set_title('Lengthscale Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[1, 0].plot(history['sigmas'])
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Sigma')
    axes[1, 0].set_title('Sigma Evolution')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 1].axis('off')
    plt.tight_layout()
    plt.show()

def plot_posterior(gp, t_test_orig, h_true_orig, t_train_orig, y_train_orig):
    predictions = []
    stds = []
    for t in t_test_orig:
        mean, var = gp.predict(t, standardized=False)
        predictions.append(mean.item())
        stds.append(torch.sqrt(var).item())
    predictions = torch.tensor(predictions, dtype=torch.float64)
    stds = torch.tensor(stds, dtype=torch.float64)
    plt.figure(figsize=(12, 8))
    plt.plot(t_test_orig.numpy(), predictions.numpy(), linewidth=2, label='Proxy GP Posterior Mean', color='blue')
    plt.fill_between(t_test_orig.numpy(), (predictions - 2 * stds).numpy(), (predictions + 2 * stds).numpy(),
                     alpha=0.2, color='blue', label='95% Confidence Interval')
    plt.plot(t_test_orig.numpy(), h_true_orig.numpy(), 'orange', linewidth=3,
             label='True E[Y|do(t)] (Structural Function)', linestyle='--')
    plt.scatter(t_train_orig.numpy(), y_train_orig.numpy(), alpha=0.5, label='Training Data', color='green', s=20)
    plt.xlabel('Price X', fontsize=12)
    plt.ylabel('Sales Y', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_arc_curve(arc_result, quantile):
    plt.figure(figsize=(8, 5))
    removed = arc_result['removed_counts']
    proportions = arc_result['proportions_below_threshold']
    plt.plot(removed, proportions, linewidth=2, label=f'Quantile {quantile}')
    plt.xlabel('Number of high-variance points removed')
    plt.ylabel('Proportion of errors below threshold')
    plt.title('ARC (Accuracy-Reduction Curve)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_multiple_arc_curve(removed_counts, proportions_mean, proportions_se, quantile):
    plt.figure(figsize=(8, 6))
    plt.plot(removed_counts, proportions_mean, linewidth=2, label='Mean ARC')
    plt.fill_between(removed_counts, proportions_mean - 1.96*proportions_se, proportions_mean + 1.96*proportions_se,
                     alpha=0.3, label='95% CI')
    plt.xlabel('Number of high-variance points removed')
    plt.ylabel('Proportion of errors below threshold')
    plt.title(f'ARC Curve (Quantile {quantile})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()