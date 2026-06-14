import matplotlib.pyplot as plt
import numpy as np
import torch
from data_generation import h_function

def visualize_optimization(history):
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

def visualize_demand_curve_p(gp, t_fixed=5.0, s_fixed=4, standardize_input=True):
    p_range = torch.linspace(2.5, 27.5, 100, dtype=torch.float64)
    true_demand = []
    pred_demand = []
    var_demand = []
    for p in p_range:
        h_t = h_function(torch.tensor(t_fixed))
        true_f = 100 + (10 + p) * s_fixed * h_t - 2 * p
        if gp.scalers is not None and 'Y' in gp.scalers:
            true_f_scaled = (true_f - gp.scalers['Y']['mean']) / gp.scalers['Y']['std']
        else:
            true_f_scaled = true_f
        true_demand.append(true_f_scaled.item())
        pred_f, var_f = gp.predict(p, torch.tensor(t_fixed, dtype=torch.float64),
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
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_demand_curve_t(gp, p_fixed=25.0, s_fixed=4, standardize_input=True):
    t_range = torch.linspace(0, 10, 100, dtype=torch.float64)
    true_demand = []
    pred_demand = []
    var_demand = []
    for t in t_range:
        h_t = h_function(torch.tensor(t))
        true_f = 100 + (10 + p_fixed) * s_fixed * h_t - 2 * p_fixed
        if gp.scalers is not None and 'Y' in gp.scalers:
            true_f_scaled = (true_f - gp.scalers['Y']['mean']) / gp.scalers['Y']['std']
        else:
            true_f_scaled = true_f
        true_demand.append(true_f_scaled.item())
        pred_f, var_f = gp.predict(torch.tensor(p_fixed, dtype=torch.float64),
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
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_single_experiment(gp, test_mse, arc_results, initial_coverage, history,
                               quantile=0.75, show_optimization=True, show_demand_curves=True,
                               show_arc=True):
    if show_optimization and len(history['losses']) > 0:
        print("\nVisualizing optimization history...")
        visualize_optimization(history)
    if show_demand_curves:
        print("\nVisualizing demand curves...")
        visualize_demand_curve_p(gp, t_fixed=2.0, s_fixed=4)
        visualize_demand_curve_t(gp, p_fixed=20.0, s_fixed=4)
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

