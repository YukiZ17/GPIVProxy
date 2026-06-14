import torch
import numpy as np
from data_generation import generate_data, median_heuristic
from proxy_gp_model import ProxyGaussianProcess
from visualization import visualize_optimization, visualize_posterior, plot_arc_curve

def set_seed(seed=442):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_single_experiment(config):
    set_seed(config['seed'])
    Z_std, U_std, T_std, Y_std, T_mean, T_std_val, Y_mean, Y_std_val = generate_data(n_samples=config['n_samples'], seed=config['seed'])
    if config['use_median_heuristic']:
        lengthscale_t = median_heuristic(T_std).item()
        lengthscale_u = median_heuristic(U_std).item()
        lengthscale_z = median_heuristic(Z_std).item()
    else:
        lengthscale_t = config.get('initial_lengthscale_t', 1.0)
        lengthscale_u = config.get('initial_lengthscale_u', 1.0)
        lengthscale_z = config.get('initial_lengthscale_z', 1.0)
    gp = ProxyGaussianProcess(lengthscale_t=lengthscale_t, lengthscale_u=lengthscale_u, lengthscale_z=lengthscale_z, sigma=config['initial_sigma'], eta=config['eta'], optimize_t=config['optimize_t'], optimize_u=config['optimize_u'], optimize_z=config['optimize_z'], optimize_sigma=config['optimize_sigma'])
    gp.set_data(z=Z_std, u=U_std, t=T_std, y=Y_std, t_mean=T_mean, t_std=T_std_val, y_mean=Y_mean, y_std=Y_std_val)
    gp.compute_kernel_matrices()
    history = gp.optimize_hyperparameters(n_iterations=config['n_iterations'], lr=config['lr'], verbose=config['verbose'])
    if config['show_plots']:
        visualize_optimization(gp, history)
    t_test_orig = torch.linspace(config['t_test_start'], config['t_test_end'], config['n_test_points'], dtype=torch.float64)
    h_true_orig = gp.compute_true_structural_function(t_test_orig, n_mc=10000,standardized=False)
    mae, mse_orig, mse_std, predictions, variances, errors, coverage = gp.evaluate_predictions(t_test_orig, h_true_orig, standardized=False)
    if config['show_plots']:
        visualize_posterior(gp, t_test_start=config['t_test_start'], t_test_end=config['t_test_end'], n_points=config['n_test_points'])
    arc_results = None
    if config['run_arc']:
        arc_results = gp.compute_arc_analysis(t_test_start=config['t_test_start'], t_test_end=config['t_test_end'], n_points=config['n_test_points'], quantile_p=config['quantile_p'], remove_step=config['remove_step'])
        if config['show_plots']:
            plot_arc_curve(arc_results)
        print(f"\nARC Analysis Results:")
        print(f"  Quantile threshold (p={config['quantile_p']}): {arc_results['quantile_threshold']:.4f}")
        print(f"  AUC (Area under ARC curve): {arc_results['final_auc']:.4f}")
    print(f"\nSingle Experiment Summary:")
    print(f"  MSE (original scale): {mse_orig:.4f}")
    print(f"  Coverage (95% CI): {coverage:.4f}")
    return gp, {'mae': mae, 'mse_orig': mse_orig, 'mse_std': mse_std, 'coverage': coverage, 'arc_results': arc_results}

def run_multiple_experiments(config):
    n_experiments = config['n_experiments']
    base_seed = config['base_seed']
    quantile_p = config['quantile_p']
    remove_step = config['remove_step']
    all_mae = []
    all_mse_orig = []
    all_mse_std = []
    all_coverage = []
    all_final_auc = []
    all_proportions = []
    all_removed_counts = None
    for i in range(n_experiments):
        seed = base_seed + i
        print(f"Running experiment {i+1}/{n_experiments}, seed={seed}...")
        try:
            exp_config = config.copy()
            exp_config['seed'] = seed
            exp_config['verbose'] = False
            exp_config['show_plots'] = False
            gp, results = run_single_experiment(exp_config)
            all_mae.append(results['mae'])
            all_mse_orig.append(results['mse_orig'])
            all_mse_std.append(results['mse_std'])
            all_coverage.append(results['coverage'])
            if results['arc_results'] is not None:
                arc = results['arc_results']
                all_final_auc.append(arc['final_auc'])
                if all_removed_counts is None:
                    all_removed_counts = arc['removed_counts']
                all_proportions.append(arc['proportions_below_threshold'])
            print(f"  MSE_orig={results['mse_orig']:.4f}, Coverage={results['coverage']:.4f}, AUC={results['arc_results']['final_auc']:.4f}" if results['arc_results'] else f"  MSE_orig={results['mse_orig']:.4f}, Coverage={results['coverage']:.4f}")
        except Exception as e:
            print(f"  Experiment {i+1} failed: {e}")
    if len(all_mse_orig) == 0:
        print("No successful experiments.")
        return
    mse_mean = np.mean(all_mse_orig)
    mse_se = np.std(all_mse_orig) / np.sqrt(len(all_mse_orig))
    coverage_mean = np.mean(all_coverage)
    coverage_se = np.std(all_coverage) / np.sqrt(len(all_coverage))
    auc_mean = np.mean(all_final_auc) if all_final_auc else float('nan')
    auc_se = np.std(all_final_auc) / np.sqrt(len(all_final_auc)) if all_final_auc else float('nan')
    print("\n" + "="*70)
    print("MULTIPLE EXPERIMENTS AGGREGATED RESULTS")
    print("="*70)
    print(f"Number of successful experiments: {len(all_mse_orig)}")
    print(f"MSE (original scale): Mean = {mse_mean:.4f}, Standard Error = {mse_se:.4f}")
    print(f"Coverage (95% CI): Mean = {coverage_mean:.4f}, Standard Error = {coverage_se:.4f}")
    print(f"AUC (ARC curve): Mean = {auc_mean:.4f}, Standard Error = {auc_se:.4f}")
    if all_proportions:
        min_len = min(len(p) for p in all_proportions)
        proportions_array = np.array([p[:min_len] for p in all_proportions])
        proportions_mean = np.mean(proportions_array, axis=0)
        proportions_std = np.std(proportions_array, axis=0)
        proportions_se = proportions_std / np.sqrt(len(all_proportions))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.plot(all_removed_counts[:min_len], proportions_mean, color='blue', linewidth=2, label='Mean proportion')
        plt.fill_between(all_removed_counts[:min_len], proportions_mean - 1.96*proportions_se, proportions_mean + 1.96*proportions_se, alpha=0.2, color='blue', label='95% CI')
        plt.axhline(y=quantile_p, color='red', linestyle='--', label=f'Quantile threshold (p={quantile_p})')
        plt.xlabel('Number of High-Variance Points Removed')
        plt.ylabel('Proportion of Errors Below Threshold')
        plt.title(f'ARC Curve (Mean AUC = {auc_mean:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    return {'mse_mean': mse_mean, 'mse_se': mse_se, 'coverage_mean': coverage_mean, 'coverage_se': coverage_se, 'auc_mean': auc_mean, 'auc_se': auc_se}