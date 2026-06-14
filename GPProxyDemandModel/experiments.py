import torch
import numpy as np
from data_generation import generate_data, median_heuristic, unstandardize_data
from gp_model import ProxyGaussianProcess
from visualization import plot_optimization_history, plot_posterior, plot_arc_curve, plot_multiple_arc_curve

def set_seed(seed=442):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_single_experiment(config):
    set_seed(config['seed'])
    Z_std, U_std, T_std, Y_std, T_mean, T_std_val, Y_mean, Y_std_val = generate_data(
        n_samples=config['n_samples'], seed=config['seed']
    )
    if config['use_median_heuristic']:
        lengthscale_t = median_heuristic(T_std).item()
        lengthscale_u = median_heuristic(U_std).item()
        lengthscale_z = median_heuristic(Z_std).item()
    else:
        lengthscale_t = config.get('initial_lengthscale_t', 1.0)
        lengthscale_u = config.get('initial_lengthscale_u', 1.0)
        lengthscale_z = config.get('initial_lengthscale_z', 1.0)
    gp = ProxyGaussianProcess(
        lengthscale_t=lengthscale_t, lengthscale_u=lengthscale_u, lengthscale_z=lengthscale_z,
        sigma=config['initial_sigma'], eta=config['eta'],
        optimize_t=config['optimize_t'], optimize_u=config['optimize_u'],
        optimize_z=config['optimize_z'], optimize_sigma=config['optimize_sigma']
    )
    gp.set_data(z=Z_std, u=U_std, t=T_std, y=Y_std,
                t_mean=T_mean, t_std=T_std_val, y_mean=Y_mean, y_std=Y_std_val)
    gp.compute_kernel_matrices()
    history = gp.optimize_hyperparameters(n_iterations=config['n_iterations'], lr=config['lr'], verbose=True)
    plot_optimization_history(history)
    t_test_orig = torch.linspace(config['t_test_start'], config['t_test_end'], config['n_test_points'], dtype=torch.float64)
    h_true_orig = gp.compute_true_structural_function(t_test_orig, n_mc=10000, standardized=False)
    t_train_orig = unstandardize_data(gp.t, gp.t_mean, gp.t_std).squeeze()
    y_train_orig = unstandardize_data(gp.y, gp.y_mean, gp.y_std).squeeze()
    plot_posterior(gp, t_test_orig, h_true_orig, t_train_orig, y_train_orig)
    quantile = config.get('arc_quantile', 0.75)
    arc_result = gp.compute_arc_analysis(t_test_orig, h_true_orig, quantile=quantile, remove_step=10)
    print(f"\nSingle Experiment Results (Quantile = {quantile}):")
    print(f"  MSE (original scale): {arc_result['mse_orig']:.4f}")
    print(f"  MSE (standardized): {arc_result['mse_std']:.4f}")
    print(f"  Coverage: {arc_result['coverage']:.4f}")
    print(f"  AUC: {arc_result['final_auc']:.4f}")
    plot_arc_curve(arc_result, quantile)
    return arc_result

def run_multiple_experiments(config):
    n_experiments = config['n_experiments']
    base_seed = config['base_seed']
    quantile = config.get('arc_quantile', 0.75)
    mse_orig_list = []
    mse_std_list = []
    coverage_list = []
    auc_list = []
    arc_proportions_list = []
    arc_removed_counts = None
    for i in range(n_experiments):
        seed = base_seed + i
        print(f"Running experiment {i+1}/{n_experiments}, seed: {seed}...")
        try:
            exp_config = config.copy()
            exp_config['seed'] = seed
            set_seed(seed)
            Z_std, U_std, T_std, Y_std, T_mean, T_std_val, Y_mean, Y_std_val = generate_data(
                n_samples=exp_config['n_samples'], seed=seed
            )
            if exp_config['use_median_heuristic']:
                lengthscale_t = median_heuristic(T_std).item()
                lengthscale_u = median_heuristic(U_std).item()
                lengthscale_z = median_heuristic(Z_std).item()
            else:
                lengthscale_t = exp_config.get('initial_lengthscale_t', 1.0)
                lengthscale_u = exp_config.get('initial_lengthscale_u', 1.0)
                lengthscale_z = exp_config.get('initial_lengthscale_z', 1.0)
            gp = ProxyGaussianProcess(
                lengthscale_t=lengthscale_t, lengthscale_u=lengthscale_u, lengthscale_z=lengthscale_z,
                sigma=exp_config['initial_sigma'], eta=exp_config['eta'],
                optimize_t=exp_config['optimize_t'], optimize_u=exp_config['optimize_u'],
                optimize_z=exp_config['optimize_z'], optimize_sigma=exp_config['optimize_sigma']
            )
            gp.set_data(z=Z_std, u=U_std, t=T_std, y=Y_std,
                        t_mean=T_mean, t_std=T_std_val, y_mean=Y_mean, y_std=Y_std_val)
            gp.compute_kernel_matrices()
            gp.optimize_hyperparameters(n_iterations=exp_config['n_iterations'], lr=exp_config['lr'], verbose=False)
            t_test_orig = torch.linspace(exp_config['t_test_start'], exp_config['t_test_end'],
                                         exp_config['n_test_points'], dtype=torch.float64)
            h_true_orig = gp.compute_true_structural_function(t_test_orig, n_mc=10000, standardized=False)
            arc_result = gp.compute_arc_analysis(t_test_orig, h_true_orig, quantile=quantile, remove_step=10)
            mse_orig_list.append(arc_result['mse_orig'])
            mse_std_list.append(arc_result['mse_std'])
            coverage_list.append(arc_result['coverage'])
            auc_list.append(arc_result['final_auc'])
            arc_proportions_list.append(arc_result['proportions_below_threshold'])
            if arc_removed_counts is None:
                arc_removed_counts = arc_result['removed_counts']
            print(f"  MSE_orig={arc_result['mse_orig']:.4f}, Coverage={arc_result['coverage']:.4f}, AUC={arc_result['final_auc']:.4f}")
        except Exception as e:
            print(f"  Experiment {i+1} failed: {e}")
    if len(mse_orig_list) == 0:
        print("No successful experiments.")
        return None
    mse_orig_mean = np.mean(mse_orig_list)
    mse_orig_se = np.std(mse_orig_list) / np.sqrt(len(mse_orig_list))
    mse_std_mean = np.mean(mse_std_list)
    mse_std_se = np.std(mse_std_list) / np.sqrt(len(mse_std_list))
    coverage_mean = np.mean(coverage_list)
    coverage_se = np.std(coverage_list) / np.sqrt(len(coverage_list))
    auc_mean = np.mean(auc_list)
    auc_se = np.std(auc_list) / np.sqrt(len(auc_list))
    min_len = min(len(p) for p in arc_proportions_list)
    proportions_array = np.array([p[:min_len] for p in arc_proportions_list])
    proportions_mean = np.mean(proportions_array, axis=0)
    proportions_se = np.std(proportions_array, axis=0) / np.sqrt(len(proportions_array))
    removed_counts = arc_removed_counts[:min_len]
    print("\n" + "="*80)
    print("Multiple Experiments Results")
    print("="*80)
    print(f"Number of successful experiments: {len(mse_orig_list)}")
    print(f"Quantile used for ARC: {quantile}")
    print(f"MSE (original scale): mean = {mse_orig_mean:.4f}, SE = {mse_orig_se:.4f}")
    print(f"MSE (standardized): mean = {mse_std_mean:.4f}, SE = {mse_std_se:.4f}")
    print(f"Coverage: mean = {coverage_mean:.4f}, SE = {coverage_se:.4f}")
    print(f"AUC: mean = {auc_mean:.4f}, SE = {auc_se:.4f}")
    print("="*80)
    plot_multiple_arc_curve(removed_counts, proportions_mean, proportions_se, quantile)
    return {
        'mse_orig_mean': mse_orig_mean, 'mse_orig_se': mse_orig_se,
        'mse_std_mean': mse_std_mean, 'mse_std_se': mse_std_se,
        'coverage_mean': coverage_mean, 'coverage_se': coverage_se,
        'auc_mean': auc_mean, 'auc_se': auc_se,
        'arc_removed_counts': removed_counts,
        'arc_proportions_mean': proportions_mean,
        'arc_proportions_se': proportions_se
    }