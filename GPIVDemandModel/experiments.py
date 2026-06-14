import torch
import numpy as np
from data_generation import generate_demand_data, standardize_data, median_heuristic
from gpiv_model import GPIV

def run_single_experiment(seed=59, n_train=500, quantile=0.75,
                         optimize_lengthscale_p=True, optimize_lengthscale_c=True,
                         optimize_lengthscale_t=True, optimize_lengthscale_s=True,
                         optimize_amplitude_p=False, optimize_amplitude_c=False,
                         optimize_amplitude_t=False, optimize_amplitude_s=False,
                         optimize_sigma=True, optimize_eta=False,
                         standardize_input=True, verbose=True):
    if verbose:
        print(f"Running single experiment with seed {seed}...")
    Y_train, P_train, T_train, S_train, C_train = generate_demand_data(
        n_samples=n_train, rho=0.9, seed=seed
    )
    data_dict = {'Y': Y_train, 'P': P_train, 'T': T_train, 'S': S_train, 'C': C_train}
    scaled_data, scalers = standardize_data(data_dict)
    Y_scaled = scaled_data['Y']
    P_scaled = scaled_data['P']
    T_scaled = scaled_data['T']
    S_scaled = scaled_data['S']
    C_scaled = scaled_data['C']
    lengthscale_p = median_heuristic(P_scaled).item()
    lengthscale_c = median_heuristic(C_scaled).item()
    lengthscale_t = median_heuristic(T_scaled).item()
    lengthscale_s = median_heuristic(S_scaled).item()
    gp = GPIV(
        lengthscale_p=lengthscale_p, lengthscale_c=lengthscale_c,
        lengthscale_t=lengthscale_t, lengthscale_s=lengthscale_s,
        amplitude_p=1.0, amplitude_c=1.0, amplitude_t=1.0, amplitude_s=1.0,
        sigma=0.5, eta=0.1,
        optimize_lengthscale_p=optimize_lengthscale_p,
        optimize_lengthscale_c=optimize_lengthscale_c,
        optimize_lengthscale_t=optimize_lengthscale_t,
        optimize_lengthscale_s=optimize_lengthscale_s,
        optimize_amplitude_p=optimize_amplitude_p,
        optimize_amplitude_c=optimize_amplitude_c,
        optimize_amplitude_t=optimize_amplitude_t,
        optimize_amplitude_s=optimize_amplitude_s,
        optimize_sigma=optimize_sigma,
        optimize_eta=optimize_eta
    )
    gp.set_data(P_scaled, C_scaled, T_scaled, S_scaled, Y_scaled, scalers)
    gp.compute_kernel_matrices()
    any_optimize = (optimize_lengthscale_p or optimize_lengthscale_c or
                   optimize_lengthscale_t or optimize_lengthscale_s or
                   optimize_amplitude_p or optimize_amplitude_c or
                   optimize_amplitude_t or optimize_amplitude_s or
                   optimize_sigma or optimize_eta)
    if any_optimize:
        if verbose:
            print("Optimizing hyperparameters...")
        history = gp.optimize_hyperparameters(n_iterations=150, lr=0.1, verbose=verbose)
    else:
        history = {'losses': [], 'parameters': {}}
    test_mse = gp.compute_mse(standardize_input=standardize_input)
    if verbose:
        print("Computing ARC and coverage...")
    rejection_rates, accuracies, delta = gp.compute_accuracy_rejection_curve(
        delta=None, quantile=quantile, standardize_input=standardize_input
    )
    arc_results = {
        'rejection_rates': rejection_rates,
        'accuracies': accuracies,
        'delta': delta
    }
    initial_coverage = gp.compute_coverage_rate(
        confidence_level=0.95, standardize_input=standardize_input
    )
    return gp, test_mse, arc_results, initial_coverage, history

def run_multiple_experiments(n_repeats=25, n_train=500, quantile=0.75, base_seed=42,
                            optimize_lengthscale_p=True, optimize_lengthscale_c=True,
                            optimize_lengthscale_t=True, optimize_lengthscale_s=True,
                            optimize_amplitude_p=False, optimize_amplitude_c=False,
                            optimize_amplitude_t=False, optimize_amplitude_s=False,
                            optimize_sigma=True, optimize_eta=False,
                            standardize_input=True):
    print(f"Running {n_repeats} experiments...")
    all_mse = []
    all_coverage = []
    all_arc_results = []
    for i in range(n_repeats):
        seed = base_seed + i
        if (i + 1) % 5 == 0:
            print(f"  Running experiment {i + 1}/{n_repeats}...")
        _, test_mse, arc_results, initial_coverage, _ = run_single_experiment(
            seed=seed,
            n_train=n_train,
            quantile=quantile,
            optimize_lengthscale_p=optimize_lengthscale_p,
            optimize_lengthscale_c=optimize_lengthscale_c,
            optimize_lengthscale_t=optimize_lengthscale_t,
            optimize_lengthscale_s=optimize_lengthscale_s,
            optimize_amplitude_p=optimize_amplitude_p,
            optimize_amplitude_c=optimize_amplitude_c,
            optimize_amplitude_t=optimize_amplitude_t,
            optimize_amplitude_s=optimize_amplitude_s,
            optimize_sigma=optimize_sigma,
            optimize_eta=optimize_eta,
            standardize_input=standardize_input,
            verbose=False
        )
        all_mse.append(test_mse)
        all_coverage.append(initial_coverage)
        all_arc_results.append(arc_results)
    all_mse = np.array(all_mse)
    all_coverage = np.array(all_coverage)
    all_accuracies = np.array([arc['accuracies'] for arc in all_arc_results])
    all_deltas = np.array([arc['delta'] for arc in all_arc_results])
    rejection_rates = all_arc_results[0]['rejection_rates'] if all_arc_results else None
    summary_stats = {
        'mse': {
            'mean': np.mean(all_mse),
            'std': np.std(all_mse),
            'median': np.median(all_mse),
            'min': np.min(all_mse),
            'max': np.max(all_mse)
        },
        'coverage': {
            'mean': np.mean(all_coverage),
            'std': np.std(all_coverage),
            'median': np.median(all_coverage),
            'min': np.min(all_coverage),
            'max': np.max(all_coverage)
        },
        'arc': {
            'mean_accuracies': np.mean(all_accuracies, axis=0),
            'std_accuracies': np.std(all_accuracies, axis=0),
            'mean_delta': np.mean(all_deltas),
            'std_delta': np.std(all_deltas)
        },
        'rejection_rates': rejection_rates
    }
    all_results = {
        'all_mse': all_mse,
        'all_coverage': all_coverage,
        'all_arc_results': all_arc_results,
        'summary_stats': summary_stats,
        'n_repeats': n_repeats,
        'n_train': n_train,
        'quantile': quantile,
        'base_seed': base_seed
    }
    return all_results