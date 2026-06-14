from experiments import run_single_experiment, run_multiple_experiments
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    mode = 'single'

    if mode == 'single':
        config = {
            'seed': 42,
            'n_samples': 200,
            'n_iterations': 250,
            'lr': 0.25,
            'optimize_t': True,
            'optimize_u': True,
            'optimize_z': False,
            'optimize_sigma': True,
            'initial_sigma': 0.5,
            'eta': 0.1,
            'use_median_heuristic': True,
            't_test_start': 10,
            't_test_end': 40,
            'n_test_points': 300,
            'arc_quantile': 0.75
        }
        print("\nRunning single experiment with default configuration.")
        run_single_experiment(config)

    elif mode == 'multiple':
        config = {
            'n_experiments': 25,
            'base_seed': 42,
            'n_samples': 200,
            'n_iterations': 250,
            'lr': 0.25,
            'optimize_t': True,
            'optimize_u': True,
            'optimize_z': False,
            'optimize_sigma': True,
            'initial_sigma': 0.5,
            'eta': 0.1,
            'use_median_heuristic': True,
            't_test_start': 10,
            't_test_end': 40,
            'n_test_points': 300,
            'arc_quantile': 0.75
        }
        print("\nRunning multiple experiments with default configuration.")
        run_multiple_experiments(config)
    else:
        print("Invalid mode. Please set mode to 'single' or 'multiple'.")