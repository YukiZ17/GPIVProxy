from experiments import run_single_experiment, run_multiple_experiments
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    mode = 'single'   # change to 'multiple' for multiple experiments

    if mode == 'single':
        single_config = {
            'seed': 47,
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
            't_test_start': -2,
            't_test_end': 4,
            'n_test_points': 300,
            'run_arc': True,
            'quantile_p': 0.75,
            'remove_step': 10,
            'verbose': True,
            'show_plots': True,
        }
        print("\n" + "="*70)
        print("RUNNING SINGLE EXPERIMENT")
        print("="*70)
        run_single_experiment(single_config)

    elif mode == 'multiple':
        multiple_config = {
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
            't_test_start': -2,
            't_test_end': 4,
            'n_test_points': 300,
            'run_arc': True,
            'quantile_p': 0.75,
            'remove_step': 10,
        }
        print("\n" + "="*70)
        print("RUNNING MULTIPLE EXPERIMENTS")
        print("="*70)
        run_multiple_experiments(multiple_config)
    else:
        print("Invalid mode. Choose 'single' or 'multiple'.")