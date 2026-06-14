import time
from single_experiment import run_single_gpiv_demo
from multiple_experiments import run_gpiv_multiple_experiments
import matplotlib
matplotlib.use('TkAgg')

if __name__ == "__main__":
    FUNCTION_TYPE = 'linear'
    EXPERIMENT_MODE = 'multiple'

    if EXPERIMENT_MODE == 'single':
        start_time = time.time()
        run_single_gpiv_demo(
            f_type=FUNCTION_TYPE, n_samples=200, seed=1,
            train_lengthscale_x=False, train_lengthscale_z=False,
            train_sigma=True, eta=0.1, n_iterations=25, lr=0.05,
            n_test_points=200, quantile_threshold=0.75)
        print(f"Execution time: {time.time()-start_time:.2f} seconds")
    elif EXPERIMENT_MODE == 'multiple':
        results = run_gpiv_multiple_experiments(
            n_experiments=25, f_type=FUNCTION_TYPE, n_samples=500,
            n_test_points=300, quantile_threshold=0.75,
            n_iterations=25, lr=0.05,
            train_lengthscale_x=True, train_lengthscale_z=False,
            train_sigma=True, eta=0.1)
    else:
        print(f"Unknown experiment mode: {EXPERIMENT_MODE}")
        print("Please choose from: 'single', 'multiple'")

    print("\nExperiment completed!")