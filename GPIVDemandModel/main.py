from experiments import run_single_experiment, run_multiple_experiments
from visualization import visualize_single_experiment, visualize_multiple_experiments
import matplotlib
matplotlib.use('TkAgg')

def main():
    print("=" * 70)
    print("Gaussian Process Instrumental Variable (GPIV) Demand Design Experiment")
    print("=" * 70)
    MODE = 'multiple'
    N_REPEATS = 25
    N_TRAIN = 200
    BASE_SEED = 420
    QUANTILE = 0.75
    OPTIMIZE_CONFIG = {
        'optimize_lengthscale_p': True,
        'optimize_lengthscale_c': False,
        'optimize_lengthscale_t': True,
        'optimize_lengthscale_s': True,
        'optimize_amplitude_p': False,
        'optimize_amplitude_c': False,
        'optimize_amplitude_t': False,
        'optimize_amplitude_s': False,
        'optimize_sigma': True,
        'optimize_eta': False,
        'standardize_input': True
    }
    if MODE == 'single':
        print("Running SINGLE experiment mode...")
        print(f"Parameters: n={N_TRAIN}, quantile={QUANTILE}, seed={BASE_SEED}")
        gp, mse, arc_results, coverage, history = run_single_experiment(
            seed=BASE_SEED,
            n_train=N_TRAIN,
            quantile=QUANTILE,
            **OPTIMIZE_CONFIG,
            verbose=True
        )
        visualize_single_experiment(
            gp, mse, arc_results, coverage, history,
            quantile=QUANTILE,
            show_optimization=True,
            show_demand_curves=True,
            show_arc=True
        )
        print(f"\nFinal Results:")
        print(f"  MSE (standardized): {mse:.6f}")
        print(f"  Initial Coverage: {coverage*100:.2f}%")
        return {
            'mse': mse,
            'coverage': coverage,
            'arc_results': arc_results
        }
    elif MODE == 'multiple':
        print("Running MULTIPLE experiments mode...")
        print(f"Parameters: {N_REPEATS} experiments, n={N_TRAIN}, quantile={QUANTILE}")
        print(f"Base seed: {BASE_SEED}")
        all_results = run_multiple_experiments(
            n_repeats=N_REPEATS,
            n_train=N_TRAIN,
            quantile=QUANTILE,
            base_seed=BASE_SEED,
            **OPTIMIZE_CONFIG
        )
        visualize_multiple_experiments(all_results)
        print(f"\nReturning results for {N_REPEATS} experiments.")
        print(f"  all_mse: {len(all_results['all_mse'])} values")
        print(f"  all_coverage: {len(all_results['all_coverage'])} values")
        print(f"  all_arc_results: {len(all_results['all_arc_results'])} ARC results")
        return all_results
    else:
        print(f"Unknown mode: {MODE}")
        print("Available modes: 'single', 'multiple'")
    print("\nExperiment completed!")

if __name__ == "__main__":
    results = main()