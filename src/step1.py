from scenarios import generate_scenarios
from models import *
from plots import *

def run_step1():
    """
    Main function for Step 1.
    """

    # Generate all combined scenarios
    scenarios = generate_scenarios()

    # Task 1.1: Offering Strategy Under a One-Price Balancing Scheme
    q_one, expected_profit_one, profits_one = solve_one_price(scenarios)
    plot_profit_distribution(profits_one, "1.1 one price")

    # Task 1.2: Offering Strategy Under a Two-Price Balancing Scheme
    q_two, expected_profit_two, profits_two, _ = solve_two_price(scenarios)
    plot_profit_distribution(profits_two, "1.2 two price")

    # hourly offers comparison
    plot_q_DA(q_one, q_two)
    plot_q_DA2(q_one, q_two)

    # Print comparison
    print("\n--- COMPARISON ---")
    print("One-price expected profit:", expected_profit_one)
    print("Two-price expected profit:", expected_profit_two)

    # Task 1.3: Ex-post Analysis
    configs = [
        (100, 16),
        (200, 8),
        (400, 4),
        (800, 2)
    ]

    all_results = {}

    for fold_size, n_folds in configs:

        run_name = f"task_1_3_{n_folds}fold_{fold_size}scen"

        results = cross_validation(
            scenarios,
            fold_size=fold_size,
            n_folds=n_folds,
            seed=42,
            run_name=run_name
        )

        all_results[run_name] = results

        plot_cv_by_fold(results, run_name)
        plot_cv_gaps(results, run_name)

    # Summary table for variability
    summarize_cv_variability(all_results)

if __name__ == "__main__":
    run_step1()