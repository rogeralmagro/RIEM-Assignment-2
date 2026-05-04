from scenarios import generate_scenarios
from models import *
from plots import *

def run_step1():
    """
    Main function for Step 1.
    """

    # Generate all combined scenarios
    scenarios = generate_scenarios()

    # # Task 1.1: One-price balancing scheme
    q_one, expected_profit_one, profits_one = solve_one_price(scenarios)
    plot_profit_distribution(profits_one, "1.1 one price")

    # Task 1.2: Two-price
    q_two, expected_profit_two, profits_two, _ = solve_two_price(scenarios)
    plot_profit_distribution(profits_two, "1.2 two price")

    # Print comparison
    print("\n--- COMPARISON ---")
    print("One-price expected profit:", expected_profit_one)
    print("Two-price expected profit:", expected_profit_two)

if __name__ == "__main__":
    run_step1()