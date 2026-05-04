from scenarios import generate_scenarios
from models import solve_one_price
from plots import *

def run_step1():
    """
    Main function for Step 1.
    """

    # Generate all combined scenarios
    scenarios = generate_scenarios()

    # Task 1.1: One-price balancing scheme
    q_one, expected_profit_one, profits_one = solve_one_price(scenarios)
    plot_profit_distribution(profits_one, "One-price profit distribution")


if __name__ == "__main__":
    run_step1()