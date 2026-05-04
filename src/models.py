import numpy as np


def evaluate_one_price(q_DA, scenarios):
    """Evaluate profit of a given day-ahead strategy over all scenarios."""
    
    profits = []

    # Compute profit for each scenario
    for s in scenarios:
        wind = s["wind"]
        da_price = s["da_price"]
        balancing_price = s["balancing_price"]

        # Profit = DA revenue + balancing settlement
        profit = np.sum(
            da_price * q_DA +
            balancing_price * (wind - q_DA)
        )

        profits.append(profit)

    # Return expected profit and full distribution
    return np.mean(profits), profits


def solve_one_price(scenarios):
    """Solve stochastic offering problem under one-price scheme."""
    
    capacity = 500

    # Expected prices per hour (averaged over all scenarios)
    expected_da = np.mean([s["da_price"] for s in scenarios], axis=0)
    expected_bal = np.mean([s["balancing_price"] for s in scenarios], axis=0)

    # Optimal decision: full capacity or zero (all-or-nothing)
    q_DA = np.where(expected_da >= expected_bal, capacity, 0)

    # Evaluate resulting strategy
    expected_profit, profits = evaluate_one_price(q_DA, scenarios)

    print("\n--- ONE PRICE ---")
    print("q_DA:", q_DA)
    print("Expected profit:", expected_profit)

    return q_DA, expected_profit, profits