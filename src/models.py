import numpy as np


def evaluate_one_price(q_DA, scenarios):
    profits = []

    for s in scenarios:
        wind = s["wind"]
        da_price = s["da_price"]
        balancing_price = s["balancing_price"]

        profit = np.sum(
            da_price * q_DA +
            balancing_price * (wind - q_DA)
        )

        profits.append(profit)

    return np.mean(profits), profits


def solve_one_price(scenarios):
    capacity = 500

    # medias por hora
    expected_da = np.mean([s["da_price"] for s in scenarios], axis=0)
    expected_bal = np.mean([s["balancing_price"] for s in scenarios], axis=0)

    # decisión óptima (muy importante)
    q_DA = np.where(expected_da >= expected_bal, capacity, 0)

    expected_profit, profits = evaluate_one_price(q_DA, scenarios)

    print("\n--- ONE PRICE ---")
    print("q_DA:", q_DA)
    print("Expected profit:", expected_profit)

    return q_DA, expected_profit, profits