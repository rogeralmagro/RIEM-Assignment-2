import numpy as np
import gurobipy as gp
from gurobipy import GRB

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

def evaluate_two_price(q_DA, scenarios):
    """
    Calculates the profit of a fixed day-ahead offer q_DAn under the two-price balancing scheme.
    """
    profits = []

    for s in scenarios:
        wind = s["wind"]
        da = s["da_price"]
        si = s["imbalance"]   # 1 = deficit, 0 = surplus

        bp = np.where(si == 1, 1.25 * da, 0.85 * da)

        delta = wind - q_DA

        settlement_price = np.where(
            ((si == 1) & (delta >= 0)) | ((si == 0) & (delta <= 0)),
            da, # beneficial imbalance
            bp  # harmful imbalance
            )

        profit = np.sum(
            da * q_DA + settlement_price * delta
        )

        profits.append(profit)

    return np.mean(profits), profits


def solve_two_price(scenarios):
    N = len(scenarios)
    T = range(24)
    S = range(N)
    capacity = 500
    prob = 1 / N
    # M = capacity

    model = gp.Model("two_price_wind")

    # Variables
    p_DA = model.addVars(T, lb=0, ub=capacity, name="p_DA")
    delta_pos = model.addVars(S, T, lb=0, name="delta_pos")
    delta_neg = model.addVars(S, T, lb=0, name="delta_neg")

    # wind - p_DA = delta_pos - delta_neg
    for s in S:
        for t in T:
            wind = scenarios[s]["wind"][t]
            model.addConstr(
                wind - p_DA[t] == delta_pos[s, t] - delta_neg[s, t],
                name=f"imbalance_{s}_{t}"
            )

    # Objective
    obj = gp.LinExpr()

    for s in S:
        for t in T:
            da = scenarios[s]["da_price"][t]
            si = scenarios[s]["imbalance"][t]
            if si == 1:
                bp = 1.25 * da
                obj += prob * (
                    da * p_DA[t]
                    + da * delta_pos[s, t]
                    - bp * delta_neg[s, t]
                )
            else:
                bp = 0.85 * da
                obj += prob * (
                    da * p_DA[t]
                    + bp * delta_pos[s, t]
                    - da * delta_neg[s, t]
                )

    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()

    q_DA = np.array([p_DA[t].X for t in T])
    expected_profit = model.ObjVal
    _, profits = evaluate_two_price(q_DA, scenarios)

    print("\n--- TWO PRICE / GUROBIPY ---")
    print("q_DA:", q_DA)
    print("Expected profit:", expected_profit)

    return q_DA, expected_profit, profits, model

def cross_validation(scenarios, fold_size=200, n_folds=8, seed=42, run_name="cv_8fold_200"):

    scenarios_cv = scenarios.copy()

    rng = np.random.default_rng(seed)
    rng.shuffle(scenarios_cv)

    required_scenarios = fold_size * n_folds

    if required_scenarios > len(scenarios_cv):
        raise ValueError(
            f"fold_size * n_folds = {required_scenarios}, "
            f"but only {len(scenarios_cv)} scenarios are available."
        )

    scenarios_cv = scenarios_cv[:required_scenarios]

    results = []

    for i in range(n_folds):

        in_sample = scenarios_cv[i*fold_size:(i+1)*fold_size]
        out_sample = scenarios_cv[:i*fold_size] + scenarios_cv[(i+1)*fold_size:]

        # --- ONE PRICE ---
        q_one, in_profit_one, _ = solve_one_price(in_sample)
        out_profit_one, _ = evaluate_one_price(q_one, out_sample)

        # --- TWO PRICE ---
        q_two, in_profit_two, _, _ = solve_two_price(in_sample)
        out_profit_two, _ = evaluate_two_price(q_two, out_sample)

        results.append({
            "run_name": run_name,
            "fold": i + 1,
            "fold_size": fold_size,
            "n_folds": n_folds,

            "one_in": in_profit_one,
            "one_out": out_profit_one,
            "one_gap": in_profit_one - out_profit_one,

            "two_in": in_profit_two,
            "two_out": out_profit_two,
            "two_gap": in_profit_two - out_profit_two,

            "q_one": q_one,
            "q_two": q_two
        })

    return results