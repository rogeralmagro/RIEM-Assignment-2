import os

import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

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


def solve_one_price(scenarios, verbose=True):
    """Solve stochastic offering problem under one-price scheme."""
    
    capacity = 500

    # Expected prices per hour (averaged over all scenarios)
    expected_da = np.mean([s["da_price"] for s in scenarios], axis=0)
    expected_bal = np.mean([s["balancing_price"] for s in scenarios], axis=0)

    # Optimal decision: full capacity or zero (all-or-nothing)
    q_DA = np.where(expected_da >= expected_bal, capacity, 0)

    # Evaluate resulting strategy
    expected_profit, profits = evaluate_one_price(q_DA, scenarios)

    if verbose:
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


def solve_two_price(scenarios, verbose=True):
    N = len(scenarios)
    T = range(24)
    S = range(N)
    capacity = 500
    prob = 1 / N
    # M = capacity

    model = gp.Model("two_price_wind")
    model.Params.OutputFlag = 0

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

    if verbose:
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
        q_one, in_profit_one, _ = solve_one_price(in_sample, verbose=False)
        out_profit_one, _ = evaluate_one_price(q_one, out_sample)

        # --- TWO PRICE ---
        q_two, in_profit_two, _, _ = solve_two_price(in_sample, verbose=False)
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

def compute_cvar_profit(profits, alpha=0.90):
    """
    Compute CVaR of profit at confidence level alpha.

    Since we maximize profit, CVaR is interpreted as the average profit
    in the worst (1-alpha) tail of the profit distribution.
    """
    profits = np.asarray(profits, dtype=float)
    sorted_profits = np.sort(profits)

    tail_count = int(np.ceil((1.0 - alpha) * len(sorted_profits)))
    tail_count = max(1, tail_count)

    return float(np.mean(sorted_profits[:tail_count]))


def solve_one_price_risk_averse(scenarios, beta=0.0, alpha=0.90):
    """
    Risk-averse one-price stochastic offering model using CVaR.
    This is solved as an LP using the standard CVaR linearization.
    """
    N = len(scenarios)
    T = range(24)
    S = range(N)
    capacity = 500.0
    prob = 1.0 / N

    model = gp.Model("one_price_risk_averse")
    model.Params.OutputFlag = 0

    # Decision variables
    p_DA = model.addVars(T, lb=0, ub=capacity, name="p_DA")

    # Scenario profit variables
    profit = model.addVars(S, lb=-GRB.INFINITY, name="profit")

    # CVaR variables
    eta = model.addVar(lb=-GRB.INFINITY, name="eta")
    z = model.addVars(S, lb=0, name="z")

    # Profit definition for each scenario
    for s in S:
        expr = gp.LinExpr()

        for t in T:
            wind = scenarios[s]["wind"][t]
            da = scenarios[s]["da_price"][t]
            bp = scenarios[s]["balancing_price"][t]

            expr += da * p_DA[t] + bp * (wind - p_DA[t])

        model.addConstr(profit[s] == expr, name=f"profit_def_{s}")

    # CVaR lower-tail linearization
    for s in S:
        model.addConstr(z[s] >= eta - profit[s], name=f"cvar_tail_{s}")

    expected_profit = prob * gp.quicksum(profit[s] for s in S)
    cvar_profit = eta - (1.0 / (1.0 - alpha)) * prob * gp.quicksum(z[s] for s in S)

    model.setObjective((1.0 - beta) * expected_profit + beta * cvar_profit,GRB.MAXIMIZE)
    model.optimize()

    q_DA = np.array([p_DA[t].X for t in T])
    expected_profit_eval, profits = evaluate_one_price(q_DA, scenarios)
    cvar_eval = compute_cvar_profit(profits, alpha=alpha)
    q_std = float(np.std(q_DA))

    return {
        "scheme": "one-price",
        "beta": beta,
        "alpha": alpha,
        "q_DA": q_DA,
        "q_mean": float(np.mean(q_DA)),
        "q_std": q_std,
        "q_min": float(np.min(q_DA)),
        "q_max": float(np.max(q_DA)),
        "expected_profit": expected_profit_eval,
        "cvar": cvar_eval,
        "profits": profits,
        "profit_std": float(np.std(profits)),
        "min_profit": float(np.min(profits)),
        "p10_profit": float(np.quantile(profits, 0.10)),
        "objective_value": model.ObjVal,
        "solve_time": model.Runtime,
        "n_variables": model.NumVars,
        "n_constraints": model.NumConstrs,
    }


def solve_two_price_risk_averse(scenarios, beta=0.0, alpha=0.90):
    """
    Risk-averse two-price stochastic offering model using CVaR.
    """
    N = len(scenarios)
    T = range(24)
    S = range(N)
    capacity = 500.0
    prob = 1.0 / N

    model = gp.Model("two_price_risk_averse")
    model.Params.OutputFlag = 0

    # Decision variables
    p_DA = model.addVars(T, lb=0, ub=capacity, name="p_DA")
    delta_pos = model.addVars(S, T, lb=0, name="delta_pos")
    delta_neg = model.addVars(S, T, lb=0, name="delta_neg")

    # Scenario profit variables
    profit = model.addVars(S, lb=-GRB.INFINITY, name="profit")

    # CVaR variables
    eta = model.addVar(lb=-GRB.INFINITY, name="eta")
    z = model.addVars(S, lb=0, name="z")

    # Imbalance definition
    for s in S:
        for t in T:
            wind = scenarios[s]["wind"][t]
            model.addConstr(
                wind - p_DA[t] == delta_pos[s, t] - delta_neg[s, t],
                name=f"imbalance_{s}_{t}"
            )

    # Profit definition for each scenario
    for s in S:
        expr = gp.LinExpr()

        for t in T:
            da = scenarios[s]["da_price"][t]
            si = scenarios[s]["imbalance"][t]

            if si == 1:
                # Deficit system:
                # positive wind deviation is beneficial and settled at DA,
                # negative wind deviation is harmful and settled at BP.
                bp = 1.25 * da
                expr += da * p_DA[t] + da * delta_pos[s, t] - bp * delta_neg[s, t]
            else:
                # Surplus system:
                # positive wind deviation is harmful and settled at BP,
                # negative wind deviation is beneficial and settled at DA.
                bp = 0.85 * da
                expr += da * p_DA[t] + bp * delta_pos[s, t] - da * delta_neg[s, t]

        model.addConstr(profit[s] == expr, name=f"profit_def_{s}")

    # CVaR lower-tail linearization
    for s in S:
        model.addConstr(z[s] >= eta - profit[s], name=f"cvar_tail_{s}")

    expected_profit = prob * gp.quicksum(profit[s] for s in S)
    cvar_profit = eta - (1.0 / (1.0 - alpha)) * prob * gp.quicksum(z[s] for s in S)

    model.setObjective((1.0 - beta) * expected_profit + beta * cvar_profit,GRB.MAXIMIZE)
    model.optimize()

    q_DA = np.array([p_DA[t].X for t in T])
    expected_profit_eval, profits = evaluate_two_price(q_DA, scenarios)
    cvar_eval = compute_cvar_profit(profits, alpha=alpha)
    q_std = float(np.std(q_DA))

    return {
        "scheme": "two-price",
        "beta": beta,
        "alpha": alpha,
        "q_DA": q_DA,
        "q_mean": float(np.mean(q_DA)),
        "q_std": q_std,
        "q_min": float(np.min(q_DA)),
        "q_max": float(np.max(q_DA)),
        "expected_profit": expected_profit_eval,
        "cvar": cvar_eval,
        "profits": profits,
        "profit_std": float(np.std(profits)),
        "min_profit": float(np.min(profits)),
        "p10_profit": float(np.quantile(profits, 0.10)),
        "objective_value": model.ObjVal,
        "solve_time": model.Runtime,
        "n_variables": model.NumVars,
        "n_constraints": model.NumConstrs,
    }


def run_risk_averse_sweep(scenarios, betas=None, alpha=0.90, verbose=True):
    """
    Run Task 1.4 for both one-price and two-price schemes.
    """
    if betas is None:
        betas = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]

    results = []

    for beta in betas:
        if verbose:
            print(f"\n--- Risk-averse run: beta = {beta} ---")

        one_result = solve_one_price_risk_averse(
            scenarios=scenarios,
            beta=beta,
            alpha=alpha
        )
        results.append(one_result)

        two_result = solve_two_price_risk_averse(
            scenarios=scenarios,
            beta=beta,
            alpha=alpha
        )
        results.append(two_result)

        if verbose:
            print(
                f"One-price: E[profit]={one_result['expected_profit']:.2f}, "
                f"CVaR={one_result['cvar']:.2f}"
            )
            print(
                f"Two-price: E[profit]={two_result['expected_profit']:.2f}, "
                f"CVaR={two_result['cvar']:.2f}"
            )
            print(f"q_DA one-price: {one_result['q_DA']}")
            print(f"q_DA two-price: {two_result['q_DA']}")

    return results

def test_risk_averse_scenario_sensitivity(
    scenarios,
    sample_sizes=None,
    n_repeats=5,
    seed=42
):
    os.makedirs("results", exist_ok=True)
    if sample_sizes is None:
        sample_sizes = [50, 100, 200, 400, 800, 1600]

    rng = np.random.default_rng(seed)
    rows = []

    for n in sample_sizes:
        for repeat in range(n_repeats):
            idx = rng.choice(len(scenarios), size=n, replace=False)
            subset = [scenarios[i] for i in idx]

            results = run_risk_averse_sweep(
                scenarios=subset,
                betas=[0, 1.0],
                alpha=0.90,
                verbose=False
            )

            for r in results:
                rows.append({
                    "n_scenarios": n,
                    "repeat": repeat + 1,
                    "scheme": r["scheme"],
                    "beta": r["beta"],
                    "expected_profit": r["expected_profit"],
                    "cvar": r["cvar"],
                    "profit_std": r["profit_std"],
                    "min_profit": r["min_profit"],
                    "p10_profit": r["p10_profit"],
                    "q_mean": r["q_mean"],
                    "q_std": r["q_std"],
                    "q_min": r["q_min"],
                    "q_max": r["q_max"],
                })

    df = pd.DataFrame(rows)
    df.to_csv("results/task_1_4_scenario_sensitivity.csv", index=False)
    return df
