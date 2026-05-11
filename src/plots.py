import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def plot_profit_distribution(profits, filename):
    # Crear carpeta si no existe
    os.makedirs("results", exist_ok=True)

    plt.figure()
    plt.hist(profits, bins=50)
    # plt.title(filename)
    plt.xlabel("Profit")
    plt.ylabel("Frequency")

    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)

    # Guardar figura
    plt.savefig(f"results/{filename}.png")

    plt.close()

def plot_q_DA(q_one, q_two):
    import matplotlib.pyplot as plt
    import os

    os.makedirs("results", exist_ok=True)

    hours = range(1, 25)

    plt.figure()

    plt.plot(hours, q_one, marker='o', label="One-price")
    plt.plot(hours, q_two, marker='s', label="Two-price")

    plt.xlabel("Hour")
    plt.ylabel("q_DA (MW)")
    # plt.title("Optimal Day-Ahead Offers")

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.savefig("results/q_DA_comparison(1&2).png")
    plt.close()

def plot_q_DA2(q_one, q_two):
    os.makedirs("results", exist_ok=True)

    hours = range(1, 25)

    plt.figure()

    # Step plots
    plt.step(hours, q_one, where='mid', label="One-price")
    plt.step(hours, q_two, where='mid', label="Two-price")

    plt.xlabel("Hour")
    plt.ylabel("q_DA (MW)")
    # plt.title("Optimal Day-Ahead Offers")

    plt.xticks(range(1, 25))
    plt.ylim(0, 520)

    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.savefig("results/q_DA_step.png")
    plt.close()

def plot_cv_by_fold(results, run_name):
    os.makedirs("results", exist_ok=True)

    folds = [r["fold"] for r in results]

    plt.figure()

    plt.plot(folds, [r["one_in"] for r in results], marker="o", label="One-price in-sample")
    plt.plot(folds, [r["one_out"] for r in results], marker="o", label="One-price out-of-sample")
    plt.plot(folds, [r["two_in"] for r in results], marker="s", label="Two-price in-sample")
    plt.plot(folds, [r["two_out"] for r in results], marker="s", label="Two-price out-of-sample")

    plt.xlabel("Fold")
    plt.ylabel("Expected profit")
    # plt.title(f"Cross-validation results ({run_name})")
    plt.xticks(folds)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/{run_name}_cv_by_fold.png")
    plt.close()

def plot_cv_gaps(results, run_name):
    os.makedirs("results", exist_ok=True)

    folds = [r["fold"] for r in results]

    plt.figure()
    plt.axhline(0, color="black", linewidth=1)

    plt.plot(folds, [r["one_gap"] for r in results], marker="o", label="One-price")
    plt.plot(folds, [r["two_gap"] for r in results], marker="s", label="Two-price")

    plt.xlabel("Fold")
    plt.ylabel("In-sample profit - Out-of-sample profit")
    # plt.title(f"Generalization gap ({run_name})")
    plt.xticks(folds)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"results/{run_name}_cv_gap.png")
    plt.close()

def summarize_cv_variability(all_results):
    os.makedirs("results", exist_ok=True)

    rows = []

    for run_name, results in all_results.items():

        one_out = np.array([r["one_out"] for r in results])
        two_out = np.array([r["two_out"] for r in results])
        one_gap = np.array([r["one_gap"] for r in results])
        two_gap = np.array([r["two_gap"] for r in results])

        rows.append({
            "run_name": run_name,
            "fold_size": results[0]["fold_size"],
            "n_folds": results[0]["n_folds"],

            "one_out_mean": np.mean(one_out),
            "one_out_std": np.std(one_out),
            "one_gap_mean": np.mean(one_gap),
            "one_gap_std": np.std(one_gap),

            "two_out_mean": np.mean(two_out),
            "two_out_std": np.std(two_out),
            "two_gap_mean": np.mean(two_gap),
            "two_gap_std": np.std(two_gap),
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/task_1_3_variability_summary.csv", index=False)

def plot_task_1_4_report_figures(risk_results, scenario_sensitivity_df):
    os.makedirs("results", exist_ok=True)

    summary_rows = []
    detailed_rows = []
    for r in risk_results:
        q = np.asarray(r["q_DA"], dtype=float)
        summary_rows.append({
            "scheme": r["scheme"],
            "beta": r["beta"],
            "expected_profit": r["expected_profit"],
            "cvar": r["cvar"],
            "profit_std": r["profit_std"],
            "min_profit": r["min_profit"],
            "p10_profit": r["p10_profit"],
            "q_mean": float(np.mean(q)),
            "q_std": float(np.std(q)),
        })
        detailed_rows.append({
            "scheme": r["scheme"],
            "beta": r["beta"],
            "expected_profit": r["expected_profit"],
            "cvar": r["cvar"],
            "profit_std": r["profit_std"],
            "min_profit": r["min_profit"],
            "p10_profit": r["p10_profit"],
            "q_DA": q,
            "profits": np.asarray(r["profits"], dtype=float),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(["scheme", "beta"])
    summary_df.to_csv("results/task_1_4_summary_table.csv", index=False)
    risk_df = pd.DataFrame(detailed_rows).sort_values(["scheme", "beta"]).reset_index(drop=True)

    def get_row(df, scheme, beta):
        mask = (df["scheme"] == scheme) & np.isclose(df["beta"], beta)
        matches = df.loc[mask]
        if matches.empty:
            return None
        return matches.iloc[0]

    two_price = risk_df[risk_df["scheme"] == "two-price"].sort_values("beta").reset_index(drop=True)
    hours = np.arange(1, 25)

    # A) Two-price frontier
    plt.figure(figsize=(8, 5))
    plt.plot(two_price["cvar"], two_price["expected_profit"], marker="o")
    for _, row in two_price.iterrows():
        plt.annotate(
            f'{row["beta"]:.2g}',
            (row["cvar"], row["expected_profit"]),
            xytext=(5, 5),
            textcoords="offset points",
        )
    plt.xlabel("CVaR of profit, worst 10% scenarios")
    plt.ylabel("Expected profit")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/task_1_4_frontier_two_price.png", dpi=300)
    plt.close()

    # B) Relative tradeoff versus beta for two-price
    two_tradeoff = two_price.copy()
    base = two_tradeoff.iloc[0]
    two_tradeoff["expected_profit_rel_pct"] = 100.0 * (
        two_tradeoff["expected_profit"] - base["expected_profit"]
    ) / base["expected_profit"]
    two_tradeoff["cvar_rel_pct"] = 100.0 * (
        two_tradeoff["cvar"] - base["cvar"]
    ) / base["cvar"]
    two_tradeoff["profit_std_rel_pct"] = 100.0 * (
        two_tradeoff["profit_std"] - base["profit_std"]
    ) / base["profit_std"]

    plt.figure(figsize=(8, 5))
    plt.plot(two_tradeoff["beta"], two_tradeoff["expected_profit_rel_pct"], marker="o", label="Expected profit")
    plt.plot(two_tradeoff["beta"], two_tradeoff["cvar_rel_pct"], marker="s", label="CVaR")
    plt.plot(two_tradeoff["beta"], two_tradeoff["profit_std_rel_pct"], marker="^", label="Profit std")
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Beta")
    plt.ylabel("Change relative to beta = 0 [%]")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/task_1_4_relative_tradeoff_two_price.png", dpi=300)
    plt.close()

    # C) Selected two-price offers
    selected_betas = [0.0, 0.5, 0.9, 1.0]
    selected_rows = []
    for beta in selected_betas:
        row = get_row(risk_df, "two-price", beta)
        if row is not None:
            selected_rows.append(row)

    plt.figure(figsize=(8, 5))
    for row in selected_rows:
        plt.step(
            hours,
            row["q_DA"],
            where="mid",
            label=f'beta={row["beta"]:.2g}'
        )
    plt.xlabel("Hour")
    plt.ylabel("q_DA (MW)")
    plt.xticks(hours)
    plt.ylim(0, 520)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/task_1_4_offers_selected_betas.png", dpi=300)
    plt.close()

    # D) Offer change from beta 0 to beta 1 for two-price
    row_beta0 = get_row(risk_df, "two-price", 0.0)
    row_beta1 = get_row(risk_df, "two-price", 1.0)
    if row_beta0 is not None and row_beta1 is not None:
        q_change = row_beta1["q_DA"] - row_beta0["q_DA"]
        plt.figure(figsize=(8, 5))
        plt.bar(hours, q_change, width=0.8)
        plt.axhline(0, color="black", linewidth=1)
        plt.xlabel("Hour")
        plt.ylabel("Change in q_DA from beta=0 to beta=1 (MW)")
        plt.xticks(hours)
        plt.grid(True, axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/task_1_4_offer_change_beta1.png", dpi=300)
        plt.close()

    # E) Lower-tail profit distribution for beta 0 vs beta 1 in two-price
    if row_beta0 is not None and row_beta1 is not None:
        selected = [row_beta0, row_beta1]
        combined_profits = np.concatenate([row["profits"] for row in selected])
        lower_tail_limit = float(np.quantile(combined_profits, 0.25))

        plt.figure(figsize=(8, 5))
        for row in selected:
            profits = row["profits"]
            sorted_profits = np.sort(profits)
            tail_count = max(1, int(np.ceil(0.10 * len(sorted_profits))))
            var_profit = float(sorted_profits[tail_count - 1])
            cvar_profit = float(np.mean(sorted_profits[:tail_count]))

            plt.hist(
                profits,
                bins=28,
                range=(float(np.min(combined_profits)), lower_tail_limit),
                alpha=0.42,
                label=f'beta={row["beta"]:.0f}'
            )
            plt.axvline(cvar_profit, linestyle="--", linewidth=2)
            plt.axvline(var_profit, linestyle=":", linewidth=2)

        plt.xlim(float(np.min(combined_profits)), lower_tail_limit)
        plt.xlabel("Profit")
        plt.ylabel("Frequency")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("results/task_1_4_profit_distribution_tail_two_price.png", dpi=300)
        plt.close()

    # F/G) Scenario sensitivity for two-price, beta 0 and 1 only
    sensitivity = scenario_sensitivity_df.copy()
    sensitivity = sensitivity[
        (sensitivity["scheme"] == "two-price")
        & (np.isclose(sensitivity["beta"], 0.0) | np.isclose(sensitivity["beta"], 1.0))
    ]

    grouped = (
        sensitivity.groupby(["n_scenarios", "beta"], as_index=False)
        .agg(
            expected_profit_mean=("expected_profit", "mean"),
            expected_profit_std=("expected_profit", "std"),
            cvar_mean=("cvar", "mean"),
            cvar_std=("cvar", "std"),
        )
        .sort_values(["beta", "n_scenarios"])
    )

    grouped["expected_profit_std"] = grouped["expected_profit_std"].fillna(0.0)
    grouped["cvar_std"] = grouped["cvar_std"].fillna(0.0)

    plt.figure(figsize=(8, 5))
    for beta, marker in [(0.0, "o"), (1.0, "s")]:
        data = grouped[np.isclose(grouped["beta"], beta)]
        if data.empty:
            continue
        plt.errorbar(
            data["n_scenarios"],
            data["cvar_mean"],
            yerr=data["cvar_std"],
            marker=marker,
            capsize=4,
            label=f"Two-price, beta={beta:.0f}"
        )
    plt.xlabel("Number of in-sample scenarios")
    plt.ylabel("CVaR")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/task_1_4_scenario_sensitivity_cvar.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    for beta, marker in [(0.0, "o"), (1.0, "s")]:
        data = grouped[np.isclose(grouped["beta"], beta)]
        if data.empty:
            continue
        plt.errorbar(
            data["n_scenarios"],
            data["expected_profit_mean"],
            yerr=data["expected_profit_std"],
            marker=marker,
            capsize=4,
            label=f"Two-price, beta={beta:.0f}"
        )
    plt.xlabel("Number of in-sample scenarios")
    plt.ylabel("Expected profit")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/task_1_4_scenario_sensitivity_profit.png", dpi=300)
    plt.close()
