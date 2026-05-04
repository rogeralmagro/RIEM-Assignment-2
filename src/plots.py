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