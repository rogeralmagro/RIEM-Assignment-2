"""
Step 2: Participation in ancillary service markets.

This module focuses on Task 2.1 of Assignment 2:
- Generate stochastic flexible load profiles.
- Determine the optimal DK2 FCR-D UP reserve bid.
- Solve the in-sample P90 requirement using ALSO-X and CVaR.
- Save numerical results and figures for the report.

"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Global settings for Step 2
# ============================================================

# Number of load profiles required by the assignment.
N_PROFILES = 300

# The first 100 profiles are used for in-sample optimization.
N_IN_SAMPLE = 100

# The remaining 200 profiles are used for out-of-sample verification.
N_OUT_OF_SAMPLE = 200

# One bidding hour is represented with minute-level resolution.
N_MINUTES = 60

# Load profile requirements from the assignment statement.
MIN_LOAD_KW = 220.0
MAX_LOAD_KW = 600.0
MAX_RAMP_KW = 35.0

# P90 reliability requirement.
RELIABILITY_P90 = 0.90

# Fixed seed to make the generated profiles reproducible.
RANDOM_SEED = 42

# File and folder paths.
DATA_PATH = Path("data/load_profiles_step2.csv")
RESULTS_DIR = Path("results")


# ============================================================
# 1. Load profile generation
# ============================================================

def generate_load_profiles(
    n_profiles=N_PROFILES,
    n_minutes=N_MINUTES,
    min_load_kw=MIN_LOAD_KW,
    max_load_kw=MAX_LOAD_KW,
    max_ramp_kw=MAX_RAMP_KW,
    seed=RANDOM_SEED,
):
    """
    Generate stochastic flexible load profiles.

    Each profile represents one bidding hour with minute-level resolution.

    The generated profiles satisfy the assignment requirements:
    - consumption remains between 220 and 600 kW,
    - minute-to-minute changes do not exceed 35 kW,
    - 300 profiles are generated in total.

    The profiles are generated using a mean-reverting random walk. This
    avoids having too many profiles stuck at the lower bound of 220 kW,
    while still creating realistic stochastic variation.
    """

    # Create a random number generator with a fixed seed.
    # This makes the generated load profiles reproducible.
    rng = np.random.default_rng(seed)

    # Store one row per profile and minute.
    records = []

    # Loop over all stochastic load profiles.
    for profile in range(1, n_profiles + 1):

        # Assign profile to in-sample or out-of-sample set.
        profile_set = "in_sample" if profile <= N_IN_SAMPLE else "out_of_sample"

        # Select a profile-specific average load level.
        # Keeping this away from the lower bound avoids trivial P90 results.
        target_load = rng.uniform(300.0, 560.0)

        # Select the initial load close to the target level.
        initial_low = max(min_load_kw, target_load - 70.0)
        initial_high = min(max_load_kw, target_load + 70.0)
        current_load = rng.uniform(initial_low, initial_high)

        # Add a mild trend and a small sinusoidal component to create
        # different profile shapes across the hour.
        trend = rng.uniform(-0.6, 0.6)
        phase = rng.uniform(0.0, 2.0 * np.pi)

        # Loop over all minutes of the bidding hour.
        for minute in range(1, n_minutes + 1):

            if minute > 1:

                # Normalized time within the hour.
                hour_position = (minute - 1) / (n_minutes - 1)

                # Smooth deterministic variation within the hour.
                smooth_variation = 12.0 * np.sin(2.0 * np.pi * hour_position + phase)

                # Minute-specific target load.
                target_this_minute = (
                    target_load
                    + trend * (minute - n_minutes / 2.0)
                    + smooth_variation
                )

                # Mean reversion pulls the load gently towards the target.
                mean_reversion = 0.12 * (target_this_minute - current_load)

                # Random noise creates stochastic behaviour.
                noise = rng.normal(0.0, 10.0)

                # Candidate ramp for this minute.
                ramp = mean_reversion + noise

                # Enforce the assignment ramping constraint.
                ramp = np.clip(ramp, -max_ramp_kw, max_ramp_kw)

                # Update the current load.
                current_load += ramp

                # Enforce the load range required by the assignment.
                current_load = np.clip(current_load, min_load_kw, max_load_kw)

            # Store the current minute of the current profile.
            records.append({
                "profile": profile,
                "minute": minute,
                "load_kW": round(float(current_load), 4),
                "set": profile_set,
            })

    # Convert records into a DataFrame.
    return pd.DataFrame(records)


def save_load_profiles(load_profiles, output_path=DATA_PATH):
    """
    Save the generated load profiles to a CSV file.
    """

    # Make sure that the data folder exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save profiles without the pandas index.
    load_profiles.to_csv(output_path, index=False)


def load_or_generate_profiles(output_path=DATA_PATH):
    """
    Load existing Step 2 profiles if available.

    If the file does not exist, the function generates new profiles and
    saves them. This avoids changing the generated data every time the
    workflow is executed.
    """

    # Check whether the load profile file already exists.
    if output_path.exists():

        # If it exists, load the same profiles again.
        print(f"Loading existing load profiles from {output_path}")
        return pd.read_csv(output_path)

    # If the file does not exist, generate the profiles from scratch.
    print("Generating new Step 2 load profiles...")
    load_profiles = generate_load_profiles()

    # Save the generated profiles so future runs use the same data.
    save_load_profiles(load_profiles, output_path)
    print(f"Saved load profiles to {output_path}")

    return load_profiles


# ============================================================
# 2. Availability calculation
# ============================================================

def compute_profile_availability(load_profiles, min_operating_load_kw=0.0):
    """
    Compute the hourly upward reserve availability of each load profile.

    For FCR-D UP, a flexible load provides upward reserve by reducing its
    consumption. Since the reserve bid is submitted for the whole hour, the
    bid must be available at every minute of that hour.

    Therefore, for each profile s, the available reserve is calculated as:

        A_s = min_m load_{m,s} - minimum operating load

    In this assignment, the flexible load can consume between 0 and 600 kW.
    Since no positive minimum operating consumption is imposed, the minimum
    operating load is set to 0 kW.

    Returns
    -------
    pandas.DataFrame
        Columns:
        profile, set, availability_kW
    """

    # For each profile, find the minimum load over the 60 minutes.
    # This is the maximum reserve bid that can be sustained for the full hour.
    availability = (
        load_profiles
        .groupby(["profile", "set"], as_index=False)["load_kW"]
        .min()
        .rename(columns={"load_kW": "availability_kW"})
    )

    # Subtract the minimum operating load.
    # Here this is zero, but keeping it as an argument makes the function flexible.
    availability["availability_kW"] -= min_operating_load_kw

    # Ensure no negative reserve availability appears.
    availability["availability_kW"] = availability["availability_kW"].clip(lower=0.0)

    return availability


# ============================================================
# 3. ALSO-X / empirical P90 solution
# ============================================================

def solve_alsox_p90(availability_values, reliability=RELIABILITY_P90):
    """
    Solve the empirical P90 reserve bidding problem.

    The P90 requirement is interpreted as:

        P(A_s >= r) >= 0.90

    where:
    - A_s is the available reserve in profile s,
    - r is the hourly FCR-D UP reserve bid.

    With 100 equally likely in-sample profiles and reliability 0.90,
    at most 10 profiles may have A_s < r.

    In this one-dimensional case, the ALSO-X / empirical chance-constrained
    solution can be obtained directly from the sorted availability values.

    Returns
    -------
    dict
        Reserve bid and diagnostic information.
    """

    # Start timer to report computational time.
    start_time = time.perf_counter()

    # Convert input to a NumPy array.
    availability_values = np.asarray(availability_values, dtype=float)

    # Number of in-sample profiles.
    n_samples = len(availability_values)

    # Calculate the maximum number of allowed violations.
    # For 100 profiles and P90, this gives 10.
    max_violations = int(np.floor((1.0 - reliability) * n_samples + 1e-9))

    # Sort availability values from lowest to highest.
    sorted_availability = np.sort(availability_values)

    # The largest feasible bid is the first value after the allowed violations.
    # Example: if 10 violations are allowed, we choose the 11th lowest value.
    bid_index = min(max_violations, n_samples - 1)
    reserve_bid = sorted_availability[bid_index]

    # Identify which profiles violate the selected reserve bid.
    violations = availability_values < reserve_bid

    # Count violations.
    n_violations = int(np.sum(violations))

    # Compute the empirical reliability achieved by the selected bid.
    achieved_reliability = 1.0 - n_violations / n_samples

    # Calculate shortfalls for all in-sample profiles.
    # Shortfall is positive only when the bid exceeds available reserve.
    shortfalls = np.maximum(0.0, reserve_bid - availability_values)

    # Stop timer.
    solve_time = time.perf_counter() - start_time

    # Return all relevant results in a dictionary.
    return {
        "method": "ALSO-X",
        "reserve_bid_kW": round(float(reserve_bid), 4),
        "reliability_target": reliability,
        "achieved_reliability": round(float(achieved_reliability), 4),
        "n_profiles": n_samples,
        "n_violations": n_violations,
        "max_allowed_violations": max_violations,
        "expected_shortfall_kW": round(float(np.mean(shortfalls)), 4),
        "max_shortfall_kW": round(float(np.max(shortfalls)), 4),
        "solve_time_s": round(float(solve_time), 6),

        # Equivalent compact MIP representation:
        # one continuous reserve bid variable + one binary violation variable per profile.
        "n_variables_equivalent_model": 1 + n_samples,

        # Equivalent constraints:
        # one feasibility constraint per profile, one violation-count constraint,
        # and simple bid bounds.
        "n_constraints_equivalent_model": n_samples + 3,
    }


# ============================================================
# 4. CVaR solution
# ============================================================

def solve_cvar_p90(availability_values, reliability=RELIABILITY_P90):
    """
    Solve the CVaR-based reserve bidding problem.

    The reserve shortfall loss is defined as:

        loss_s = r - A_s

    where:
    - r is the reserve bid,
    - A_s is the available reserve in profile s.

    The CVaR approximation controls the worst tail of this loss
    distribution. For P90, the relevant tail is the worst 10% of profiles.

    In this simple scalar case, the CVaR-feasible bid is obtained as the
    average availability in the lower 10% tail. This is usually more
    conservative than the empirical P90 / ALSO-X bid.

    Returns
    -------
    dict
        Reserve bid and diagnostic information.
    """

    # Start timer to report computational time.
    start_time = time.perf_counter()

    # Convert input to a NumPy array.
    availability_values = np.asarray(availability_values, dtype=float)

    # Number of in-sample profiles.
    n_samples = len(availability_values)

    # Compute the number of profiles included in the lower tail.
    # For P90 and 100 profiles, the tail contains 10 profiles.
    if reliability >= 1.0:
        tail_count = 1
    else:
        tail_count = int(np.ceil((1.0 - reliability) * n_samples))

    # Ensure that the tail contains at least one profile and at most all profiles.
    tail_count = max(1, min(tail_count, n_samples))

    # Sort availability from lowest to highest.
    sorted_availability = np.sort(availability_values)

    # Compute the CVaR-based reserve bid.
    # This is the mean availability of the worst lower-tail profiles.
    reserve_bid = np.mean(sorted_availability[:tail_count])

    # Identify which profiles violate the selected bid.
    violations = availability_values < reserve_bid

    # Count violations.
    n_violations = int(np.sum(violations))

    # Compute empirical reliability.
    achieved_reliability = 1.0 - n_violations / n_samples

    # Compute shortfalls for all profiles.
    shortfalls = np.maximum(0.0, reserve_bid - availability_values)

    # Stop timer.
    solve_time = time.perf_counter() - start_time

    # Return all relevant results in a dictionary.
    return {
        "method": "CVaR",
        "reserve_bid_kW": round(float(reserve_bid), 4),
        "reliability_target": reliability,
        "achieved_reliability": round(float(achieved_reliability), 4),
        "n_profiles": n_samples,
        "n_violations": n_violations,
        "tail_count": tail_count,
        "expected_shortfall_kW": round(float(np.mean(shortfalls)), 4),
        "max_shortfall_kW": round(float(np.max(shortfalls)), 4),
        "solve_time_s": round(float(solve_time), 6),

        # Equivalent linear CVaR model:
        # variables: reserve bid r, VaR-related variable tau,
        # and one auxiliary variable per profile.
        "n_variables_equivalent_model": 2 + n_samples,

        # Equivalent constraints:
        # one CVaR constraint, one shortfall constraint per profile,
        # non-negativity constraints for auxiliary variables, and bid bounds.
        "n_constraints_equivalent_model": 2 * n_samples + 3,
    }


# ============================================================
# 5. Plotting for Task 2.1
# ============================================================

def plot_in_sample_profiles(load_profiles, results, output_path):
    """
    Plot the 100 in-sample load profiles and the reserve bids.

    This figure shows the minute-level load trajectories used for the
    in-sample optimization. The horizontal dashed lines represent the
    reserve bids obtained with ALSO-X and CVaR.
    """

    # Select only the 100 in-sample profiles.
    in_sample = load_profiles[load_profiles["set"] == "in_sample"]

    # Create a new figure.
    plt.figure(figsize=(9, 5))

    # Plot each in-sample load profile as a thin line.
    for _, group in in_sample.groupby("profile"):
        plt.plot(
            group["minute"],
            group["load_kW"],
            linewidth=0.8,
            alpha=0.35,
        )

    # Add one horizontal line for each reserve bid.
    for result in results:
        plt.axhline(
            result["reserve_bid_kW"],
            linestyle="--",
            linewidth=2,
            label=f'{result["method"]} bid = {result["reserve_bid_kW"]:.2f} kW',
        )

    # Add labels, title and legend.
    plt.xlabel("Minute")
    plt.ylabel("Load / available upward reserve [kW]")
    plt.title("In-sample flexible load profiles and optimal reserve bids")
    plt.legend()

    # Improve layout before saving.
    plt.tight_layout()

    # Make sure the results folder exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the figure.
    plt.savefig(output_path, dpi=300)

    # Close the figure to avoid memory issues.
    plt.close()


def plot_availability_distribution(availability, results, output_path):
    """
    Plot the distribution of in-sample hourly reserve availability.

    The hourly availability of each profile is computed as the minimum load
    over the bidding hour. The vertical dashed lines show the reserve bids
    obtained with ALSO-X and CVaR.
    """

    # Select availability values for the 100 in-sample profiles.
    in_sample = availability[availability["set"] == "in_sample"]

    # Create a new figure.
    plt.figure(figsize=(8, 5))

    # Plot histogram of hourly reserve availability.
    plt.hist(
        in_sample["availability_kW"],
        bins=15,
        edgecolor="black",
        alpha=0.75,
    )

    # Add vertical lines for the reserve bids.
    for result in results:
        plt.axvline(
            result["reserve_bid_kW"],
            linestyle="--",
            linewidth=2,
            label=f'{result["method"]} bid = {result["reserve_bid_kW"]:.2f} kW',
        )

    # Add labels, title and legend.
    plt.xlabel("Hourly reserve availability [kW]")
    plt.ylabel("Number of profiles")
    plt.title("Distribution of in-sample reserve availability")
    plt.legend()

    # Improve layout before saving.
    plt.tight_layout()

    # Make sure the results folder exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the figure.
    plt.savefig(output_path, dpi=300)

    # Close the figure to avoid memory issues.
    plt.close()


# ============================================================
# 6. Task 2.1 workflow
# ============================================================

def run_task_2_1():
    """
    Run Task 2.1 completely.

    This function executes the full Task 2.1 workflow:
    1. Load or generate the 300 load profiles.
    2. Compute the hourly reserve availability of each profile.
    3. Select the 100 in-sample profiles.
    4. Solve the P90 reserve bid with ALSO-X.
    5. Solve the P90 reserve bid with CVaR.
    6. Save tables and figures for the report.
    """

    # Make sure the results folder exists.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load or generate the 300 stochastic load profiles.
    load_profiles = load_or_generate_profiles()

    # Compute hourly reserve availability A_s for each profile.
    # This is the minimum load over the 60 minutes of each profile.
    availability = compute_profile_availability(load_profiles)

    # Select only the 100 in-sample profiles for Task 2.1.
    in_sample_availability = availability.loc[
        availability["set"] == "in_sample",
        "availability_kW",
    ].to_numpy()

    # Solve the empirical P90 / ALSO-X reserve bidding problem.
    alsox_result = solve_alsox_p90(
        in_sample_availability,
        reliability=RELIABILITY_P90,
    )

    # Solve the CVaR-based reserve bidding problem.
    cvar_result = solve_cvar_p90(
        in_sample_availability,
        reliability=RELIABILITY_P90,
    )

    # Store both results in a list for tables and plots.
    results = [alsox_result, cvar_result]

    # Convert result dictionaries to a DataFrame.
    results_df = pd.DataFrame(results)

    # Save the Task 2.1 numerical results.
    results_df.to_csv(
        RESULTS_DIR / "task_2_1_results.csv",
        index=False,
    )

    # Save the computed availability values for later report checks.
    availability.to_csv(
        RESULTS_DIR / "task_2_1_availability.csv",
        index=False,
    )

    # Plot all in-sample load profiles and overlay the reserve bids.
    plot_in_sample_profiles(
        load_profiles,
        results,
        RESULTS_DIR / "task_2_1_in_sample_profiles.png",
    )

    # Plot the distribution of hourly availability and overlay the reserve bids.
    plot_availability_distribution(
        availability,
        results,
        RESULTS_DIR / "task_2_1_availability_distribution.png",
    )

    # Print a compact summary in the terminal.
    print("\nTask 2.1 completed successfully.")
    print("--------------------------------")
    print(results_df[[
        "method",
        "reserve_bid_kW",
        "achieved_reliability",
        "n_violations",
        "expected_shortfall_kW",
        "max_shortfall_kW",
        "solve_time_s",
    ]])

    # Print the files generated by this workflow.
    print("\nGenerated files:")
    print(f"- {DATA_PATH}")
    print(f"- {RESULTS_DIR / 'task_2_1_results.csv'}")
    print(f"- {RESULTS_DIR / 'task_2_1_availability.csv'}")
    print(f"- {RESULTS_DIR / 'task_2_1_in_sample_profiles.png'}")
    print(f"- {RESULTS_DIR / 'task_2_1_availability_distribution.png'}")

# ============================================================
# 7. Out-of-sample verification for Task 2.2
# ============================================================

def verify_reserve_bid_out_of_sample(load_profiles, reserve_bid_kw, method):
    """
    Verify a fixed reserve bid on the 200 out-of-sample profiles.

    In Task 2.2 we do not solve a new optimization problem. Instead, we take
    the reserve bids obtained in Task 2.1 and check whether they also
    satisfy the P90 requirement on unseen load profiles.

    The verification is done at two levels:
    1. Profile level:
       - uses the hourly availability A_s = min load during the hour.
       - checks if A_s >= reserve bid.
    2. Minute level:
       - compares the reserve bid with the load at every minute.
       - calculates minute-by-minute shortfalls.

    Parameters
    ----------
    load_profiles : pandas.DataFrame
        Full load profile dataset with in-sample and out-of-sample profiles.

    reserve_bid_kw : float
        Fixed reserve bid from Task 2.1.

    method : str
        Name of the method used to obtain the bid, e.g. "ALSO-X" or "CVaR".

    Returns
    -------
    tuple
        summary, profile_shortfalls, minute_shortfalls
    """

    # Select only the 200 out-of-sample profiles.
    out_of_sample = load_profiles[load_profiles["set"] == "out_of_sample"].copy()

    # Compute the hourly reserve availability for each out-of-sample profile.
    # This is the minimum load over the 60 minutes of each profile.
    profile_shortfalls = compute_profile_availability(out_of_sample)

    # Store the method name and the fixed reserve bid used for verification.
    profile_shortfalls["method"] = method
    profile_shortfalls["reserve_bid_kW"] = reserve_bid_kw

    # Compute the profile-level shortfall.
    # If availability is lower than the bid, the difference is a shortfall.
    profile_shortfalls["profile_shortfall_kW"] = np.maximum(
        0.0,
        reserve_bid_kw - profile_shortfalls["availability_kW"],
    )

    # A profile violates the reserve requirement if its availability is below the bid.
    profile_shortfalls["profile_violation"] = (
        profile_shortfalls["availability_kW"] < reserve_bid_kw
    )

    # Compute minute-level shortfalls.
    # This compares the fixed bid with the load at every minute.
    minute_shortfalls = out_of_sample.copy()
    minute_shortfalls["method"] = method
    minute_shortfalls["reserve_bid_kW"] = reserve_bid_kw

    # If the load at a minute is lower than the bid, there is a shortfall.
    minute_shortfalls["minute_shortfall_kW"] = np.maximum(
        0.0,
        reserve_bid_kw - minute_shortfalls["load_kW"],
    )

    # A minute violates the reserve requirement if load is below the bid.
    minute_shortfalls["minute_violation"] = (
        minute_shortfalls["load_kW"] < reserve_bid_kw
    )

    # Count out-of-sample profiles.
    n_profiles = profile_shortfalls["profile"].nunique()

    # Count profile-level violations.
    n_violating_profiles = int(profile_shortfalls["profile_violation"].sum())

    # Compute achieved out-of-sample reliability.
    achieved_reliability = 1.0 - n_violating_profiles / n_profiles

    # Count minute-level violations.
    n_violating_minutes = int(minute_shortfalls["minute_violation"].sum())
    n_total_minutes = len(minute_shortfalls)

    # Compute all relevant summary metrics.
    summary = {
        "method": method,
        "fixed_reserve_bid_kW": round(float(reserve_bid_kw), 4),
        "n_out_of_sample_profiles": n_profiles,
        "achieved_out_of_sample_reliability": round(float(achieved_reliability), 4),
        "n_violating_profiles": n_violating_profiles,
        "expected_profile_shortfall_kW": round(
            float(profile_shortfalls["profile_shortfall_kW"].mean()), 4
        ),
        "max_profile_shortfall_kW": round(
            float(profile_shortfalls["profile_shortfall_kW"].max()), 4
        ),
        "n_total_minutes": n_total_minutes,
        "n_violating_minutes": n_violating_minutes,
        "minute_violation_share": round(
            float(n_violating_minutes / n_total_minutes), 4
        ),
        "expected_minute_shortfall_kW": round(
            float(minute_shortfalls["minute_shortfall_kW"].mean()), 4
        ),
        "max_minute_shortfall_kW": round(
            float(minute_shortfalls["minute_shortfall_kW"].max()), 4
        ),
    }

    return summary, profile_shortfalls, minute_shortfalls


def plot_out_of_sample_profiles(load_profiles, bids, output_path):
    """
    Plot the 200 out-of-sample load profiles and the fixed reserve bids.

    This figure is useful to visually check whether the bids obtained in
    Task 2.1 remain reasonable for unseen profiles.
    """

    # Select only out-of-sample profiles.
    out_of_sample = load_profiles[load_profiles["set"] == "out_of_sample"]

    # Create a new figure.
    plt.figure(figsize=(9, 5))

    # Plot each out-of-sample profile.
    for _, group in out_of_sample.groupby("profile"):
        plt.plot(
            group["minute"],
            group["load_kW"],
            linewidth=0.7,
            alpha=0.25,
        )

    # Add one horizontal line for each fixed reserve bid.
    for method, bid in bids.items():
        plt.axhline(
            bid,
            linestyle="--",
            linewidth=2,
            label=f"{method} bid = {bid:.2f} kW",
        )

    # Add labels, title and legend.
    plt.xlabel("Minute")
    plt.ylabel("Load / available upward reserve [kW]")
    plt.title("Out-of-sample flexible load profiles and fixed reserve bids")
    plt.legend()

    # Save the figure.
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_out_of_sample_shortfalls(profile_shortfalls, output_path):
    """
    Plot the distribution of profile-level out-of-sample shortfalls.

    The profile-level shortfall is based on the minimum load in each
    out-of-sample profile. This is the most relevant shortfall measure for
    the hourly reserve bid, because the bid must be feasible throughout
    the full hour.
    """

    # Create a new figure.
    plt.figure(figsize=(8, 5))

    # Define common bins so ALSO-X and CVaR can be compared fairly.
    max_shortfall = max(profile_shortfalls["profile_shortfall_kW"].max(), 1.0)
    bins = np.linspace(0.0, max_shortfall, 16)

    # Plot one histogram per method.
    for method in profile_shortfalls["method"].unique():
        method_data = profile_shortfalls[
            profile_shortfalls["method"] == method
        ]

        plt.hist(
            method_data["profile_shortfall_kW"],
            bins=bins,
            alpha=0.65,
            edgecolor="black",
            label=method,
        )

    # Add labels, title and legend.
    plt.xlabel("Profile-level reserve shortfall [kW]")
    plt.ylabel("Number of out-of-sample profiles")
    plt.title("Out-of-sample reserve shortfall distribution")
    plt.legend()

    # Save the figure.
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_positive_out_of_sample_shortfalls(profile_shortfalls, output_path):
    """
    Plot the distribution of positive profile-level out-of-sample shortfalls only.

    This figure excludes the many profiles with zero shortfall, so it provides
    a clearer comparison of the severity of shortfalls under ALSO-X and CVaR.
    """

    # Keep only profiles with strictly positive shortfall.
    positive_shortfalls = profile_shortfalls[
        profile_shortfalls["profile_shortfall_kW"] > 0
    ].copy()

    # If no positive shortfalls exist, skip the plot safely.
    if positive_shortfalls.empty:
        print("No positive shortfalls found. Positive-shortfall plot not created.")
        return

    # Create a new figure.
    plt.figure(figsize=(8, 5))

    # Define common bins for a fair comparison between methods.
    max_shortfall = positive_shortfalls["profile_shortfall_kW"].max()
    bins = np.linspace(0.0, max_shortfall, 12)

    # Plot one histogram per method.
    for method in positive_shortfalls["method"].unique():
        method_data = positive_shortfalls[
            positive_shortfalls["method"] == method
        ]

        plt.hist(
            method_data["profile_shortfall_kW"],
            bins=bins,
            alpha=0.65,
            edgecolor="black",
            label=method,
        )

    # Add labels, title and legend.
    plt.xlabel("Positive profile-level reserve shortfall [kW]")
    plt.ylabel("Number of violating out-of-sample profiles")
    plt.title("Distribution of positive out-of-sample reserve shortfalls")
    plt.legend()

    # Save the figure.
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_task_2_2():
    """
    Run Task 2.2 completely.

    This function:
    1. Loads the same 300 load profiles used in Task 2.1.
    2. Recomputes the Task 2.1 bids from the in-sample profiles.
    3. Tests those fixed bids on the 200 out-of-sample profiles.
    4. Saves result tables and figures for the report.
    """

    # Make sure the results folder exists.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load the existing load profiles.
    load_profiles = load_or_generate_profiles()

    # Compute profile-level availability for all 300 profiles.
    availability = compute_profile_availability(load_profiles)

    # Select the 100 in-sample profiles.
    in_sample_availability = availability.loc[
        availability["set"] == "in_sample",
        "availability_kW",
    ].to_numpy()

    # Recompute the fixed bids from Task 2.1.
    # This avoids manually hard-coding the bid values.
    alsox_result = solve_alsox_p90(
        in_sample_availability,
        reliability=RELIABILITY_P90,
    )

    cvar_result = solve_cvar_p90(
        in_sample_availability,
        reliability=RELIABILITY_P90,
    )

    # Store the fixed bids in a dictionary.
    bids = {
        "ALSO-X": alsox_result["reserve_bid_kW"],
        "CVaR": cvar_result["reserve_bid_kW"],
    }

    # Lists used to collect results from both methods.
    summary_rows = []
    profile_shortfall_tables = []
    minute_shortfall_tables = []

    # Verify each fixed bid on the out-of-sample profiles.
    for method, bid in bids.items():

        # Run out-of-sample verification for this method.
        summary, profile_shortfalls, minute_shortfalls = (
            verify_reserve_bid_out_of_sample(
                load_profiles=load_profiles,
                reserve_bid_kw=bid,
                method=method,
            )
        )

        # Store results.
        summary_rows.append(summary)
        profile_shortfall_tables.append(profile_shortfalls)
        minute_shortfall_tables.append(minute_shortfalls)

    # Convert results to DataFrames.
    results_df = pd.DataFrame(summary_rows)
    profile_shortfalls_df = pd.concat(profile_shortfall_tables, ignore_index=True)
    minute_shortfalls_df = pd.concat(minute_shortfall_tables, ignore_index=True)

    # Save numerical results for the report.
    results_df.to_csv(
        RESULTS_DIR / "task_2_2_results.csv",
        index=False,
    )

    profile_shortfalls_df.to_csv(
        RESULTS_DIR / "task_2_2_profile_shortfalls.csv",
        index=False,
    )

    minute_shortfalls_df.to_csv(
        RESULTS_DIR / "task_2_2_minute_shortfalls.csv",
        index=False,
    )

    # Plot the out-of-sample profiles and fixed bids.
    plot_out_of_sample_profiles(
        load_profiles,
        bids,
        RESULTS_DIR / "task_2_2_out_of_sample_profiles.png",
    )

    # Plot the distribution of out-of-sample shortfalls.
    plot_out_of_sample_shortfalls(
        profile_shortfalls_df,
        RESULTS_DIR / "task_2_2_shortfall_distribution.png",
    )

    # Plot only the positive profile-level shortfalls.
    plot_positive_out_of_sample_shortfalls(
        profile_shortfalls_df,
        RESULTS_DIR / "task_2_2_positive_shortfall_distribution.png",
    )

    # Print a compact summary in the terminal.
    print("\nTask 2.2 completed successfully.")
    print("--------------------------------")
    print(results_df[[
        "method",
        "fixed_reserve_bid_kW",
        "achieved_out_of_sample_reliability",
        "n_violating_profiles",
        "expected_profile_shortfall_kW",
        "max_profile_shortfall_kW",
    ]])

    # Print generated files.
    print("\nGenerated files:")
    print(f"- {RESULTS_DIR / 'task_2_2_results.csv'}")
    print(f"- {RESULTS_DIR / 'task_2_2_profile_shortfalls.csv'}")
    print(f"- {RESULTS_DIR / 'task_2_2_minute_shortfalls.csv'}")
    print(f"- {RESULTS_DIR / 'task_2_2_out_of_sample_profiles.png'}")
    print(f"- {RESULTS_DIR / 'task_2_2_shortfall_distribution.png'}")
    print(f"- {RESULTS_DIR / 'task_2_2_positive_shortfall_distribution.png'}")

    # ============================================================
# 8. Energinet perspective for Task 2.3
# ============================================================

def run_task_2_3():
    """
    Run Task 2.3:
    analyse the effect of the reliability requirement on:
    - the optimal ALSO-X reserve bid,
    - the expected reserve shortfall.
    """

    # Load data
    load_profiles = pd.read_csv(DATA_PATH)

    # Compute hourly reserve availability
    availability = (
        load_profiles.groupby(['profile', 'set'])['load_kW']
        .min()
        .reset_index()
        .rename(columns={'load_kW': 'availability_kW'})
    )

    # Split data
    in_sample = availability[
        availability['set'] == 'in_sample'
    ]['availability_kW'].values

    out_sample = availability[
        availability['set'] == 'out_of_sample'
    ]['availability_kW'].values

    # ALSO-X reserve bid function
    def alsox(avail, rel):

        avail_sorted = np.sort(avail)
        n = len(avail_sorted)

        if rel >= 0.999:
            return avail_sorted[0]

        k = int(np.floor((1 - rel) * n))
        k = min(max(k, 0), n - 1)

        return avail_sorted[k]

    # Run analysis
    rows = []

    for r in np.arange(0.80, 1.01, 0.01):

        bid = alsox(in_sample, r)

        shortfall = np.maximum(0, bid - out_sample)

        rows.append([r, bid, np.mean(shortfall)])

    # Create results table
    res = pd.DataFrame(
        rows,
        columns=['Reliability', 'Bid', 'Expected Shortfall']
    )

    # Plot reserve bid versus reliability
    plt.figure()

    plt.plot(res['Reliability'], res['Bid'], marker='o')

    plt.xlabel('Reliability')
    plt.ylabel('Reserve Bid (kW)')
    plt.title('Reliability vs Reserve Bid')

    plt.grid()

    plt.savefig(
        RESULTS_DIR / 'task_2_3_reserve_bid.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    # Plot expected shortfall versus reliability
    plt.figure()

    plt.plot(
        res['Reliability'],
        res['Expected Shortfall'],
        marker='o'
    )

    plt.xlabel('Reliability')
    plt.ylabel('Expected Shortfall (kW)')
    plt.title('Reliability vs Shortfall')

    plt.grid()

    plt.savefig(
        RESULTS_DIR / 'task_2_3_shortfall.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    # Save results
    res.to_csv(
        RESULTS_DIR / 'task_2_3_results.csv',
        index=False
    )

    print("\nTask 2.3 completed successfully.")
    print("--------------------------------")
    print(res)

def run_step2():
    """
    Run the Step 2 workflow.

    Task 2.1 determines the in-sample reserve bids.
    Task 2.2 verifies those fixed bids on the out-of-sample profiles.
    Task 2.3
    """

    # Run Task 2.1.
    run_task_2_1()

    # Run Task 2.2.
    run_task_2_2()

    # Run Task 2.3
    run_task_2_3()