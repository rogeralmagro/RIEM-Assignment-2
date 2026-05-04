import numpy as np
import pandas as pd


def load_wind_scenarios(path="data/wind_scenarios.csv"):
    """Load wind scenarios and reshape into (n_scenarios × 24h) matrix."""
    
    df = pd.read_csv(path)

    # Convert long format → matrix (rows=scenarios, columns=hours)
    wind_matrix = df.pivot(
        index="scenario",
        columns="hour",
        values="wind_MW"
    )

    # Ensure correct ordering (important for hour alignment)
    wind_matrix = wind_matrix.sort_index(axis=0).sort_index(axis=1)

    return wind_matrix.to_numpy()


def load_price_scenarios(path="data/dk2_prices_raw.csv", n_days=20):
    """Load day-ahead prices and build daily price scenarios."""
    
    df = pd.read_csv(path, sep=";", decimal=",")

    # Convert timestamp column to datetime
    df["HourDK"] = pd.to_datetime(df["HourDK"])

    # Extract day and hour (1–24)
    df["date"] = df["HourDK"].dt.date
    df["hour"] = df["HourDK"].dt.hour + 1

    # Pivot → each row is one day, columns are hours
    price_matrix = df.pivot(
        index="date",
        columns="hour",
        values="SpotPriceEUR"
    )

    # Remove incomplete days and keep first n_days
    price_matrix = price_matrix.dropna()
    price_matrix = price_matrix.iloc[:n_days]

    return price_matrix.to_numpy()


def generate_imbalance_scenarios(n_scenarios=4, n_hours=24, seed=42):
    """Generate binary imbalance scenarios (1=deficit, 0=surplus)."""
    
    rng = np.random.default_rng(seed)

    # Each hour independently: 50% deficit / 50% surplus
    return rng.binomial(1, 0.5, size=(n_scenarios, n_hours))


def generate_scenarios():
    """Combine wind, price, and imbalance into full stochastic scenarios."""
    
    wind_scenarios = load_wind_scenarios()
    price_scenarios = load_price_scenarios()
    imbalance_scenarios = generate_imbalance_scenarios()

    scenarios = []

    # Cartesian product of all uncertainties
    for wind in wind_scenarios:
        for da_price in price_scenarios:
            for imbalance in imbalance_scenarios:

                # Compute balancing price depending on system state
                balancing_price = np.where(
                    imbalance == 1,
                    1.25 * da_price,
                    0.85 * da_price
                )

                # Store complete 24h scenario
                scenarios.append({
                    "wind": wind,
                    "da_price": da_price,
                    "imbalance": imbalance,
                    "balancing_price": balancing_price
                })

    print(f"Wind scenarios: {wind_scenarios.shape}")
    print(f"Price scenarios: {price_scenarios.shape}")
    print(f"Imbalance scenarios: {imbalance_scenarios.shape}")
    print(f"Combined scenarios: {len(scenarios)}")

    return scenarios