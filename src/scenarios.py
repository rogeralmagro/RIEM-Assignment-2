import numpy as np
import pandas as pd


def load_wind_scenarios(path="data/wind_scenarios.csv"):
    df = pd.read_csv(path)

    wind_matrix = df.pivot(
        index="scenario",
        columns="hour",
        values="wind_MW"
    )

    wind_matrix = wind_matrix.sort_index(axis=0).sort_index(axis=1)

    return wind_matrix.to_numpy()


def load_price_scenarios(path="data/dk2_prices_raw.csv", n_days=20):
    df = pd.read_csv(path, sep=";", decimal=",")

    df["HourDK"] = pd.to_datetime(df["HourDK"])
    df["date"] = df["HourDK"].dt.date
    df["hour"] = df["HourDK"].dt.hour + 1

    price_matrix = df.pivot(
        index="date",
        columns="hour",
        values="SpotPriceEUR"
    )

    price_matrix = price_matrix.dropna()
    price_matrix = price_matrix.iloc[:n_days]

    return price_matrix.to_numpy()


def generate_imbalance_scenarios(n_scenarios=4, n_hours=24, seed=42):
    rng = np.random.default_rng(seed)

    # 1 = system deficit, 0 = system surplus
    return rng.binomial(1, 0.5, size=(n_scenarios, n_hours))


def generate_scenarios():
    wind_scenarios = load_wind_scenarios()
    price_scenarios = load_price_scenarios()
    imbalance_scenarios = generate_imbalance_scenarios()

    scenarios = []

    for wind in wind_scenarios:
        for da_price in price_scenarios:
            for imbalance in imbalance_scenarios:

                balancing_price = np.where(
                    imbalance == 1,
                    1.25 * da_price,
                    0.85 * da_price
                )

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