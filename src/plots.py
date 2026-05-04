import matplotlib.pyplot as plt
import os

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