import matplotlib.pyplot as plt


def plot_profit_distribution(profits, title="Profit distribution"):
    plt.figure()
    plt.hist(profits, bins=50)
    plt.title(title)
    plt.xlabel("Profit")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()