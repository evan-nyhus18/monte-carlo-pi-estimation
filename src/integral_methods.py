"""
Monte Carlo π estimation via average‑value and area methods.
"""
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """Function for quarter‑circle: y = sqrt(1 - x^2)."""
    return np.sqrt(1 - x**2)

def estimate_pi_avg(n):
    """Average‑value method: E[f(x)] * 4."""
    x = np.random.uniform(0, 1, n)
    return 4 * np.mean(f(x))

def estimate_pi_area(n):
    """Area‑method: fraction of points under curve * 4."""
    x = np.random.uniform(0, 1, n)
    y = np.random.uniform(0, 1, n)
    count = np.sum(y < f(x))
    return 4 * (count / n)

def run_single_estimates(sample_sizes):
    """Print π estimates for each n in sample_sizes."""
    print("n".ljust(10) + "Avg Method".ljust(20) + "Area Method")
    for n in sample_sizes:
        print(f"{n:<10}{estimate_pi_avg(n):<20.8f}{estimate_pi_area(n):.8f}")

def run_distribution(n_fixed, num_runs=1000):
    """
    Generate histograms of π estimates at fixed n for both methods.
    """
    avg_est = np.array([estimate_pi_avg(n_fixed) for _ in range(num_runs)])
    area_est = np.array([estimate_pi_area(n_fixed) for _ in range(num_runs)])

    print("\nMonte Carlo Distribution:")
    print(f"Avg method:  mean={avg_est.mean():.6f}, std={avg_est.std():.6f}")
    print(f"Area method: mean={area_est.mean():.6f}, std={area_est.std():.6f}")

    bins = np.linspace(np.pi - 0.05, np.pi + 0.05, 30)
    plt.figure(figsize=(8,6))
    plt.hist(avg_est, bins=bins, alpha=0.6, label='Avg method')
    plt.hist(area_est, bins=bins, alpha=0.6, label='Area method')
    plt.xlabel("π estimate"); plt.ylabel("Frequency")
    plt.title(f"Distribution (n={n_fixed}, runs={num_runs})")
    plt.legend(); plt.grid(True)
    plt.show()

def demo_area_method(n_demo=500):
    """Scatter of points vs. quarter‑circle curve."""
    x_demo = np.random.uniform(0, 1, n_demo)
    y_demo = np.random.uniform(0, 1, n_demo)
    inside = y_demo < f(x_demo)

    x_curve = np.linspace(0,1,300)
    plt.figure(figsize=(6,6))
    plt.plot(x_curve, f(x_curve), 'k-', lw=2, label='Curve')
    plt.scatter(x_demo[inside], y_demo[inside], edgecolor='k', label='Inside')
    plt.scatter(x_demo[~inside], y_demo[~inside], edgecolor='k', label='Outside')
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Area Method Demo"); plt.legend(); plt.grid(True)
    plt.show()
