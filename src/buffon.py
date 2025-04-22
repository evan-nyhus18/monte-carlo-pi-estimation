"""
Buffon’s Needle simulation methods (vectorized + visualization).
"""
import math
import numpy as np
import matplotlib.pyplot as plt

def simulate_buffon_vectorized(needle_length, line_distance, n_drops, chunk_size=10**7):
    """
    Perform a vectorized Buffon's Needle simulation using chunking.

    Returns total_hits: number of crossings.
    """
    total_hits = 0
    rng = np.random.default_rng()
    remaining = n_drops

    while remaining > 0:
        current_chunk = min(chunk_size, remaining)
        xs = rng.uniform(0, line_distance / 2, current_chunk)
        thetas = rng.uniform(0, np.pi / 2, current_chunk)
        hits = np.count_nonzero(xs <= (needle_length / 2) * np.sin(thetas))
        total_hits += hits
        remaining -= current_chunk

    return total_hits

def refined_simulation_vectorized(needle_length, line_distance, n_drops, chunk_size=10**7):
    """
    Run optimized simulation, compute observed/theoretical probabilities & π approximation.
    """
    hits = simulate_buffon_vectorized(needle_length, line_distance, n_drops, chunk_size)
    observed_prob = hits / n_drops

    if needle_length <= line_distance:
        theo_prob = (2 * needle_length) / (line_distance * math.pi)
        approx_pi = (2 * needle_length * n_drops) / (line_distance * hits) if hits else None
    else:
        term1 = (needle_length - math.sqrt(needle_length**2 - line_distance**2)) / line_distance
        term2 = math.acos(line_distance / needle_length)
        theo_prob = (2 / math.pi) * (term1 + term2)
        approx_pi = (2 * n_drops / hits) * (term1 + term2) if hits else None

    print("\n--- Optimized Vectorized Simulation Results ---")
    print(f"Needle length (l): {needle_length}")
    print(f"Distance between lines (d): {line_distance}")
    print(f"Drops (N): {n_drops}, Hits: {hits}")
    print(f"Observed P: {observed_prob:.6f}, Theoretical P: {theo_prob:.6f}")
    print(f"Approximated π: {approx_pi:.6f}" if approx_pi else "No hits; π undefined.")

    return observed_prob, theo_prob, approx_pi

def visualize_buffon_experiment(needle_length, line_distance, n_samples=20):
    """
    Plot a handful of needle drops on parallel lines to illustrate crossings.
    """
    rng = np.random.default_rng()
    board_width = 10

    x_centers = rng.uniform(0, board_width, n_samples)
    y_centers = rng.uniform(0, line_distance, n_samples)
    thetas    = rng.uniform(0, np.pi, n_samples)

    plt.figure(figsize=(8, 4))
    plt.axhline(0, color='k', lw=2)
    plt.axhline(line_distance, color='k', lw=2)

    for x, y, theta in zip(x_centers, y_centers, thetas):
        dx = (needle_length / 2) * np.cos(theta)
        dy = (needle_length / 2) * np.sin(theta)
        x1, y1 = x - dx, y - dy
        x2, y2 = x + dx, y + dy
        color = 'red' if (y1 < 0 or y2 > line_distance) else 'blue'
        plt.plot([x1, x2], [y1, y2], color=color, lw=2)
        plt.plot(x, y, 'ko', markersize=3)

    plt.xlim(0, board_width)
    plt.ylim(-0.1*line_distance, 1.1*line_distance)
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("Visualization of Buffon's Needle Drops")
    plt.show()

def average_pi_approximation(needle_length, line_distance, n_drops, n_trials, chunk_size=10**7):
    """
    Run multiple trials and return average π approximation.
    """
    pi_values = []
    for _ in range(n_trials):
        hits = simulate_buffon_vectorized(needle_length, line_distance, n_drops, chunk_size)
        if hits:
            if needle_length <= line_distance:
                pi_values.append((2 * needle_length * n_drops) / (line_distance * hits))
            else:
                term1 = (needle_length - math.sqrt(needle_length**2 - line_distance**2)) / line_distance
                term2 = math.acos(line_distance / needle_length)
                pi_values.append((2 * n_drops / hits) * (term1 + term2))

    if pi_values:
        avg_pi = sum(pi_values) / len(pi_values)
        print(f"\nAverage π over {n_trials} trials: {avg_pi:.6f}")
        return avg_pi
    print("\nNo valid π approximations obtained.")
    return None

def main():
    print("🔍 Buffon's Needle Simulation")
    l = float(input("Needle length (l): "))
    d = float(input("Line distance (d): "))
    N = int(input("Drops (N): "))
    chunk = int(input("Chunk size (e.g. 10000000): "))
    refined_simulation_vectorized(l, d, N, chunk)

    if input("Average π over multiple trials? (y/n): ")=='y':
        trials = int(input("Number of trials: "))
        average_pi_approximation(l, d, N, trials, chunk)

    if input("Visualize drops? (y/n): ")=='y':
        samples = int(input("Samples to plot: "))
        visualize_buffon_experiment(l, d, samples)

if __name__ == "__main__":
    main()
