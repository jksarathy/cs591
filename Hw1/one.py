from common import hamming_dist
from common import frac_incorrect
from scipy.linalg import hadamard
from scipy.linalg import lstsq
from scipy.optimize import least_squares

import numpy as np
import math
import matplotlib.pyplot as plt
import sys


# Hadamard attack.
def attack_had(a, n):
    H = hadamard(n) # generate Hadamard matrix.
    z = H * a
    x_hat = np.rint(z) 
    return x_hat # return vector of guesses.

# Random query attack.
def attack_rand(a, B, n):
    z, _, _, _ = lstsq((1.0/float(n)) * B, a) # calculate argmin.
    x_hat = np.rint(z)
    return x_hat # return vector of guesses.


# Release mechanism with Hadamard queries.
def generate_had(n, theta):
    x = np.random.randint(2, size = (n, 1)) # sample uniformly random x.
    H = hadamard(n) # generate Hadamard matrix
    Y = np.random.normal(0, theta**2, (n, 1)) # sample error from gaussian.
    a = (1.0/float(n)) * H * np.matrix(x) + Y  # calculate answers.
    return (x, a)

# Release mechanism with random queries.
def generate_rand(n, m, theta):
    x = np.random.randint(2, size = (n, 1)) # sample uniformly random x.
    B = np.random.randint(2, size = (m, n)) # sample random query matrix.
    Y = np.random.normal(0, theta**2, (m, 1)) # sample error from gaussian.
    a = (1.0/float(n)) * B * np.matrix(x) + Y  # calculate answers.
    return (x, a, B)

# Run Hadamard and Random Query attack with specified parameters.
# Use fresh randomness for each trial.
def run_trials(trials, n, m, theta): 
    results_had = np.array([])
    results_rand = np.array([])
    for i in range(trials):
        x_had, a_had = generate_had(n, theta) # Get Hadamard answers.
        x_hat_had = attack_had(a_had, n) # Run Hadamard attack to get guesses.
        x_rand, a_rand, B = generate_rand(n, m, theta) # Get Random Query answers.
        x_hat_rand = attack_rand(a_rand, B, n) # Run Random Query attack to get guesses.
        results_had = np.append(results_had, frac_incorrect(x_had, x_hat_had, n)) # Store fraction incorrect for Hadamard.
        results_rand = np.append(results_rand, frac_incorrect(x_rand, x_hat_rand, n)) # Store fraction incorrect for Random.
    return (results_had, results_rand)

# Compute mean and standard deviation of trials of Hadamard and Random query attack.
def compute_stats(mean_had, mean_rand, std_had, std_rand, results_had, results_rand):
    mean_had = np.append(mean_had, np.mean(results_had))
    mean_rand = np.append(mean_rand, np.mean(results_rand))
    std_had = np.append(std_had, np.std(results_had))
    std_rand = np.append(std_rand, np.std(results_rand))
    return (mean_had, mean_rand, std_had, std_rand)

# Plot results.
def plot_trials(ax, dep_var, dep_label, mean_had, mean_rand, std_had, std_rand):
    ax.plot(dep_var, mean_had, 'bo-', label='Mean for Hadamard')
    ax.plot(dep_var, std_had, 'b^:', label='Standard deviation for Hadamard')
    ax.plot(dep_var, mean_rand, 'go-', label='Mean for Random')
    ax.plot(dep_var, std_rand, 'g^:', label='Standard deviation for Random')
    ax.set_xlabel(dep_label)
    ax.set_ylabel('Fraction of bits of x incorrectly recovered')
    #ax.set_yscale("log")
    ax.set_title("Hadamard vs. Random Query Attack as " + dep_label + " varies.")
    ax.legend(loc='best')


def main():
    # Set number of trials.
    trials = 20

    # Initialize the plot and axes.
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Plot n as dependent variable.
    # Fix theta and m/n.
    theta = 2**(-3) 
    n_vals = [16, 32, 64]
    # Initialize arrays for storing statistics.
    mean_had, mean_rand, std_had, std_rand = np.array([]), np.array([]), np.array([]), np.array([])
    for n in n_vals:
        m = 4*n 
        results_had, results_rand = run_trials(trials, n, m, theta)
        mean_had, mean_rand, std_had, std_rand = compute_stats(mean_had, mean_rand, std_had, std_rand, results_had, results_rand)
    plot_trials(ax1, n_vals, "n", mean_had, mean_rand, std_had, std_rand)

    # Plot m as dependent variable.
    # Fix theta and n.
    n = 16
    theta = 2**(-3)
    m_vals = [int(1.1*n), 4*n, 16*n]
    # Re-initialize arrays for storing statistics.
    mean_had, mean_rand, std_had, std_rand = np.array([]), np.array([]), np.array([]), np.array([])
    for m in m_vals:
        results_had, results_rand = run_trials(trials, n, m, theta)
        mean_had, mean_rand, std_had, std_rand = compute_stats(mean_had, mean_rand, std_had, std_rand, results_had, results_rand)
    plot_trials(ax2, m_vals, "m", mean_had, mean_rand, std_had, std_rand)

    # Plot theta as dependent variable.
    # Fix n and m.
    n = 16
    m = 4*n
    theta_vals = [2**(-i) for i in range(1, int(math.sqrt(32*n)) + 1)] 
    # Re-nitialize arrays for storing statistics.
    mean_had, mean_rand, std_had, std_rand = np.array([]), np.array([]), np.array([]), np.array([])
    for theta in theta_vals:
        print str(theta)
        results_had, results_rand = run_trials(trials, n, m, theta)
        mean_had, mean_rand, std_had, std_rand = compute_stats(mean_had, mean_rand, std_had, std_rand, results_had, results_rand)
    plot_trials(ax3, theta_vals, "theta", mean_had, mean_rand, std_had, std_rand)

    # Display plot.
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()

