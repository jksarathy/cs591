from common import hamming_dist
from common import frac_incorrect
from scipy.linalg import hadamard
from scipy.linalg import lstsq
from scipy.optimize import least_squares

import numpy as np
import math
import matplotlib.pyplot as plt
import sys


def attack_had(a, n):
    H = hadamard(n)
    z = H * a
    x_hat = np.rint(z)
    return x_hat


def attack_rand(a, B, n):
    z, _, _, _ = lstsq((1.0/float(n)) * B, a)
    x_hat = np.rint(z)
    return x_hat


def generate_had(n, theta):
    x = np.random.randint(2, size = (n, 1))
    H = hadamard(n)
    Y = np.random.normal(0, theta*theta, (n, 1))
    a = (1.0/float(n)) * H * np.matrix(x) + Y   
    return (x, a)

def generate_rand(n, m, theta):
    x = np.random.randint(2, size = (n, 1))
    B = np.random.randint(2, size = (m, n))
    Y = np.random.normal(0, theta*theta, (m, 1))
    a = (1.0/float(n)) * B * np.matrix(x) + Y   
    return (x, a, B)

def run_trials(trials, n, m, theta):
    results_had = np.array([])
    results_rand = np.array([])
    for i in range(trials):
        x_had, a_had = generate_had(n, theta)
        x_hat_had = attack_had(a_had, n)
        x_rand, a_rand, B = generate_rand(n, m, theta)
        x_hat_rand = attack_rand(a_rand, B, n)
        results_had = np.append(results_had, frac_incorrect(x_had, x_hat_had, n))
        results_rand = np.append(results_rand, frac_incorrect(x_rand, x_hat_rand, n))
    return (results_had, results_rand)

def compute_stats(mean_had, mean_rand, std_had, std_rand, results_had, results_rand):
    mean_had = np.append(mean_had, np.mean(results_had))
    mean_rand = np.append(mean_rand, np.mean(results_rand))
    std_had = np.append(std_had, np.std(results_had))
    std_rand = np.append(std_rand, np.std(results_rand))
    return (mean_had, mean_rand, std_had, std_rand)

def plot_trials(dep_var, dep_label, mean_had, mean_rand, std_had, std_rand):
    plt.plot(dep_var, mean_had, 'bo-', label='Mean for Hadamard')
    plt.plot(dep_var, std_had, 'b^:', label='Standard deviationfor Hadamard')
    plt.plot(dep_var, mean_rand, 'go-', label='Mean for Random')
    plt.plot(dep_var, std_rand, 'g^:', label='Standard deviation for Random')
    plt.xlabel(dep_label)
    plt.ylabel('Fraction of bits of x incorrectly recovered')
    plt.title("Hadamard vs. Random Query Attack as " + dep_label + " varies.")
    plt.legend(loc='best')
    plt.show()


def main():
    trials = 20

    mean_had, mean_rand, std_had, std_rand = np.array([]), np.array([]), np.array([]), np.array([])
    theta = 2**(-3)
    #n_vals = [128, 512] # 2048, 8192]

    # Plot n as dependent variable
    n_vals = [16, 32, 64]
    for n in n_vals:
        m = 4*n
        results_had, results_rand = run_trials(trials, n, m, theta)
        mean_had, mean_rand, std_had, std_rand = compute_stats(mean_had, mean_rand, std_had, std_rand, results_had, results_rand)
    plot_trials(n_vals, "n", mean_had, mean_rand, std_had, std_rand)

    # Plot m as dependent variable
    mean_had, mean_rand, std_had, std_rand = np.array([]), np.array([]), np.array([]), np.array([])
    n = 16
    theta = 2**(-3)
    m_vals = [int(1.1*n), 4*n, 16*n]
    for m in m_vals:
        results_had, results_rand = run_trials(trials, n, m, theta)
        mean_had, mean_rand, std_had, std_rand = compute_stats(mean_had, mean_rand, std_had, std_rand, results_had, results_rand)
    plot_trials(m_vals, "m", mean_had, mean_rand, std_had, std_rand)

    # Plot theta as dependent variable
    mean_had, mean_rand, std_had, std_rand = np.array([]), np.array([]), np.array([]), np.array([])
    n = 16
    m = 4*n
    ln_theta_vals = [i for i in range(1, int(math.sqrt(32*n)) + 1)]
    theta_vals = [2**(-i) for i in range(1, int(math.sqrt(32*n)) + 1)] 
    for theta in theta_vals:
        print str(theta)
        results_had, results_rand = run_trials(trials, n, m, theta)
        mean_had, mean_rand, std_had, std_rand = compute_stats(mean_had, mean_rand, std_had, std_rand, results_had, results_rand)
    plot_trials(ln_theta_vals, "ln(theta)", mean_had, mean_rand, std_had, std_rand)



if __name__ == "__main__":
    main()
    #test()

