from common import hamming_dist
from common import frac_correct
import matplotlib.pyplot as plt
import numpy as np
import sys
import random

# Running counter attack a (without extra information).
def attack_a(a, n):
    # Set default guesses.
    x_hat = [0] * n 

    # Based on a[i], set x[i] to the most likely value.
    # For first bit:
    if a[0] == 0:
        x_hat[0] = 0 # with probability 1.
    elif a[0] == 2:
        x_hat[0] = 1 # with probability 1.

    # For rest of bits:
    for i in range(1, n):
        a_diff = a[i]-a[i-1] 
        if a_diff == 2:
            x_hat[i] = 1 # with probability 1.
        elif a_diff == -1:
            x_hat[i] = 0 # with probability 1.
        elif a_diff == 1:
            x_hat[i] = 1 # with probability 2/3.
        else:
            x_hat[i] = 0 # with probability 2/3.
    return x_hat


# Running counter attack b (with extra information, w).
def attack_b(a, w, n):
    # Set default guesses to w.
    x_hat = [w[i] for i in range(n)] 

    # Initialize z for storing noise bits we are certain about.
    z = [-1] * n

    # Based on a[i], set x[i] to the most likely value.
    # For first bit:
    if a[0] == 0:
        x_hat[0] = 0 # with probability 1
        z[0] = 0 # store z[0].
    elif a[0] == 2:
        x_hat[0] = 1 # with probability 1
        z[0] = 1 # store z[0].

    # For rest of bits:
    for i in range(1, n):
        a_diff = a[i]-a[i-1] 
        if a_diff == 2:
            x_hat[i] = 1 # with probability 1
            z[i] = 1 # store z[i].

        elif a_diff == -1:
            x_hat[i] = 0 # with probability 1
            z[i] = 0 # store z[i].

        elif a_diff == 1:
            if z[i-1] == 1:
                x_hat[i] = 1 # with probability 1
                z[i] = 1 # store z[i].
            elif z[i-1] == 0:
                x_hat[i] = w[i] # with probability 2/3
            else:
                x_hat[i] = 1 # with probability 2/3

        elif a_diff == 0:
            if z[i-1] == 0:
                x_hat[i] = 0 # with probability 1
                z[i] = 0 # store z[i].
            elif z[i-1] == 1:
                x_hat[i] = w[i] # with probability 2/3
            else:
                x_hat[i] = 0 # with probability 2/3

    return x_hat


# Generate noisy running counter
def generate_data(n):
    x = np.random.randint(2, size = n) # Sample x uniformly at random
    exact_answers = np.cumsum(x) # Compute running counter of x
    z = np.random.randint(2, size = n) # Generate noise for each bit
    a = exact_answers + z # Create noisy running counter
    return (x.tolist(), exact_answers.tolist(), a.tolist())

# Generate extra information, w
def generate_w(x, n):
    w = [1] * n
    for i in range(n):
        rand = random.randint(0, 2) # Sample random integer in [0,2]
        if rand < 2:
            w[i] = x[i] # w[i] is correct with probability 2/3
        else:
            w[i] = (x[i] + 1) % 2 # w[i] is incorrect with probability 1/3
    return w


def main():
    # Set number of trials.
    trials = 50

    # Initialize arrays for storing statistics.
    mean_a, mean_b, std_a, std_b = np.array([]), np.array([]), np.array([]), np.array([])

    # Loop through values of n.
    n_vals = [100, 500, 1000, 5000]
    for n in n_vals:
        # Initialize arrays for storing results.
        results_a, results_b = np.array([]), np.array([])

        # Run trials.
        for i in range(trials):
            x, _, a = generate_data(n) # Get x and noisy running counter.
            w = generate_w(x, n) # Get extra information.

            x_hat_a = attack_a(a, n) # Run attack a to get guesses.
            x_hat_b = attack_b(a, w, n) # Run attack b to get guesses.

            results_a = np.append(results_a, frac_correct(x, x_hat_a, n)) # Store fraction recovered for attack a.
            results_b = np.append(results_b, frac_correct(x, x_hat_b, n)) # Store fraction recovered by attack b.

        # Compute mean and standard deviation for both attacks.
        mean_a = np.append(mean_a, np.mean(results_a))
        mean_b = np.append(mean_b, np.mean(results_b))
        std_a = np.append(std_a, np.std(results_a))
        std_b = np.append(std_b, np.std(results_b))

    print "Mean a: " + str(mean_a)
    print "Mean b: " + str(mean_b)

    # Plot results.
    plt.plot(n_vals, mean_a, 'bo-', label='Mean without extra info')
    plt.plot(n_vals, std_a, 'b^:', label='Standard deviation without extra info')
    plt.plot(n_vals, mean_b, 'go-', label='Mean with extra info')
    plt.plot(n_vals, std_b, 'g^:', label='Standard deviation with extra info')
    plt.xlabel('Length of x (n)')
    plt.ylabel('Fraction of bits of x recovered')
    plt.title("Recovering x from noisy running counter with and without extra information.")
    plt.legend(loc='best')
    plt.show()

if __name__ == "__main__":
    main()
