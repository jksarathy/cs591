from common import hamming_dist
from common import frac_correct
import matplotlib.pyplot as plt
import numpy as np
import sys
import random


def attack_a(a, n):
    x_hat = [1] * n

    if a[0] == 0:
        x_hat[0] = 0
    elif a[0] == 2:
        x_hat[0] = 1

    for i in range(2, n):
        a_diff = a[i]-a[i-1] 
        if a_diff == 2:
            x_hat[i] = 1
        elif a_diff == -1:
            x_hat[i] = 0
    return x_hat


def attack_b(a, w, n):
    x_hat = [1] * n

    if a[0] == 0:
        x_hat[0] = 0
    elif a[0] == 2:
        x_hat[0] = 1
    else:
        x_hat[0] = w[0]

    for i in range(2, n):
        a_diff = a[i]-a[i-1] 
        if a_diff == 2:
            x_hat[i] = 1
        elif a_diff == -1:
            x_hat[i] = 0
        else:
            x_hat[i] = w[i]
    return x_hat


def generate_data(n):
    x = np.random.randint(2, size = n)
    exact_answers = np.cumsum(x) # This computes the array of prefix sums of x.
    z = np.random.randint(2, size = n)
    a = exact_answers + z
    return (x.tolist(), exact_answers.tolist(), a.tolist())


def generate_w(x, n):
    w = [1] * n
    for i in range(n):
        rand = random.randint(0, 2)
        if rand < 2:
            w[i] = x[i]
        else:
            w[i] = (x[i] + 1) % 2
    return w

def test():
    n = int(sys.argv[1])
    x, _, a = generate_data(n)
    w = generate_w(x, n)


def main():
    trials = 50
    mean_a = np.array([])
    mean_b = np.array([])
    std_a = np.array([])
    std_b = np.array([])

    n_vals = [100, 500, 1000, 5000]

    for n in n_vals:
        results_a = np.array([])
        results_b = np.array([])

        for i in range(trials):
            x, _, a = generate_data(n)
            w = generate_w(x, n)

            x_hat_a = attack_a(a, n)
            x_hat_b = attack_b(a, w, n)

            results_a = np.append(results_a, frac_correct(x, x_hat_a, n))
            results_b = np.append(results_b, frac_correct(x, x_hat_b, n))

        mean_a = np.append(mean_a, np.mean(results_a))
        mean_b = np.append(mean_b, np.mean(results_b))
        std_a = np.append(std_a, np.std(results_a))
        std_b = np.append(std_b, np.std(results_b))

        print "for n = " + str(n)
        print "results_a: " + str(results_a)
        print "mean a: " + str(np.mean(results_a))
        print "stddev a: " + str(np.std(results_a))

        print "results_b: " + str(results_b)
        print "mean b: " + str(np.mean(results_b))
        print "stddev b: " + str(np.std(results_b))

    plt.plot(n_vals, mean_a, 'bo-', label='Mean fraction recovered without extra info')
    plt.plot(n_vals, mean_b, 'go-', label='Mean fraction recovered with extra info')
    plt.plot(n_vals, std_a, 'b^:', label='Standard deviation of fraction recovered without extra info')
    plt.plot(n_vals, std_b, 'g^:', label='Standard deviation of fraction recovered with extra info')
    plt.xlabel('Length of x (n)')
    plt.ylabel('Fraction of bits of x recovered')
    plt.title("How well can we recover x from a noisy running counter?")
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    main()
    #test()