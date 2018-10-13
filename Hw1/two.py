import numpy as np
import sys
import random

def hamming_dist(x, x_hat, n):
	diffs = 0
	for i in range(n):
		if x[i] != x_hat[i]:
			diffs += 1
	return diffs

def frac_correct(x, x_hat, n):
	return 1 - float(hamming_dist(x, x_hat, n)) / float(n)


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
    n = int(sys.argv[1])
    trials = 20
    results_a = np.array([])
    results_b = np.array([])

    for i in range(trials):
	    x, _, a = generate_data(n)
	    w = generate_w(x, n)

	    x_hat_a = attack_a(a, n)
	    x_hat_b = attack_b(a, w, n)

	    results_a = np.append(results_a, frac_correct(x, x_hat_a, n))
	    results_b = np.append(results_b, frac_correct(x, x_hat_b, n))

    print "mean a: " + str(np.mean(results_a))
    print "mean b: " + str(np.mean(results_b))


if __name__ == "__main__":
    main()
    #test()