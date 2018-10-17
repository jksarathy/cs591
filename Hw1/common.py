def hamming_dist(x, x_hat, n):
	diffs = 0
	for i in range(n):
		if x[i] != x_hat[i]:
			diffs += 1
	return diffs

def frac_incorrect(x, x_hat, n):
	return float(hamming_dist(x, x_hat, n)) / float(n)

def frac_correct(x, x_hat, n):
	return 1.0 - float(hamming_dist(x, x_hat, n)) / float(n)