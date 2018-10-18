# Compute number of bits at which x and x_hat differ.
def hamming_dist(x, x_hat, n):
	diffs = 0
	for i in range(n):
		if x[i] != x_hat[i]: # Increase counter if bits are different.
			diffs += 1
	return diffs

# Compute fraction of bits in which x and x_hat differ.
def frac_incorrect(x, x_hat, n):
	return float(hamming_dist(x, x_hat, n)) / float(n)

# Compute fraction of bits in which x and x_hat match.
def frac_correct(x, x_hat, n):
	return 1.0 - float(hamming_dist(x, x_hat, n)) / float(n)