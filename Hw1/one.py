from common import hamming_dist
from common import frac_correct
from scipy.linalg import hadamard
from scipy.linalg import lstsq
from scipy.optimize import least_squares

import numpy as np
import sys


def attack_had(a, n):
	H = hadamard(n)
	z = H * a
	print "z: " + str(z)
	x_hat = np.rint(z)
	print "x_hat: " + str(x_hat)
	return x_hat


def attack_rand(a, B, n):
	z, _, _, _ = lstsq((1.0/float(n)) * B, a)
	print "z: " + str(z)
	x_hat = np.rint(z)
	print "x_hat: " + str(x_hat)
	return x_hat


def generate_had(n, theta):
	x = np.random.randint(2, size = (n, 1))
	print "x: " + str(x)
	H = hadamard(n)
	Y = np.random.normal(0, theta*theta, (n, 1))
	print "y: " + str(Y) 
	a = (1.0/float(n)) * H * np.matrix(x) + Y	
	print "a: " + str(a)
	return a

def generate_rand(n, m, theta):
	x = np.random.randint(2, size = (n, 1))
	print "x: " + str(x)
	B = np.random.randint(2, size = (m, n))
	print "B: " + str(B) 
	Y = np.random.normal(0, theta*theta, (m, 1))
	print "y: " + str(Y) 
	a = (1.0/float(n)) * B * np.matrix(x) + Y	
	print "a: " + str(a)
	return (a, B)



def main():
    n = int(sys.argv[1])
    theta = float(sys.argv[2])
    m = int(sys.argv[3])

    #a_had = generate_had(n, theta)
    #x_hat_had = attack_had(a, n)

    a_rand, B = generate_rand(n, m, theta)
    x_hat_rand = attack_rand(a_rand, B, n)



# n   |  m   |   theta   |  Had  |  Rand  
# ---------------------------------------
# 128. 1.1n.    2
#
# 512.   4n.    2^-1
#
# 2048. 16n.    2^-2
# 
# 8192.         2^-32n




if __name__ == "__main__":
    main()
    #test()

