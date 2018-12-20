import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.linalg import lstsq
import numpy as np
import math
import sys

true_corr = 0.3
true_beta = 0.2
coverage = .95
eps = 0.5
delta = 0.005
def_alpha = 0.4
min_slope = -50
max_slope = 50

# ==== Median ===========
# Computes eps-differentially private median for bounded data.
def dpMedian(x, lower_bound=min_slope, upper_bound=max_slope, epsilon=eps):
    """
    :param x: List of real numbers
    :param lower_bound: Lower bound on values in x 
    :param upper_bound: Upper bound on values in x 
    :return: eps-DP approximate median
    """
    # First, sort the values
    x.sort()
    n = len(x)
    # Bookend x with lower bound and upper bound
    # x[0] = lower_bound, x[n+1] = upper_bound
    x.insert(0,lower_bound) 
    x.append(upper_bound) 

    # Iterate through x, assigning scores to each interval given by adjacent indices
    # currentMax and currentInt keep track of highest score and corresponding interval
    currentMax = float("-inf")
    currentInt = -1
    for i in xrange(1, n+1):
        start = x[i-1]
        end = x[i]
        # Compute length of interval on logarithmic scale
        length = end-start
        loglength = float("-inf") if (length <= 0) else math.log(length)
        # The score has two components:
        # (1) Distance from index to median (closer -> higher score)
        # (2) Length of the interval on a logarithmic scale (larger -> higher score)
        score = -abs((i - 1/2) - (n+1)/2) + loglength
        # Add noise scaled to global sensitivity using exponential mechanism 
        noisyscore = score + float(np.random.exponential())/float(epsilon)
        if (noisyscore > currentMax):
            currentInt = i
            currentMax = noisyscore
    # Select uniformly from the highest scoring interval given by currentInt
    return np.random.uniform(low=x[currentInt-1], high=x[currentInt])

# ==== Trimmed Mean ===========
# Computes alpha-interquantile range.
def IQRalpha(x, alpha):
    """
    :param x: List of real numbers
    :param alpha: Proportion of values to disregard 
    :return: alpha-interquantile range, upper index, lower index
    """
    n = len(x)
    upper_ind = int(n*(1.0-(alpha/2.0)))
    lower_ind = int(n*(alpha/2.0))
    return x[upper_ind] - x[lower_ind], upper_ind, lower_ind

# Computes distance of data from highly sensitive data.
def distFromHighSensitivity(x, alpha, disc):
    """
    :param x: List of real numbers
    :param alpha: Proportion of values to disregard 
    :param disc: 1 or 2, according to discretization
    :return: Number of data points that would need to be changed to make x highly sensitive
    """
    n = len(x)
    x = [x[i]*100 for i in range(n)]
    x.sort()

    iqr, upper_ind, lower_ind = IQRalpha(x, alpha)
    base = 2 #1 + (1.0 / np.log(n))
    k1 = float("-inf") if iqr <= 0 else math.floor(math.log(iqr, base))

    for i in range(n-upper_ind):
        upper_int = x[upper_ind + i] - x[upper_ind]
        lower_int = x[lower_ind] - x[lower_ind - i]
        if upper_int >= 2**(k1 + disc) or lower_int >= 2**(k1 + disc):
            break
    return i

# Computes eps-differentially private approximation of inter-alpha range.
def dpScale(x, alpha, epsilon=eps):
    """
    :param x: List of real numbers
    :param alpha: Proportion of values to disregard 
    :param epsilon: Privacy loss parameter
    :return: eps-differentially approximation of inter-alpha range
    """
    s = [None, None]
    ret = float("-inf")
    for disc in range(2):
        z = np.random.laplace(loc=0.0, scale=float(1)/float(eps))
        r = distFromHighSensitivity(x, alpha, disc) + z
        if r <= (1 + np.log(1.0/delta)):
            s[disc] = None
        else: 
            iqr, _, _ = IQRalpha(x, alpha)
            y = np.random.laplace(loc=0.0, scale=float(1)/float(eps))
            s[disc] = iqr * 2**y
    if s[0] != None:
        ret = s[0]
    else:
        ret = s[1]
    return ret

# Compute non-private alpha-trimmed mean.
def trimmedMean(x, alpha):
    """
    :param x: List of real numbers
    :param alpha: Proportion of values to disregard 
    :return: alpha-trimmed mean
    """
    n = len(x)
    upper_ind = int(n*(1.0-(alpha/2.0)))
    lower_ind = int(n*(alpha/2.0)) 
    nIn = upper_ind - lower_ind 

    x.sort()
    sum = 0.0
    for i in range(nIn):
        elem = x[lower_ind + i]
        sum += elem
    trimmed_mean = float(sum) / float(nIn)
    return trimmed_mean

# Computes eps-differentially private approximation of alpha-trimmed mean.
def dpTrimmedMean(x, alpha, dataset_size, epsilon=eps):
    """
    :param x: List of real numbers
    :param alpha: Proportion of values to disregard 
    :param epsilon: Privacy loss parameter
    :param dataset_size: Size of underlying dataset
    :return: eps-differentially approximation of alpha-trimmed mean
    """
    n = len(x)
    z = np.random.laplace(loc=0.0, scale=1.0)
    k = 0
    # Compute differentially-private alpha-interquantile range
    s = dpScale(x, alpha, epsilon)
    # If algorithm does not return a value, use full range
    if s == None:
        s = max_slope - min_slope
    noise = float(s* (n**k) * z) / float((1-alpha)*n - 2)
    # Add noise to non-private trimmed mean
    noisy_mean = trimmedMean(x, alpha) + noise
    return noisy_mean

# ==== Generate Data ===========
# Truncates data to be within [0,1]
def boundData(x, y):
    """
    :param x: List of real numbers
    :param y: List of real numbers
    :return: alpha-trimmed mean
    """
    for i in range(len(x)):
        x[i] = max(x[i], 0) if x[i] < 0.5 else min(x[i], 1)
        y[i] = max(y[i], 0) if y[i] < 0.5 else min(y[i], 1)
    return x, y

# Computes errors according to specified distribution
def genErrors(dataset_size, base_scale=0.01, het=False, het_arr=None):
    """
    :param dataset_size: Size of error array to be returned
    :param loc: Center of error distribution
    :param base_scale: Scale of error distribution
    :param het_arr: List of real numbers for computing heteroskedastic error
    :return: List of error values
    """
    errors = [0]*dataset_size
    if het == False or het_arr == None:
        # Scale for each error is fixed at five times the base scale
        errors = st.norm.rvs(scale=base_scale*10, size=dataset_size)
    else:
        for i in range(dataset_size):
            # Standard deviation of each error correlated with x value
            errors[i] = st.norm.rvs(scale=base_scale*het_arr[i], size=1)[0]
    return errors

# Puts outliers into dataset
def genOutliers(num_outliers, x, y, base_scale=0.1):
    x = np.append(x, st.norm.rvs(loc=0.2, scale=base_scale, size=num_outliers))
    y = np.append(y, st.norm.rvs(loc=0.7, scale=base_scale, size=num_outliers))
    print y
    return x, y

# Generates dataset of specified size and distribution
def genData(dataset_size, prop_outliers=0.0, het=False):
    """
    :param dataset_size: Number of (x,y) points to be returned
    :param prop_outliers: Proportion of outliers in dataset
    :param het: Boolean value determining heteroskedasticity
    :return: x points, y points
    """
    num_outliers = int(dataset_size * prop_outliers)
    print num_outliers
    num_points = dataset_size - num_outliers
    x = st.norm.rvs(loc=0.5, scale=0.05, size=num_points)
    errors = genErrors(num_points, het=het, het_arr=x)
    y = [0]*num_points
    for i in range(num_points):
        y[i] = true_corr*x[i]+true_beta+errors[i]
    x, y = genOutliers(num_outliers, x, y)
    x, y = boundData(x, y)
    return np.array(x), np.array(y)

# Compute size of smallest interval that captures true value
# for given coverage probability
def computeIntervalSize(stats, coverage, trials):
    """
    :param stats: List of computed statistics
    :param coverage: Coverage probability (ie. 95 for 95% CI)
    :param trials: Number of statistics computed
    :return: Size of smallest interval
    """
    # Compute absolute value of differences of statistic from true value
    diffs = [abs(stats[i]-true_corr) for i in range(trials)]
    # Sort differences
    diffs.sort()
    index = int(trials * coverage) - 1
    return diffs[index]

# Generate data, run algorithms, and return confidence interval sizes
# for a given sample size
def runTrials(stat_trials, dataset_size, prop_outliers=0.0, het=False):
    """
    :param stat_trials: Number of trials for computing statistic
    :param dataset_size: Number of points in data set
    :param prop_outliers: Proportion of outliers in data set
    :param het: Boolean value determining heteroskedasticity
    :return: Size of smallest interval for each algorithm
    """
    ols_coeffs = [0]*stat_trials
    dpols_coeffs = [0]*stat_trials
    ts_coeffs = [0]*stat_trials
    ts_mean_coeffs = [0]*stat_trials
    dpmed_ts_coeffs = [0]*stat_trials
    dpmean_ts_coeffs = [0]*stat_trials

    for i in range(stat_trials):
        x, y = genData(dataset_size, prop_outliers=prop_outliers, het=het)
        M = x[:, np.newaxis]**[0, 1]

        #ols
        ols, _, _, _ = lstsq(M, y)
        ols_coeffs[i] = ols[1]
        ols_ci = computeIntervalSize(ols_coeffs, coverage, stat_trials)

        #dpols
        noise_scale = float(2)/float(dataset_size*eps)
        dpcov = np.cov(x, y)[0][1] + np.random.laplace(loc=0.0, scale=noise_scale)
        dpvar = np.var(x) + np.random.laplace(loc=0.0, scale=noise_scale)
        dpalpha = float(dpcov)/float(dpvar)
        dpols_coeffs[i] = dpalpha
        dpols_ci = computeIntervalSize(dpols_coeffs, coverage, stat_trials)

        #ts
        tsslope, tsint, _, _ = st.mstats.theilslopes(y, x)
        ts_coeffs[i] = tsslope
        ts_ci = computeIntervalSize(ts_coeffs, coverage, stat_trials)

        #ts slopes
        slopes = []
        for p in range(len(x)):
            for q in range(p+1, len(x)):
                x_delta = float(x[q]-x[p])
                if x_delta == 0:
                    x_delta = 0.0001
                slope = float(y[q]-y[p])/ float(x_delta)
                if slope > min_slope and slope < max_slope: # make sure slopes are bounded
                    slopes.append(slope)

        #dpmed_ts
        epsilon = float(eps)/float(dataset_size)
        dpmed_ts_coeffs[i] = dpMedian(slopes, epsilon=epsilon)
        dpmed_ts_ci = computeIntervalSize(dpmed_ts_coeffs, coverage, stat_trials)

        #ts_mean
        ts_mean_coeffs[i] = trimmedMean(x, def_alpha)
        ts_mean_ci = computeIntervalSize(ts_mean_coeffs, coverage, stat_trials)

        #dpmean_ts
        dpmean_ts_coeffs[i] = dpTrimmedMean(x, def_alpha, dataset_size, epsilon=epsilon)
        dpmean_ts_ci = computeIntervalSize(dpmean_ts_coeffs, coverage, stat_trials)

        # #testing
        # if i == 1: 
        #     plotEstimators(x, y, ols, [tsint, dpmed_ts_coeffs[i]])

    return ols_ci, dpols_ci, ts_ci, dpmed_ts_ci, ts_mean_ci, dpmean_ts_ci

# Plot interval sizes against varying sample sizes for all algorithms
def plotTrials(x, y1, y2, y3, y4, y5, y6):
    """
    :param x: List of sample sizes
    :param y1: Non-private OLS interval sizes
    :param y2: DP OLS interval sizes
    :param y3: Non-private T-S interval sizes
    :param y4: DP T-S median interval sizes
    :param y5: Non-private T-S trimmed-mean interval sizes
    :param y6: DP T-S trimmed mean interval sizes
    """
    fig, ax = plt.subplots()
    ax.plot(x, y1, 'bo-', label='Non-private OLS')
    ax.plot(x, y2, 'b*:', label='DP OLS')
    ax.plot(x, y3, 'go-', label='Non-private T-S')
    ax.plot(x, y4, 'g*:', label='DP T-S (median)')
    ax.plot(x, y5, 'ro-', label='Non-private T-S (trimmed mean)')
    ax.plot(x, y6, 'r*:', label='DP T-S (trimmed mean)')
    #plt.legend()
    plt.xlabel('Sample Size')
    plt.ylabel('95% Confidence Interval Size')
    plt.yscale("log")
    plt.show()

# Run trials and plot results
def main():
    dataset_sizes = [10, 50, 100, 500]
    ols_intervals = [0]*len(dataset_sizes)
    dpols_intervals = [0]*len(dataset_sizes)
    ts_intervals = [0]*len(dataset_sizes)
    dpmed_ts_intervals = [0]*len(dataset_sizes)
    ts_mean_intervals = [0]*len(dataset_sizes)
    dpmean_ts_intervals = [0]*len(dataset_sizes)
    stat_trials = 100
    ci_trials = 3

    for i in range(len(dataset_sizes)):
        for j in range(ci_trials):
            ols_ci, dpols_ci, ts_ci, dpmed_ts_ci, ts_mean_ci, dpmean_ts_ci = runTrials(stat_trials, dataset_sizes[i], prop_outliers=float(sys.argv[1]), het=bool(sys.argv[2]))
            ols_intervals[i] += ols_ci
            dpols_intervals[i] += dpols_ci
            ts_intervals[i] += ts_ci
            dpmed_ts_intervals[i] += dpmed_ts_ci
            ts_mean_intervals[i] += ts_mean_ci
            dpmean_ts_intervals[i] += dpmean_ts_ci
        ols_intervals[i] = float(ols_intervals[i])/float(ci_trials)
        dpols_intervals[i] = float(dpols_intervals[i])/float(ci_trials)
        ts_intervals[i] = float(ts_intervals[i])/float(ci_trials)
        dpmed_ts_intervals[i] = float(dpmed_ts_intervals[i])/float(ci_trials)
        ts_mean_intervals[i] = float(ts_mean_intervals[i])/float(ci_trials)
        dpmean_ts_intervals[i] = float(dpmean_ts_intervals[i])/float(ci_trials)

    plotTrials(np.array(dataset_sizes), np.array(ols_intervals), np.array(dpols_intervals), np.array(ts_intervals), np.array(dpmed_ts_intervals), np.array(ts_mean_intervals), np.array(dpmean_ts_intervals))

main()

#testStats(100)
# ---- Just for testing ----
# def plotEstimator(x, y, p):
#     plt.plot(x, y, 'o', label='data')
#     xx = np.linspace(0, 1, 101)
#     yy = p[0] + p[1]*xx
#     plt.plot(xx, yy, label='least squares fit, $y = a + bx$')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend(framealpha=1, shadow=True)
#     plt.grid(alpha=0.25)
#     plt.show()

# def plotEstimators(x, y, p, p2):
#     plt.plot(x, y, 'o', label='data')
#     xx = np.linspace(0, 1, 101)
#     yy = p[0] + p[1]*xx
#     plt.plot(xx, yy, 'rs', label='ols fit')
#     yy2 = p[0] + p2[1]*xx
#     plt.plot(xx, yy2, 'g*:', label='ts fit')
#     plt.xlabel('x')
#     plt.ylabel('y')
#     plt.legend(framealpha=1, shadow=True)
#     plt.grid(alpha=0.25)
#     plt.show()

# def testStats(dataset_size):
#     x, y = genHetData(dataset_size)
#     M = x[:, np.newaxis]**[0, 1]

#     #ols
#     ols, _, _, _ = lstsq(M, y)

#     #ts
#     #tsslope, tsint, _, _ = theilslopes(y, x)

#     slopes = []
#     ints = []
#     for p in range(len(x)):
#         for q in range(p+1, len(x)):
#             slope = float(y[q]-y[p])/float(x[q]-x[p])
#             slopes.append(slope)
#             ints.append(y[q] - slope*x[q])
#     tsslope = np.median(slopes)

#     plotEstimators(x, y, ols, [0.2, tsslope]
