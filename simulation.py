import numpy as np
import scipy.stats as stats
import multiprocessing
import matplotlib.pyplot as plt
from scipy.stats import random_correlation

# Set matplotlib style
plt.style.use(['ggplot'])

# Simulation parameters
ALPHA = 0.05  # False Discovery Rate threshold
N_TRUE = 100   # Number of true null hypotheses
N_FALSE = 100  # Number of false null hypotheses
N = N_TRUE + N_FALSE  # Total number of hypotheses
M = 100000  # Number of simulations
T = 2000  # Number of time steps


def generate_correlation_matrix(n):
    """
    Generate a random correlation matrix using an exponential distribution.
    """
    rng = np.random.default_rng()
    ev = np.random.exponential(size=n)
    ev = n * ev / np.sum(ev)
    return random_correlation.rvs(ev, random_state=rng)


def bayes_factor(x, mean_true, mean_false, sigma):
    """
    Compute the Bayes Factor for sequential hypothesis testing.
    """
    return stats.norm(mean_false, sigma).pdf(x) / stats.norm(mean_true, sigma).pdf(x)


def ebh_procedure(e_values, true_flags):
    """
    Apply the Benjamini-Hochberg based on e-values (eBH) procedure to control FDR.
    """
    sorted_indices = np.argsort(e_values)[::-1]
    sorted_e_values = e_values[sorted_indices]
    sorted_true_flags = true_flags[sorted_indices]
    
    rejected = np.max(np.where(sorted_e_values >= N / ((1.0 + np.arange(N)) * ALPHA), 1 + np.arange(N), 0))
    
    fdr = 0 if rejected == 0 else np.sum(sorted_true_flags[:rejected]) / rejected
    power = 0 if rejected == 0 else np.sum(1 - sorted_true_flags[:rejected]) / np.sum(1 - sorted_true_flags)
    
    return fdr, power, rejected


def simulate_experiment(i, mean_true, mean_false, sigma, marg_sigma, true_flags):
    """
    Run a single simulation of sequential hypothesis testing with four different strategies.
    """
    np.random.seed(2 * T * i + 72346)
    
    # Generate correlated normal data
    X = np.random.multivariate_normal(mean=[mean_true] * N_TRUE + [mean_false] * N_FALSE, cov=sigma, size=T).T
    
    # Compute e-values using the Bayes Factor
    E = bayes_factor(X, mean_true, mean_false, marg_sigma)
    E_cumulative = np.cumprod(E, axis=1)
    cmax_Ep = np.maximum.accumulate(E_cumulative, axis=1)
    
    # Initialize result arrays for four different testing strategies
    fdr_results, power_results, rejected_results = np.zeros((4, T)), np.zeros((4, T)), np.zeros((4, T))
    
    for t in range(T):
        fdr_results[0, t], power_results[0, t], rejected_results[0, t] = ebh_procedure(E[:, t], true_flags)
        fdr_results[1, t], power_results[1, t], rejected_results[1, t] = ebh_procedure(cmax_Ep[:, t], true_flags)
        normEpmax1 = (cmax_Ep[:, t] - 1.0 - np.log(cmax_Ep[:, t])) / (np.log(cmax_Ep[:, t])**2)
        fdr_results[2, t], power_results[2, t], rejected_results[2, t] = ebh_procedure(normEpmax1, true_flags)
        normEpmax2 = np.sqrt(cmax_Ep[:, t]) - 1
        fdr_results[3, t], power_results[3, t], rejected_results[3, t] = ebh_procedure(normEpmax2, true_flags)
    
    return fdr_results, power_results, rejected_results


def run_simulations():
    """
    Run multiple simulations in parallel using multiprocessing.
    """
    sigma = generate_correlation_matrix(N)
    marg_sigma = np.mean(np.diag(sigma))
    true_flags = np.array([1] * N_TRUE + [0] * N_FALSE)
    
    # Run simulations in parallel
    with multiprocessing.Pool(20) as pool:
        results = pool.starmap(simulate_experiment, [(i, 0.0, 0.1, sigma, marg_sigma, true_flags) for i in range(M)])
    
    # Aggregate result
    
    fdr_values, power_values, rejected_values = zip(*results)
    
        
    fdr_mean = np.mean(np.stack(fdr_values), axis=0)
    supfdr_mean = np.mean(np.maximum.accumulate(np.stack(fdr_values), axis=2), axis=0)
    power_mean = np.mean(np.stack(power_values), axis=0)
    rejected_mean = np.mean(np.stack(rejected_values), axis=0)
    
    return fdr_mean, supfdr_mean, power_mean, rejected_mean


def plot_results(fdr, supfdr, power, rejected):
    """
    Plot the simulation results for the four different strategies.
    """
    labels = ['eBH', 'Running Maximum eBH', 'Adjusted (1) Running Maximum eBH', 'Adjusted (2) Running Maximum eBH']
    # colors = ['r', 'b', 'g', 'purple']
    
    plt.figure(figsize=(12, 6))
    
    for i in range(4):
        plt.subplot(1, 3, 1)
        plt.axhline(y=ALPHA, linestyle='-', color='g')
        plt.plot(fdr[i], label=labels[i])#, color=colors[i])
        plt.title('False Discovery Rate')
        plt.xlabel('Time')
        plt.ylabel('FDR')
        plt.legend(bbox_to_anchor=(1.01, 0.85))
    
        plt.subplot(1, 3, 2)
        plt.plot(supfdr[i], label=labels[i])#, color=colors[i])
        plt.title('supFDR Over Time')
        plt.xlabel('Time')
        plt.ylabel('Rejected')
        # plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(power[i], label=labels[i])#, color=colors[i])
        plt.title('Power Over Time')
        plt.xlabel('Time')
        plt.ylabel('Power')
        # plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'sim4.png')
    plt.show()

# Run the simulation
fdr_results, supfd_results, power_results, rejected_results = run_simulations()

# Plot results
plot_results(fdr_results, supfd_results, power_results, rejected_results)