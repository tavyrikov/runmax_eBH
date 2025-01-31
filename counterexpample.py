import numpy as np

# Number of Monte Carlo simulations
M = 1_000_000  

# Number of time steps
T = 100       

# Generate uniform random values for the probability of each (e1, e2) pair
p = np.random.rand(M, T)

# Simulating the e-processes
# We are not simulating X_0 explicitly, because it only affects the scaling
# Instead, we simulate e_t directly, making the simulation independent of alpha.
e1 = np.where(
    p <= 1.0/3.0, -1,  # Case (1/2, 1/2) -> log(1/2) ≈ -1 in log domain
    np.where(p <= 2.0/3.0, -1, 1)  # Case (2, 1/2) and (1/2, 2)
)

e2 = np.where(
    p <= 1.0/3.0, -1,  # Case (1/2, 1/2) -> log(1/2) ≈ -1 in log domain
    np.where(p <= 2.0/3.0, 1, -1)  # Case (2, 1/2) and (1/2, 2)
)

# Compute the running maximum of cumulative sums for each process
max_cumsum_e1 = np.max(np.cumsum(e1, axis=1), axis=1)
max_cumsum_e2 = np.max(np.cumsum(e2, axis=1), axis=1)

# Compute the probability of rejecting at least one hypothesis
# The rejection event occurs when:
# - Either e1 or e2 exceeds 2 (corresponding to 2/alpha in the original scaling)
# - Or both exceed 1 (corresponding to 1/alpha in the original scaling)
rejections = np.where(
    (max_cumsum_e1 >= 2) | (max_cumsum_e2 >= 2) |
    ((max_cumsum_e1 >= 1) & (max_cumsum_e2 >= 1)), 
     1.0, 0.0
)

rejection_prob = np.mean(rejections)

# The factor 2 accounts for the probability mass of 2α in the original model
estimated_fdr = 2 * rejection_prob

# Compute the standard error of the Monte Carlo estimate
# Since we estimate a probability, the standard error is sqrt(p(1-p)/M)
standard_error = 2 * np.sqrt(rejection_prob * (1 - rejection_prob) / M)

# Print results
print(f"Estimated FDR: {estimated_fdr:.4f} ± {standard_error:.4f} (SE)")