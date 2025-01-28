import numpy as np

M = 1_000_000  
T = 100       

p = np.random.rand(M, T)

e1 = np.where(
    p <= 1.0/3.0, -1,
    np.where(p <= 2.0/3.0, -1, 1)
)

e2 = np.where(
    p <= 1.0/3.0, -1,
    np.where(p <= 2.0/3.0, 1, -1)
)


max_cumsum_e1 = np.amax(np.cumsum(e1, axis=1), axis=1)
max_cumsum_e2 = np.amax(np.cumsum(e2, axis=1), axis=1)

result = np.mean(
    np.where(
        (max_cumsum_e1 >= 2) | (max_cumsum_e2 >= 2) |
        ((max_cumsum_e1 >= 1) & (max_cumsum_e2 >= 1)),
        1, 0
    )
)

print("Result:", 2*result)