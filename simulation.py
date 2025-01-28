import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt

from scipy.stats import random_correlation

plt.style.use(['ggplot'])
plt.rcParams["figure.figsize"] = (20, 10)
plt.rcParams["figure.dpi"] = 150

alpha = 0.05
ntrue=100
nfalse=100
n = ntrue+nfalse

M = 10000
T = 2000

import multiprocessing

def unpacking_apply_along_axis(all_args):
    """…"""
    (func1d, axis, arr, args, kwargs) = all_args
    return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)

def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
    """
    Like numpy.apply_along_axis(), but takes advantage of multiple
    cores.
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
              for sub_arr in np.array_split(arr, 5)]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)

for simi in range(1):
    print("Iteration ", simi)

    
    rng = np.random.default_rng()
    ev = np.random.exponential(size = n)
    ev = n * ev / np.sum(ev)
    sigma = random_correlation.rvs(ev, random_state=rng)

    mean_true = 0.0
    mean_false = 0.1
    marg_sigma = np.mean(np.diag(sigma))
    print('mean_true ', mean_true)
    print('mean_false ', mean_false)
    print('marg_sigma ', marg_sigma)

    true_flag = np.array([1]*ntrue+[0]*nfalse)
    
    def simulate(i):
        np.random.seed(2*T*i+72346)
        
        X = np.random.multivariate_normal(mean=[mean_true]*ntrue+[mean_false]*nfalse, cov=sigma, size=T).T
        E = scipy.stats.norm(mean_false, marg_sigma).pdf(X) / scipy.stats.norm(mean_true, marg_sigma).pdf(X)
        Ep = np.cumprod(E, axis = 1)

        if i == 0:
            print('mean E', np.mean(E[:ntrue, :]))

        def BH(E, true_flag):
            true_flag = true_flag.copy()
            true_flag = true_flag[np.argsort(E)[::-1]]
            E = E[np.argsort(E)[::-1]]
            rejected = np.max(np.where(E >= n/((1.0+np.arange(n)) * alpha), 1+np.arange(n), 0))

            if rejected == 0:
                fdr = 0
            else:
                fdr = np.sum(true_flag[:rejected])*1.0 / rejected

            if rejected == 0:
                power = 0
            else:
                power = np.sum(1 - true_flag[:rejected])*1.0 / np.sum(1 - true_flag)

            return fdr, power, rejected

        cmaxEp = np.maximum.accumulate(Ep, axis = 1)
        
        fdr1, fdr2, fdr3, fdr4  = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
        power1, power2, power3, power4 = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
        rejected1, rejected2, rejected3, rejected4 = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
        globe1, globe2, globe3, globe4 = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)

        prev_record = n
        for t in range(T):
            max_i = np.argmax(E[:, t])
            if E[max_i, t] > n / (prev_record * alpha):
                E[max_i, t:] = E[max_i, t]
                prev_record -= 1

            res = BH(E[:, t], true_flag)
            
            fdr1[t] = res[0]
            power1[t] = res[1]
            rejected1[t] = res[2]
            if np.sum(E[:, t]) >= n/alpha:
                globe1[t] = 1 
            else:
                globe1[t] = 0 
            
        
            res = BH(cmaxEp[:, t], true_flag)
            fdr2[t] = res[0]
            power2[t] = res[1]
            rejected2[t] = res[2]
            if np.sum(cmaxEp[:, t]) >= n/alpha:
                globe2[t] = 1 
            else:
                globe2[t] = 0
            
            Epmax = cmaxEp[:, t].copy()
            
            normEpmax1 = (Epmax - 1.0 - np.log(Epmax)) / (np.log(Epmax)*np.log(Epmax))
            res =  BH(normEpmax1, true_flag)
            fdr3[t] = res[0]
            power3[t] = res[1]
            rejected3[t] = res[2]
            if np.sum(normEpmax1) >= n/alpha:
                globe3[t] = 1 
            else:
                globe3[t] = 0
                
            normEpmax2 = np.sqrt(Epmax) - 1
            res =  BH(normEpmax2, true_flag)
            fdr4[t] = res[0]
            power4[t] = res[1]
            rejected4[t] = res[2]
            if np.sum(normEpmax2) >= n/alpha:
                globe4[t] = 1 
            else:
                globe4[t] = 0
                    
        return fdr1, fdr2, fdr3, fdr4, power1, power2, power3, power4, rejected1, rejected2, rejected3, rejected4, globe1, globe2, globe3, globe4

        
    pool = multiprocessing.Pool(20)
    fdr1, fdr2, fdr3, fdr4, power1, power2, power3, power4, rejected1, rejected2, rejected3, rejected4, globe1, globe2, globe3, globe4 = zip(*pool.map(simulate, range(0, M)))

    fdr1 = np.vstack(fdr1)
    supfdr1 = np.maximum.accumulate(fdr1, axis=1)
    power1 = np.vstack(power1)
    rejected1 = np.vstack(rejected1)
    globe1 = np.vstack(globe1)
       
    fdr1 = np.mean(fdr1, axis=0)
    supfdr1 = np.mean(supfdr1, axis=0)
    power1 = np.mean(power1, axis=0)
    rejected1 = np.mean(rejected1, axis=0)
    globe1 = np.mean(globe1, axis=0)
    
    
    fdr2 = np.vstack(fdr2)
    supfdr2 = np.maximum.accumulate(fdr2, axis=1)
    power2 = np.vstack(power2)
    rejected2 = np.vstack(rejected2)
    globe2 = np.vstack(globe2)
       
    fdr2 = np.mean(fdr2, axis=0)
    supfdr2 = np.mean(supfdr2, axis=0)
    power2 = np.mean(power2, axis=0)
    rejected2 = np.mean(rejected2, axis=0)
    globe2 = np.mean(globe2, axis=0)
    
    
    fdr3 = np.vstack(fdr3)
    supfdr3 = np.maximum.accumulate(fdr3, axis=1)
    power3 = np.vstack(power3)
    rejected3 = np.vstack(rejected3)
    globe3 = np.vstack(globe3)
       
    fdr3 = np.mean(fdr3, axis=0)
    supfdr3 = np.mean(supfdr3, axis=0)
    power3 = np.mean(power3, axis=0)
    rejected3 = np.mean(rejected3, axis=0)
    globe3 = np.mean(globe3, axis=0)
    
    
    fdr4 = np.vstack(fdr4)
    supfdr4 = np.maximum.accumulate(fdr4, axis=1)
    power4 = np.vstack(power4)
    rejected4 = np.vstack(rejected4)
    globe4 = np.vstack(globe4)
       
    fdr4 = np.mean(fdr4, axis=0)
    supfdr4 = np.mean(supfdr4, axis=0)
    power4 = np.mean(power4, axis=0)
    rejected4 = np.mean(rejected4, axis=0)
    globe4 = np.mean(globe4, axis=0)

    plt.rcParams["figure.figsize"] = (20,10)
    fig, axes = plt.subplots(2, 3, constrained_layout=True)
    
    axes[0,0].axhline(y=alpha, linestyle='-', color='g')
    
    axes[0,0].plot(np.arange(T), fdr1, label='eBH')
    axes[0,0].plot(np.arange(T), fdr2, label='cumulative maximum eBH')
    axes[0,0].plot(np.arange(T), fdr3, label='adjusted(1) running maximum eBH')
    axes[0,0].plot(np.arange(T), fdr4, label='adjusted(2) running maximum eBH')
        
    axes[0,0].set_title('FDR', fontsize=15)
    axes[0,0].set_xlabel('Time', fontsize=15)
    axes[0,0].set_ylabel('FDR', fontsize=15) 
    axes[0,0].tick_params(axis='both', which='major', labelsize=10)
    axes[0,0].tick_params(axis='both', which='minor', labelsize=8)
    
    axes[0,1].axhline(y=alpha, linestyle='-', color='g')
    
    axes[0,1].plot(np.arange(T), supfdr1, label='eBH')
    axes[0,1].plot(np.arange(T), supfdr2, label='cumulative maximum eBH')
    axes[0,1].plot(np.arange(T), supfdr3, label='adjusted(1) running maximum eBH')
    axes[0,1].plot(np.arange(T), supfdr4, label='adjusted(2) running maximum eBH')
        
    axes[0,1].set_title('supFDR', fontsize=15)
    axes[0,1].set_xlabel('Time', fontsize=15)
    axes[0,1].set_ylabel('supFDR', fontsize=15) 
    axes[0,1].tick_params(axis='both', which='major', labelsize=10)
    axes[0,1].tick_params(axis='both', which='minor', labelsize=8)
    
    axes[0,2].plot(np.arange(T), globe1, label='eBH')
    axes[0,2].plot(np.arange(T), globe2, label='cumulative maximum eBH')
    axes[0,2].plot(np.arange(T), globe3, label='adjusted(1) running maximum eBH')
    axes[0,2].plot(np.arange(T), globe4, label='adjusted(2) running maximum eBH')
        
    axes[0,2].set_title('Global Error Type 1', fontsize=15)
    axes[0,2].set_xlabel('Time', fontsize=15)
    axes[0,2].set_ylabel('Global Error Type 1', fontsize=15) 
    axes[0,2].tick_params(axis='both', which='major', labelsize=10)
    axes[0,2].tick_params(axis='both', which='minor', labelsize=8)
        
    axes[1,0].plot(np.arange(T), power1, label='eBH')
    axes[1,0].plot(np.arange(T), power2, label='cumulative maximum eBH')
    axes[1,0].plot(np.arange(T), power3, label='adjusted(1) running maximum eBH')
    axes[1,0].plot(np.arange(T), power4, label='adjusted(2) running maximum eBH')
        
    axes[1,0].set_title('Power', fontsize=15)
    axes[1,0].set_xlabel('Time', fontsize=15)
    axes[1,0].set_ylabel('Power', fontsize=15) 
    axes[1,0].tick_params(axis='both', which='major', labelsize=10)
    axes[1,0].tick_params(axis='both', which='minor', labelsize=8)
        
    axes[1,1].plot(np.arange(T), rejected1, label='eBH')
    axes[1,1].plot(np.arange(T), rejected2, label='cumulative maximum eBH')
    axes[1,1].plot(np.arange(T), rejected3, label='adjusted(1) running maximum eBH')
    axes[1,1].plot(np.arange(T), rejected4, label='adjusted(2) running maximum eBH')
        
    axes[1,1].set_title('Rejected', fontsize=15)
    axes[1,1].set_xlabel('Time', fontsize=15)
    axes[1,1].set_ylabel('Rejected', fontsize=15) 
    axes[1,1].tick_params(axis='both', which='major', labelsize=10)
    axes[1,1].tick_params(axis='both', which='minor', labelsize=8)

    lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes[:1]]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]

    axes[1,2].remove()

    fig.legend(lines, labels, loc='lower right', ncol=1, fontsize=15, bbox_to_anchor=(0.99, 0.3))

    plt.savefig(f'sim2.png')
    plt.show()