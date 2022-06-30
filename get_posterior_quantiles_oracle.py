import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import torch

# For parallelizing
import multiprocessing
import joblib

from scipy.interpolate import interp2d
from scipy.stats import beta, multivariate_normal, nct, t

from conformal_utils import split_X_and_y

np.random.seed(0)

## 1) Setup 

## 1a

# Load data 
softmax_scores = torch.load('/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/logits/imagenet_train_subset_softmax.pt', map_location=torch.device('cpu'))
softmax_scores = softmax_scores.numpy()
labels = torch.load('/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/logits/imagenet_train_subset_labels.pt', map_location=torch.device('cpu'))
labels = labels.type(torch.LongTensor).numpy()

# Select subset of data

n = 20 # Number of points per class k used to fit Beta distributions
n_tune = 10 # Number of points per class k used to perform conformal adjustment 
num_classes = 1000

num_samples = 100000 # Number of Monte Carlo samples 

# Split into calibration and validation datasets, then further break down calibration set
calib_scores, calib_labels, _, _ = split_X_and_y(softmax_scores, labels, n + n_tune, num_classes=1000, seed=0)
softmax_scores_subset, labels_subset, _, _ = split_X_and_y(calib_scores, calib_labels, n, num_classes=1000, seed=0)

# softmax_scores_subset = np.zeros((num_classes * n, num_classes))
# labels_subset = np.zeros((num_classes * n, ), dtype=np.int32)

# for k in range(num_classes):
    
#     # Randomly select n instances of class k
#     idx = np.argwhere(labels==k).flatten()
#     selected_idx = np.random.choice(idx, replace=False, size=(n,))
    
#     softmax_scores_subset[n*k:n*(k+1), :] = softmax_scores[selected_idx, :]
#     labels_subset[(n*k):(n*(k+1))] = k
    
# Only select data for which k is true class
scores_subset = 1 - np.array([softmax_scores_subset[row,labels_subset[row]] for row in range(len(labels_subset))])

# Load KDE estimate of prior
with open('.cache/kde.pkl', 'rb') as f:
    kde = pickle.load(f)
    
## 1b

# ===== Hyperparameters =====

# Grid
xmin, xmax = 0, 4 # Grid bounds
ymin, ymax = 0, 4 # Grid bounds
nbins = 100 # Use 100 x 100 grid for now
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j] # Use 100 x 100 grid for now
positions = np.vstack([X.ravel(), Y.ravel()])

# Grid for discretizing Beta mixture
mixture_grid = np.linspace(1e-5,1-(1e-5),2000) # Exclude 0 and 1 since Beta density blows up at those points

# Specify quantile
alpha = 0.1

D = kde(positions) # Evaluate D on grid

def compute_classk_prod_f(k, a, b, n, scores_subset,logscale=False):
    '''
    Computes $\prod_{i=1}^{n_k} f(s_{k,i}; \alpha_k, \beta_k)$
    
    Inputs:
        k: class
        a, b: parameters of Beta(a,b)
        scores_subset: vector in which first n elements are scores for class 0, 
        second n elements are scores for class 1, and so on
    '''
    classk_scores = scores_subset[k*n:(k+1)*n]
    
    # Weirdly, some scores are exactly 0, which is a problem because the Beta 
    # density blows up at 0 for some alpha, beta values. We replace these 0 
    # values with randomly sampled values from classk_scores
    classk_scores[classk_scores == 0] = np.random.choice(classk_scores[~(classk_scores == 0)], 
                                                         size=classk_scores[classk_scores == 0].shape,
                                                         replace=True)
    
    f_ski = beta.pdf(classk_scores, a, b)
    
    if logscale:
        log_prod = np.sum(np.log(f_ski))
        return log_prod
    else:     
        prod = np.prod(f_ski)
        return prod

def compute_prob_on_grid(k, positions, D, xmin, xmax, ymin, ymax, n_k, scores_subset):
    '''
    Applies compute_classk_prod_f to all grid points in positions. Replaces nan entries
    with 0 and normalizes the distribution 
    
    Input:
        -k: class
        -n_k: number of instances of class k
    
    Outputs:
        prob: vector of probabilities
        density: prob reshaped into a matrix
    '''

    prod_f = np.array([compute_classk_prod_f(k, positions[0,i], positions[1,i], n_k, scores_subset) 
                       for i in range(len(positions[0]))])
    prob = prod_f * D

    # Replace nan entries with 0
    prob[np.isnan(prob)] = 0

    # We can normalize this discretized distribution
    grid_area = ((xmax - xmin) / nbins) * ((ymax - ymin) / nbins)
    prob = prob / (np.sum(prob * grid_area))
    
    # Reshape probs from vector into square matrix
    density = np.reshape(prob, X.shape) 
    
    # Check if probability contains NaNs or infs
    if np.sum(np.isnan(prob)) + np.sum(np.isinf(prob)) > 0:
        print(f"WARNING: Probabilities for class {k} contains NaNs and/or inf")

    return prob, density


# def get_quantile(density, grid, alpha):
#     assert(np.sum(np.isnan(density) + np.isinf(density)) == 0)
    
# #     density /= np.sum(density) # ensure that density is normalized to sum to 1
#     grid_width = grid[1] - grid[0]
#     sums = np.cumsum(density) * grid_width
#     min_idx = np.argwhere(sums >= 1 - alpha)[0,0]
    
#     return grid[min_idx]


## 2) Estimate quantiles

def estimate_quantile(k, return_samples=False):
    prob, prob_matrix = compute_prob_on_grid(k, positions, D, xmin, xmax, ymin, ymax, n, scores_subset)
    normalized_prob = prob / np.sum(prob)
    
    samples = np.zeros((num_samples,))
    
    for i in range(num_samples):
        
        # 1. Sample alpha_k, beta_k
        idx = np.random.choice(np.arange(len(normalized_prob)), p=normalized_prob)
        alpha_k = positions[0,idx]
        beta_k = positions[1, idx]
#         print('alpha_k, beta_k:', alpha_k, beta_k)
        
        # 2. Sample score
        samples[i] = np.random.beta(alpha_k, beta_k)

    # Compute quantile
    quantile = np.quantile(samples, 1-alpha, interpolation='higher') 
    
    print(f"Class {k} quantile: {quantile:.4f}")
    
    if return_samples:
        return quantile, samples
    else:
        return quantile


# # OPTION 1: Basic for loop
# for k in range(num_classes):
#     quantiles[k] = estimate_quantile(k)
    
    
# # OPTION 2a: Parallelized for loop  
# num_cpus = 72
# print(f'Splitting into {num_cpus} jobs...')

# quantiles = joblib.Parallel(n_jobs=num_cpus)(joblib.delayed(estimate_quantile)(k) for k in range(num_classes))

# OPTION 2b: Parallelized for loop and cache samples
num_cpus = 36 # 72
print(f'Splitting into {num_cpus} jobs...')

qhat_and_sample_list = joblib.Parallel(n_jobs=num_cpus)(joblib.delayed(estimate_quantile)(k, return_samples=True) for k in range(num_classes))

quantiles = [x[0] for x in qhat_and_sample_list]
cached_samples = [x[1] for x in qhat_and_sample_list]

quantiles = np.array(quantiles)
cached_samples = np.array(cached_samples)

print('cached_samples.shape:', cached_samples.shape)

save_to = '.cache/cached_samples_06-10-22.npy'
np.save(save_to, cached_samples)
print(f'Saved cached_samples to {save_to}')
    
## 3) Save quantiles
print('quantiles:', quantiles)

save_to = '.cache/quantiles_06-08-22.npy'
np.save(save_to, quantiles)
print(f'Saved quantiles to {save_to}')


