import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import torch

from scipy.interpolate import interp2d
from scipy.stats import beta, multivariate_normal, nct, t

np.random.seed(0)

## 1) Setup 

## 1a

# Load data 
softmax_scores = torch.load('/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/logits/imagenet_train_subset_softmax.pt', map_location=torch.device('cpu'))
softmax_scores = softmax_scores.numpy()
labels = torch.load('/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/logits/imagenet_train_subset_labels.pt', map_location=torch.device('cpu'))
labels = labels.type(torch.LongTensor).numpy()

# Select subset of data
np.random.seed(0)

n = 20 # Number of calibration points per class k
num_classes = 1000

softmax_scores_subset = np.zeros((num_classes * n, num_classes))
labels_subset = np.zeros((num_classes * n, ), dtype=np.int8)

for k in range(num_classes):
    
    # Randomly select n instances of class k
    idx = np.argwhere(labels==k).flatten()
    selected_idx = np.random.choice(idx, replace=False, size=(n,))
    
    softmax_scores_subset[n*k:n*(k+1), :] = softmax_scores[selected_idx, :]
    labels_subset[n*k:(n+1)*k] = k
    
# Only select data for which k is true class
scores_subset = np.array([softmax_scores_subset[row,labels_subset[row]] for row in range(len(labels_subset))])

# Load KDE estimate of prior
with open('.cache/kde.pkl', 'rb') as f:
    kde = pickle.load(f)
    
## 1b

# ===== Hyperparameters =====

num_classes = 1000

# Grid
xmin, xmax = 0, 4 # Grid bounds
ymin, ymax = 0, 4 # Grid bounds
nbins = 100 # Use 100 x 100 grid for now
X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j] # Use 100 x 100 grid for now
positions = np.vstack([X.ravel(), Y.ravel()])

# # Threshold for truncating probability distribution
# threshold = .001 # this is over a small area 

# Number of rejection sampling samplies
num_samples = 1000

# Grid for discretizing Beta mixture
mixture_grid = np.linspace(1e-5,1-(1e-5),2000) # Exclude 0 and 1 since Beta density blows up at those points

# Specify quantile
alpha = 0.1

D = kde(positions) # Evaluate D on grid

def compute_classk_prod_f(k, a, b, scores_subset,logscale=False):
    '''
    Computes $\prod_{i=1}^{n_k} f(s_{k,i}; \alpha_k, \beta_k)$
    
    Inputs:
        k: class
        a, b: parameters of Beta(a,b)
    '''
    f_ski = beta.pdf(scores_subset[k*n:(k+1)*n], a, b)
    
    if logscale:
        log_prod = np.sum(np.log(f_ski))
        return log_prod
    else:     
        prod = np.prod(f_ski)
        return prod

def compute_prob_on_grid(k, positions, D, xmin, xmax, ymin, ymax, scores_subset):
    '''
    Applies compute_classk_prod_f to all grid points in positions. Replaces nan entries
    with 0 and normalizes the distribution 
    
    Outputs:
        prob: vector of probabilities
        density: prob reshaped into a matrix
    '''

    prod_f = np.array([compute_classk_prod_f(k, positions[0,i], positions[1,i], scores_subset) 
                       for i in range(len(positions[0]))])
    prob = prod_f * D

    # Replace nan entries with 0
    prob[np.isnan(prob)] = 0

    # We can normalize this discretized distribution
    grid_area = ((xmax - xmin) / nbins) * ((ymax - ymin) / nbins)
    prob = prob / (np.sum(prob * grid_area))
    
    # Reshape probs from vector into square matrix
    density = np.reshape(prob, X.shape) 

    return prob, density


def get_quantile(density, grid, alpha):
    assert(np.sum(np.isnan(density) + np.isinf(density)) == 0)
    
#     density /= np.sum(density) # ensure that density is normalized to sum to 1
    grid_width = grid[1] - grid[0]
    sums = np.cumsum(density) * grid_width
    min_idx = np.argwhere(sums >= 1 - alpha)[0,0]
    
    return grid[min_idx]


## 2) Estimate quantiles

quantiles = np.zeros((num_classes,))

num_samples = 100000


for k in range(num_classes):
    print(f"Class {k}")

    # Computes $\prod_{i=1}^{n_k} f(s_{k,i}; \alpha_k, \beta_k)$
    prob, prob_matrix = compute_prob_on_grid(k, positions, D, xmin, xmax, ymin, ymax, scores_subset)
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
    samples = np.sort(samples)
    quantile = np.quantile(samples, np.ceil((1-alpha)*(n+1))/n)
    
    print(f"Quantile: {quantile:.4f}")
    quantiles[k] = quantile
    
    
## 3) Save estimated quantiles 
save_to = '.cache/quantiles_06-02-22.npy'
np.save(save_to, quantiles)
print(f'Saved quantiles to {save_to}')
