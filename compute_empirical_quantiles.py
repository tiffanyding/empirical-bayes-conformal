import matplotlib.pyplot as plt
import numpy as np
import torch

# For parallelizing
import multiprocessing
import joblib

from collections import Counter
from scipy import special
from scipy.stats import beta

from conformal_utils import *
from utils.imagenet_helpers import ImageNetHierarchy

# np.random.seed(0)

alpha = 0.1

print(f"Using alpha={alpha}")

print('Loading data...')
scores = 1 - torch.load('/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/logits/imagenet_train_subset_softmax.pt', map_location=torch.device('cpu'))
labels = torch.load('/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/logits/imagenet_train_subset_labels.pt', map_location=torch.device('cpu')).type(torch.LongTensor)
print('Done loading data')

num_classes = 1000 # 1000

# num_cpus = multiprocessing.cpu_count()
num_cpus = 1 # 24
print(f'Splitting into {num_cpus} jobs...')

def process(k):
    print(k)
    class_k_scores = scores[labels==k,k]
    n = len(class_k_scores)
    return torch.quantile(class_k_scores, np.ceil((n+1)*(1-alpha))/n)
        
quantiles = joblib.Parallel(n_jobs=num_cpus)(joblib.delayed(process)(k) for k in range(num_classes))
quantiles = np.array(quantiles)

print(quantiles)
save_to = f'/home/eecs/tiffany_ding/code/empirical-bayes-conformal/data/empirical_quantiles_alpha={alpha}.npy'
np.save(save_to, quantiles)
print('Saved empirical quantiles to', save_to)





