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


np.random.seed(0)

# print('Loading data...')
# softmax_scores = torch.load('/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/logits/imagenet_train_subset_softmax.pt', map_location=torch.device('cpu'))
# softmax_scores = softmax_scores.numpy()
# labels = torch.load('/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/logits/imagenet_train_subset_labels.pt', map_location=torch.device('cpu'))
# labels = labels.type(torch.LongTensor).numpy()
# print('Done loading data')

# scores = 1 - softmax_scores

# print('Done computing scores')

print('Loading data...')
scores = 1 - torch.load('/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/logits/imagenet_train_subset_softmax.pt', map_location=torch.device('cpu')).numpy()
labels = torch.load('/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/logits/imagenet_train_subset_labels.pt', map_location=torch.device('cpu')).type(torch.LongTensor).numpy()
print('Done loading data')

num_classes = 1000

#num_cpus = multiprocessing.cpu_count()
num_cpus = 72
print(f'Splitting into {num_cpus} jobs...')

def process(k):
    try:
        return beta.fit(scores[labels==k,k])[:2]
    except: # If the Beta fitting fails in any way, just return infinity, infinity 
        return (np.inf, np.inf) 
        
beta_params = joblib.Parallel(n_jobs=num_cpus)(joblib.delayed(process)(k) for k in range(num_classes))

print(beta_params)
save_to = '/home/eecs/tiffany_ding/code/empirical-bayes-conformal/data/est_beta_params.npy'
np.save(save_to, beta_params)
print('Saved estimated Beta parameters to', save_to)





