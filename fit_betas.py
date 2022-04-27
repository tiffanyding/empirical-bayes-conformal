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


def compute_variance(k):
    '''
    Computes variances of class k scores
    '''
    return np.std(scores[labels==k,k]) ** 2

def beta_variance(alpha, beta):
    '''
    Implements variance formula for Beta distribution
    '''
    num = alpha * beta
    denom = (alpha + beta)^2 * (alpha + beta + 1)
    
    return num / denom

def beta_MoM(data):
    '''
    Method of Moments estimates for Beta parameters 
    (See https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/07%3A_Point_Estimation/7.02%3A_The_Method_of_Moments)
    '''
    M = np.mean(data)
    M2 = np.mean(data ** 2)
    est_alpha = M * (M - M2) / (M2 - M**2)
    est_beta = (1 - M) * (M - M2) / (M2 - M**2)
    
    return est_alpha, est_beta
                             

def fit_beta(k, min_variance):
    '''
    Fit Beta distribution to class k scores. If beta.fit() returns an estimated distribution that is 
    unreasonably peaked (i.e., has variance lower than min_variance), we use method of moments instead
    '''
    try:
        class_k_scores = scores[labels==k,k]
        est_alpha, est_beta = beta.fit(class_k_scores, method='MLE')[:2]
        
        # If estimated Beta distribution is unreasonaby peaked, use MoM
        if beta_variance(est_beta, est_beta) < min_variance:
            print('Using Method of Moments for class', k)
            est_alpha, est_beta = beta_MoM(class_k_scores)
            
    except: # If the Beta fitting fails in any way, use MoM
        print('Using Method of Moments for class', k)
        est_alpha, est_beta = beta_MoM(class_k_scores)
    
    return est_alpha, est_beta

if __name__ == '__main__':
    print('Loading data...')
    scores = 1 - torch.load('/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/logits/imagenet_train_subset_softmax.pt', map_location=torch.device('cpu')).numpy()
    labels = torch.load('/home/eecs/tiffany_ding/code/SimCLRv2-Pytorch/.cache/logits/imagenet_train_subset_labels.pt', map_location=torch.device('cpu')).type(torch.LongTensor).numpy()
    print('Done loading data')

    num_classes = 1000

    #num_cpus = multiprocessing.cpu_count()
    num_cpus = 72
    print(f'Splitting into {num_cpus} jobs...')


    # Compute minimum variance and use it to detect instances in which beta.fit returns a bad fit
    variances = joblib.Parallel(n_jobs=num_cpus)(joblib.delayed(compute_variance)(k) for k in range(num_classes))
    min_variance = min(variances)
    print(f'Smallest class score variance: {min_variance:.3f}')

    beta_params = joblib.Parallel(n_jobs=num_cpus)(joblib.delayed(fit_beta)(k, min_variance) for k in range(num_classes))

    print(beta_params)
    save_to = '/home/eecs/tiffany_ding/code/empirical-bayes-conformal/data/est_beta_params.npy'
    np.save(save_to, beta_params)
    print('Saved estimates Beta parameters to', save_to)





