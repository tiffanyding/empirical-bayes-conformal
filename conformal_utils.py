import matplotlib.pyplot as plt
import numpy as np

#========================================
#   Standard conformal inference
#========================================

def compute_qhat(class_scores, true_labels, alpha=.05, plot_scores=False):
    '''
    Compute quantile q_hat that will result in marginal coverage of (1-alpha)
    
    Inputs:
        class_scores: num_instances x num_instances array of scores, where a higher score indicates more uncertainty
        true_labels: num_instances length array of ground truth labels
    
    '''
    # Select scores that correspond to correct label
    scores = np.squeeze(np.take_along_axis(class_scores, np.expand_dims(true_labels, axis=1), axis=1))
    
    # Sort scores
    scores = np.sort(scores)

    # Identify score q_hat such that ~(1-alpha) fraction of scores are below qhat 
    #    Note: More precisely, it is (1-alpha) times a small correction factor
    n = len(true_labels)
    q_hat = np.quantile(scores, np.ceil((n+1)*(1-alpha))/n)
    
    # Plot score distribution
    if plot_scores:
        plt.hist(scores)
        plt.title('Score distribution')
        plt.show()

    return q_hat

# Create prediction sets
def create_prediction_sets(class_probs, q_hat):
    assert(not hasattr(q_hat, '__iter__')), "q_hat should be a single number and not a list or array"
    set_preds = []
    num_samples = len(class_probs)
    for i in range(num_samples):
        set_preds.append(np.where(class_probs[i,:] <= q_hat)[0])
        
    return set_preds

#========================================
#   Class-balanced conformal inference
#========================================

def compute_class_specific_qhats(cal_class_scores, cal_true_labels, alpha=.05, default_qhat=None):
    '''
    Computes class-specific quantiles (one for each class) that will result in marginal coverage of (1-alpha)
    
    Inputs:
        - cal_class_scores: num_instances x num_classes array where class_scores[i,j] = score of class j for instance i
        - cal_true_labels: num_instances-length array of true class labels (0-indexed)
        - alpha: Determines desired coverage level
        - default_qhat: For classes that do not appear in cal_true_labels, the class specific qhat is set to default_qhat
    '''
    num_samples = len(cal_true_labels)
    q_hats = np.zeros((cal_class_scores.shape[1],)) # q_hats[i] = quantile for class i
    for k in range(cal_class_scores.shape[1]):
        # Only select data for which k is true class
        idx = (cal_true_labels == k)
        scores = cal_class_scores[idx, k]
        
        if len(scores) == 0:
            assert default_qhat is not None, f"Class {k} does not appear in the calibration set, so the quantile for this class cannot be computed. Please specify a value for default_qhat to use in this case."
            print(f'Warning: Class {k} does not appear in the calibration set,', 
                  f'so default q_hat value of {default_qhat} will be used')
            q_hats[k] = default_qhat
        else:
            scores = np.sort(scores)
            num_samples = len(scores)
            val = np.ceil((num_samples+1)*(1-alpha)) / num_samples
            if val > 1:
                assert default_qhat is not None, f"Class {k} does not appear enough times to compute a proper quantile. Please specify a value for default_qhat to use in this case."
                print(f'Warning: Class {k} does not appear enough times to compute a proper quantile,', 
                      f'so default q_hat value of {default_qhat} will be used')
                q_hats[k] = default_qhat
#                 q_hats[k] = np.inf
            else:
                q_hats[k] = np.quantile(scores, val)
       
#     print('q_hats', q_hats)
    return q_hats

# Create class_balanced prediction sets
def create_cb_prediction_sets(class_scores, q_hats):
    '''
    Inputs:
        - class_scores: num_instances x num_classes array where class_scores[i,j] = score of class j for instance i
        - q_hats: as output by compute_class_specific_quantiles
    '''
    set_preds = []
    num_samples = len(class_scores)
    for i in range(num_samples):
        set_preds.append(np.where(class_scores[i,:] <= q_hats)[0])
        
    return set_preds

#========================================
#   Evaluation
#========================================


# Helper function for computing accuracy (marginal coverage) of confidence sets
def compute_coverage(true_labels, set_preds):
    num_correct = 0
    for true_label, preds in zip(true_labels, set_preds):
        if true_label in preds:
            num_correct += 1
    set_pred_acc = num_correct / len(true_labels)
    
    return set_pred_acc

# Helper function for computing class-specific coverage of confidence sets
def compute_class_specific_coverage(true_labels, set_preds):
    num_classes = max(true_labels) + 1
    class_specific_cov = np.zeros((num_classes,))
    for k in range(num_classes):
        idx = np.where(true_labels == k)[0]
        selected_preds = [set_preds[i] for i in idx]
        num_correct = np.sum([1 if np.any(pred_set == k) else 0 for pred_set in selected_preds])
        class_specific_cov[k] = num_correct / len(selected_preds)
        
    return class_specific_cov