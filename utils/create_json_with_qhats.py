import numpy as np
import json

"""
Instructions:
1. Go to the empirical-bayes-conformal directory
2. Run python utils/create_json_with_qhats.py, which creates 'data/imagenet_with_qhats.json'
3. Go https://observablehq.com/d/5bb27b3940f0bc23. Click the paper clip icon on the right side 
   and upload 'data/imagenet_with_qhats.json'
"""
   
def get_qhat(idx):
    '''
    Return the qhat for a given class index
    '''
    return qhats[idx]

def get_coverage(idx):
    '''
    Return the class specific coverage for a given class index
    '''
    return class_specific_cvgs[idx]

def add_qhat_to_all_names(children):
    for c in children:
        if 'index' in c:
            # To save space, we truncate all names at the first comma if one exists
            # before adding the extra information
            c['name'] = c['name'].split(',')[0]
            c['name'] += f" (qhat: {get_qhat(c['index']):.3f} | cov: {get_coverage(c['index']):.3f})"
            # c['name'] += f" [idx {c['index']}] ({get_qhat(c['index']):.3f})"
        else:
            pass
            # print(f"{c['name']} does not have an index")

        if 'children' in c:
            add_qhat_to_all_names(c['children']) # Recursive call


# Load data
with open('data/imagenet.json') as f:
    imagenet_json = json.load(f)
   
qhats = np.load('data/class_balanced_qhats.npy')
class_specific_cvgs = np.load('data/class_specific_coverages.npy')

# Add qhats to end of names
add_qhat_to_all_names(imagenet_json['children'])

print(imagenet_json)

# Save new json file
save_to = 'data/imagenet_with_qhats.json'
with open(save_to, 'w') as f:
    json.dump(imagenet_json, f)
    print(f'Saved ImageNet with qhats appended to names to {save_to}')


# Check if all indices appear in imagenet.json
# def check_indices(unseen_indices, curr_node):
#     if 'index' in curr_node:
#         print('Encountered index', curr_node['index'])
#         unseen_indices.discard(curr_node['index'])
#     if 'children' in curr_node:
#         for c in curr_node['children']:
#             check_indices(unseen_indices, c)

# unseen_indices = set(np.arange(1000))
# check_indices(unseen_indices, imagenet_json)
# print('Unseen indicies:', unseen_indices)
