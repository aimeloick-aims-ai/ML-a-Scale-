import numpy as np

def top_feature_contributions(
    user_vector,    
    item_idx,          
    item_features,     
    feature_factors    
):

    contributions = []
    features_of_item = item_features[item_idx]

    if len(features_of_item) == 0:
        return contributions 

    F_i = len(features_of_item)
    coef = 1.0 / np.sqrt(F_i) 
    for f_idx in features_of_item:
        f_vector = feature_factors[f_idx]
        contrib = np.dot(user_vector, coef * f_vector)
        contributions.append((f_idx, contrib))

    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions
