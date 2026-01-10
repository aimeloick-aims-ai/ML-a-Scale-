import numpy as np

def top_historical_contributions(
    item_idx,
    V,
    user_history,          
    lambda_val=0.1,
    tau=1.9
):
    k = V.shape[1]

    # --- User matrix A_u ---
    A = np.zeros((k, k), dtype=np.float32)
    for j, r in user_history:
        vj = V[j]
        A += r * np.outer(vj, vj)

    A = lambda_val * A + tau * np.eye(k)

    # --- Weighted item vector ---
    vi_weighted = np.linalg.solve(A, V[item_idx])

    # --- Contributions ---
    contributions = []
    for j, r in user_history:
        contrib = np.dot(vi_weighted, V[j]) * r
        contributions.append((j, contrib))

    contributions.sort(key=lambda x: x[1], reverse=True)
    return contributions


def top_historical_contributionsd(
    user_idx, item_idx, 
    U, V, f, 
    user_history,         
    item_features,
    lambda_val=0.1,
    tau=0.1
):
    
    k = U.shape[1]
    
    # --- Compute user matrix A_u ---
    A = np.zeros((k, k), dtype=np.float32)
    for j in user_history:
        vj = V[j]
        A += np.outer(vj, vj)
    A = lambda_val * A
    A += tau * np.eye(k)  # regularization

    # --- Compute weighted vi for the target item ---
    vi = V[item_idx]
    vi_weighted = np.linalg.solve(A, vi)

    # --- Contributions of historical items ---
    contributions = []
    for j in user_history:
        vj = V[j]
        cuj = 1.0  # Binary value here; can use rating if desired
        contrib = np.dot(vi_weighted, vj) * cuj
        contributions.append((j, contrib))
    
    # --- Sort in descending order and return top-k ---
    contributions.sort(key=lambda x: x[1], reverse=True)
    
    return contributions
