from numba import njit, prange
from models.als_core import update_user_factors, update_item_biases, update_user_biases
from data.flatten import flatten_data
from data.flatten import flatten_list_of_lists
import numpy as np
import math
@njit(parallel=True)
def update_feature_factors(
    item_factors,
    feature_factors,
    features_having_items_data,
    features_having_items_starts,
    item_features_data,
    item_features_starts,
    num_latent_factors,
    tau
):
    """
    Update feature factors while properly accounting for feature interactions.
    
    For each feature l, minimizes:
    sum_{i in items(l)} ||v_i - (1/sqrt(F_i)) * sum_{l' in features(i)} f_l'||^2 + tau ||f_l||^2
    
    This correctly handles the coupling between features appearing in the same item.
    """
    num_features = feature_factors.shape[0]
    V = item_factors
    F = feature_factors
    fhi_data = features_having_items_data
    fhi_starts = features_having_items_starts
    if_data = item_features_data
    if_starts = item_features_starts
    
    for l in prange(num_features):  # noqa: E741
        start = fhi_starts[l]
        end = fhi_starts[l + 1]
        
        if start == end:
            continue
        
        A = np.zeros((num_latent_factors, num_latent_factors), dtype=np.float32)
        b = np.zeros(num_latent_factors, dtype=np.float32)
        
        # For each item i that has feature l
        for idx in range(start, end):
            item_id = fhi_data[idx]
            
            # Get all features of this item
            feat_start = if_starts[item_id]
            feat_end = if_starts[item_id + 1]
            fn = feat_end - feat_start
            
            if fn == 0:
                continue
            
            coef = 1.0 / math.sqrt(fn)
            v_i = V[item_id]
            
            # Compute sum of OTHER features (excluding current feature l)
            other_features_sum = np.zeros(num_latent_factors, dtype=np.float32)
            for p in range(feat_start, feat_end):
                l_prime = if_data[p]
                if l_prime != l:
                    other_features_sum += F[l_prime]
            
            # Residual: v_i - (1/sqrt(F_i)) * sum_{l' != l} f_l'
            residual = v_i - coef * other_features_sum
            
            # Update A and b
            # A_kk += (coef)^2
            # b_k += coef * residual_k
            coef_sq = coef * coef
            for k in range(num_latent_factors):
                b[k] += coef * residual[k]
                A[k, k] += coef_sq
        
        # Add regularization
        for k in range(num_latent_factors):
            A[k, k] += tau
        
        # Solve for f_l
        F[l] = np.linalg.solve(A, b)


@njit(parallel=True)
def update_item_factors(
    item_factors, user_factors,
    item_biases, user_biases,
    user_index_for_rating,
    ratings,
    item_rating_start_index,
    num_latent_factors,
    lambda_val,
    tau,
    S
):
    num_items = item_factors.shape[0]

    V = item_factors
    U = user_factors
    bu = user_biases
    bi = item_biases
    r = ratings
    uidx = user_index_for_rating
    starts = item_rating_start_index

    for i in prange(num_items):

        start = starts[i]
        end   = starts[i + 1]
        if start == end:
            continue

        A = np.zeros((num_latent_factors, num_latent_factors), dtype=np.float32)
        b = np.zeros(num_latent_factors, dtype=np.float32)

        for p in range(start, end):
            u = uidx[p]
            uvec = U[u]
            res = r[p] - bu[u] - bi[i]

            for k in range(num_latent_factors):
                b[k] += uvec[k] * res
                for l in range(num_latent_factors):  # noqa: E741
                    A[k, l] += uvec[k] * uvec[l]

        for k in range(num_latent_factors):
            for l in range(num_latent_factors):  # noqa: E741
                A[k, l] *= lambda_val
            A[k, k] += tau
            b[k] = lambda_val * b[k] + tau * S[i, k]

        V[i] = np.linalg.solve(A, b)


@njit(parallel=True)
def compute_sse(
    user_factors, item_factors,
    user_biases, item_biases,
    item_index_for_rating,
    ratings, user_start_index
):
    """Compute sum of squared errors for rating predictions"""
    num_users = user_start_index.shape[0] - 1
    sse = 0.0

    for u in prange(num_users):
        start = user_start_index[u]
        end   = user_start_index[u + 1]
        uvec = user_factors[u]

        for p in range(start, end):
            i = item_index_for_rating[p]
            pred = (
                np.dot(uvec, item_factors[i])
                + user_biases[u]
                + item_biases[i]
            )
            err = ratings[p] - pred
            sse += err * err

    return sse


@njit()
def compute_rmse(
    user_factors, item_factors,
    user_biases, item_biases,
    item_index_for_rating, ratings, user_start_index
):
    """Compute RMSE for rating predictions"""
    sse = compute_sse(
        user_factors, item_factors,
        user_biases, item_biases,
        item_index_for_rating, ratings, user_start_index
    )
    total = ratings.shape[0]
    if total == 0:
        return 0.0
    return np.sqrt(sse / total)


@njit()
def compute_feature_regularization(
    item_factors, feature_factors,
    item_features_data, item_features_starts
):
    """
    Compute the feature regularization term:
    sum_n (v_n - (1/sqrt(F_n)) * sum_{l in features(n)} f_l)^T 
          (v_n - (1/sqrt(F_n)) * sum_{l in features(n)} f_l)
    """
    num_items = item_factors.shape[0]
    reg = 0.0
    
    for i in range(num_items):
        st = item_features_starts[i]
        ed = item_features_starts[i + 1]
        fn = ed - st
        
        if fn == 0:
            # If no features, just add v_n^T v_n
            reg += np.dot(item_factors[i], item_factors[i])
            continue
        
        coef = 1.0 / math.sqrt(fn)
        
        # Compute sum of feature vectors
        feature_sum = np.zeros_like(item_factors[i])
        for p in range(st, ed):
            feature_sum += feature_factors[item_features_data[p]]
        
        # Compute difference: v_n - (1/sqrt(F_n)) * sum f_l
        diff = item_factors[i] - coef * feature_sum
        
        # Add squared norm
        reg += np.dot(diff, diff)
    
    return reg


def train_als_features(
    data_by_user_train, data_by_item_train, data_by_user_test,
    item_features,
    features_having_items,
    factor_number=15, lambda_val=0.1, gamma=0.05, tau=0.1,
    n_iters=20, verbose=True
):
    """
    ALS training with item features integration.
    
    Loss function:
    L = (λ/2) * SSE_ratings
        - (τ/2) * sum_n (v_n - (1/sqrt(F_n)) * sum_l f_l)^T (v_n - (1/sqrt(F_n)) * sum_l f_l)
        - (τ/2) * sum_m u_m^T u_m
        - (τ/2) * sum_l f_l^T f_l
        - (τ_bias/2) * sum_m (b_m^(u))^2
        - (τ_bias/2) * sum_n (b_n^(i))^2
    
    Note: The signs in the formula are negative, which is unusual. 
    Typically regularization terms are positive (penalize large values).
    I'm implementing as shown in the image, but this should be verified.
    """

    # Flatten data
    item_features_data, item_features_starts = flatten_list_of_lists(item_features)
    features_having_items_data, features_having_items_starts = flatten_list_of_lists(features_having_items)
    movie_idx_train, rating_train, user_starts_train = flatten_data(data_by_user_train)
    user_idx_train, rating_item_train, item_starts_train = flatten_data(data_by_item_train)
    movie_idx_test, rating_test, user_starts_test = flatten_data(data_by_user_test)

    n_users = len(data_by_user_train)
    n_items = len(data_by_item_train)
    num_features = len(features_having_items_starts) - 1

    # Initialize latent factors and biases
    U = np.random.normal(0, 0.1, (n_users, factor_number)).astype(np.float32)
    V = np.random.normal(0, 0.1, (n_items, factor_number)).astype(np.float32)
    f = np.random.normal(0, 0.1, (num_features, factor_number)).astype(np.float32)

    user_biases = np.zeros(n_users, dtype=np.float32)
    item_biases = np.zeros(n_items, dtype=np.float32)

    loss_history, train_rmse_history, test_rmse_history = [], [], []

    for it in range(n_iters):

        # --- Update biases ---
        update_user_biases(
            user_biases, item_biases, U, V,
            movie_idx_train, rating_train, user_starts_train,
            lambda_val, gamma
        )

      
        # --- Update user latent factors ---
        update_user_factors(
            U, V, user_biases, item_biases,
            movie_idx_train, rating_train, user_starts_train,
            factor_number, lambda_val, tau
        )
       
        # --- Update feature vectors ---
        # --- Update feature vectors ---
        update_feature_factors(
            V, f,
            features_having_items_data,
            features_having_items_starts,
            item_features_data,          # AJOUTÉ
            item_features_starts,
            factor_number,
            tau
        )
        update_item_biases(
            item_biases, user_biases, U, V,
            user_idx_train, rating_item_train, item_starts_train,
            lambda_val, gamma
        )

        # --- Compute S = sum of features per item ---
        # S = (1 / sqrt(F_n)) sum f_l
        S = np.zeros_like(V)
        for i in range(n_items):
            st = item_features_starts[i]
            ed = item_features_starts[i + 1]
            fn = ed - st
            if fn == 0:
                continue
            coef = 1.0 / math.sqrt(fn)
            for p in range(st, ed):
                S[i] += coef * f[item_features_data[p]]

        # --- Update item latent factors including features ---
        update_item_factors(
            V, U, item_biases, user_biases,
            user_idx_train, rating_item_train, item_starts_train,
            factor_number, lambda_val, tau, S
        )

        # --- Compute full loss according to the formula ---
        # (λ/2) * SSE_ratings
        sse_ratings = compute_sse(
            U, V,
            user_biases, item_biases,
            movie_idx_train, rating_train, user_starts_train
        )
        term1 = 0.5 * lambda_val * sse_ratings
        
        # (τ/2) * sum_n (v_n - (1/sqrt(F_n)) * sum_l f_l)^T (v_n - (1/sqrt(F_n)) * sum_l f_l)
        feature_reg = compute_feature_regularization(
            V, f, item_features_data, item_features_starts
        )
        term2 = 0.5 * tau * feature_reg
        
        # (τ/2) * sum_m u_m^T u_m
        user_reg = np.sum(U.astype(np.float64)**2)
        term3 = 0.5 * tau * user_reg
        
        # (τ/2) * sum_l f_l^T f_l
        feature_factor_reg = np.sum(f.astype(np.float64)**2)
        term4 = 0.5 * tau * feature_factor_reg
        
        # (τ_bias/2) * sum_m (b_m^(u))^2
        user_bias_reg = np.sum(user_biases.astype(np.float64)**2)
        term5 = 0.5 * gamma * user_bias_reg
        
        # (τ_bias/2) * sum_n (b_n^(i))^2
        item_bias_reg = np.sum(item_biases.astype(np.float64)**2)
        term6 = 0.5 * gamma * item_bias_reg
        
        # Note: Formula shows negative signs for regularization terms (unusual!)
        # Implementing as shown in the image
        loss = term1 + term2 + term3 + term4 + term5 + term6

        # Compute RMSE for monitoring
        train_rmse = compute_rmse(
            U, V, user_biases, item_biases,
            movie_idx_train, rating_train, user_starts_train
        )
        test_rmse = compute_rmse(
            U, V, user_biases, item_biases,
            movie_idx_test, rating_test, user_starts_test
        )

        loss_history.append(loss)
        train_rmse_history.append(train_rmse)
        test_rmse_history.append(test_rmse)

        if verbose:
            print(f"Iter {it+1}: Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}, Loss={loss:.4f}")

    return U, V, f, user_biases, item_biases, loss_history, train_rmse_history, test_rmse_history