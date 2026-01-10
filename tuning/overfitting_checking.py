import numpy as np

def per_user_rmse_from_userlists(U, V, ub, ib, user_lists):
    n_users = len(user_lists)
    rmses = np.full(n_users, np.nan, dtype=float)
    counts = np.zeros(n_users, dtype=int)
    for u, items in enumerate(user_lists):
        if items:
            se = sum((ub[u] + ib[i] + float(np.dot(U[u], V[i])) - r)**2 for i, r in items)
            counts[u] = len(items)
            rmses[u] = np.sqrt(se / counts[u])
    return rmses, counts


def evaluate_over_K(train_als, K, data_by_user_train, data_by_movie_train, data_by_user_test,
                    lambda_val=0.03, gamma=0.05, tau=6.9, n_iters=20):
    results = {}
    print(f"Training K={K} ...")
    U, V, ub, ib, loss_hist, tr_rmse_hist, te_rmse_hist = train_als(
        data_by_user_train, data_by_movie_train, data_by_user_test,
        factor_number=K, lambda_val=lambda_val, gamma=gamma, tau=tau, n_iters=n_iters
    )
    rmse_train_u, counts_train = per_user_rmse_from_userlists(U, V, ub, ib, data_by_user_train)
    rmse_test_u, counts_test   = per_user_rmse_from_userlists(U, V, ub, ib, data_by_user_test)
    gap_u = np.full_like(rmse_train_u, np.nan)
    mask = ~np.isnan(rmse_train_u) & ~np.isnan(rmse_test_u)
    gap_u[mask] = rmse_test_u[mask] - rmse_train_u[mask]

    results = {
        'U': U, 'V': V, 'ub': ub, 'ib': ib,
        'rmse_train_u': rmse_train_u, 'rmse_test_u': rmse_test_u, 'gap_u': gap_u,
        'counts_train': counts_train, 'counts_test': counts_test
    }
    return results




def evaluate_over_K_with_f(train_als_features, data_by_user_train, data_by_movie_train, data_by_user_test,
    item_features,features_having_items, K= 15,
                    lambda_val=0.1, gamma=0.04, tau=1.9, n_iters=20):
    results = {}
    print(f"Training K={K} ...")
    U, V, f,  ub, ib, loss_hist, tr_rmse_hist, te_rmse_hist = train_als_features(
    data_by_user_train, data_by_movie_train, data_by_user_test,
    item_features,
    features_having_items,
    factor_number=K, lambda_val=lambda_val, gamma=gamma, tau=tau, n_iters=n_iters,
    verbose=True
    )
    rmse_train_u, counts_train = per_user_rmse_from_userlists(U, V, ub, ib, data_by_user_train)
    rmse_test_u, counts_test   = per_user_rmse_from_userlists(U, V, ub, ib, data_by_user_test)
    gap_u = np.full_like(rmse_train_u, np.nan)
    mask = ~np.isnan(rmse_train_u) & ~np.isnan(rmse_test_u)
    gap_u[mask] = rmse_test_u[mask] - rmse_train_u[mask]

    results = {
        'U': U, 'V': V, 'ub': ub, 'ib': ib,
        'rmse_train_u': rmse_train_u, 'rmse_test_u': rmse_test_u, 'gap_u': gap_u,
        'counts_train': counts_train, 'counts_test': counts_test
    }
    return results
