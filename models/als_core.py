import numpy as np
from data.flatten import flatten_data
from numba import njit, prange


@njit(parallel=True)
def update_item_factors(
    item_factors, user_factors,
    item_biases, user_biases,
    user_index_for_rating,
    ratings,
    item_rating_start_index,
    num_latent_factors,
    weight_lambda,
    reg_tau
):

    num_items = item_factors.shape[0]

    #identity_matrix = np.eye(num_latent_factors, dtype=np.float32)

    lambda_f = np.float32(weight_lambda)
    tau_f = np.float32(reg_tau)

    # local variables to  numba
    V_loc = item_factors
    U_loc = user_factors
    item_bias_loc = item_biases
    user_bias_loc = user_biases
    ratings_loc = ratings
    user_idx_loc = user_index_for_rating
    start_loc = item_rating_start_index

    for item_id in prange(num_items):

        start_pos = start_loc[item_id]
        end_pos   = start_loc[item_id + 1]
        num_ratings_item = end_pos - start_pos

        if num_ratings_item == 0:
            continue


        normal_matrix = np.zeros((num_latent_factors, num_latent_factors), dtype=np.float32)
        rhs_vector    = np.zeros(num_latent_factors, dtype=np.float32)

        bias_item = item_bias_loc[item_id]


        for rating_pos in range(start_pos, end_pos):

            user_id = user_idx_loc[rating_pos]
            user_vec = U_loc[user_id]

            residual = ratings_loc[rating_pos] - user_bias_loc[user_id] - bias_item


            for k in range(num_latent_factors):
                uk = user_vec[k]
                rhs_vector[k] += uk * residual


                for l in range(num_latent_factors):  # noqa: E741
                    normal_matrix[k, l] += uk * user_vec[l]


        for k in range(num_latent_factors):
            for l in range(num_latent_factors):  # noqa: E741
                normal_matrix[k, l] *= lambda_f
            normal_matrix[k, k] += tau_f


        rhs_vector *= lambda_f

        # Résolution du système linéaire
        V_loc[item_id] = np.linalg.solve(normal_matrix, rhs_vector)


# Same step like the last one
@njit(parallel=True)
def update_user_factors(user_factors, item_factors, user_biases, item_biases,
                        item_index_for_rating, ratings, user_start_index,
                        num_factors, lambda_reg, tau_reg):

    num_users = user_factors.shape[0]

    #I = np.eye(num_factors, dtype=np.float32)
    lambda_val = np.float32(lambda_reg)
    tau_val = np.float32(tau_reg)

    # local variables to help numba
    item_factors_loc = item_factors
    user_biases_loc = user_biases
    item_biases_loc = item_biases
    ratings_loc = ratings
    item_index_loc = item_index_for_rating
    user_start_loc = user_start_index

    for user_id in prange(num_users):

        start = user_start_loc[user_id]
        end = user_start_loc[user_id + 1]
        num_ratings_user = end - start
        if num_ratings_user == 0:
            continue

        gram_matrix = np.zeros((num_factors, num_factors), dtype=np.float32)
        rhs_vector = np.zeros(num_factors, dtype=np.float32)

        user_bias = user_biases_loc[user_id]

        for rating_pos in range(start, end):

            item_id = item_index_loc[rating_pos]
            item_vector = item_factors_loc[item_id]

            centered_rating = ratings_loc[rating_pos] - user_bias - item_biases_loc[item_id]

            for k in range(num_factors):
                item_factor_k = item_vector[k]
                rhs_vector[k] += item_factor_k * centered_rating

                for l in range(num_factors):  # noqa: E741
                    gram_matrix[k, l] += item_factor_k * item_vector[l]


        for k in range(num_factors):
            for l in range(num_factors):  # noqa: E741
                gram_matrix[k, l] = lambda_val * gram_matrix[k, l]
            gram_matrix[k, k] += tau_val

        right_hand_side = lambda_val * rhs_vector

        user_factors[user_id] = np.linalg.solve(gram_matrix, right_hand_side)


@njit(parallel=True)
def update_user_biases(
    user_biases, item_biases,
    user_factors, item_factors,
    item_index_for_rating,
    ratings,
    user_start_index,
    lambda_reg,
    gamma_reg
):
    num_users = user_biases.shape[0]

    lambda_f = np.float32(lambda_reg)
    gamma_f  = np.float32(gamma_reg)

    # local variables to help numba
    U_loc = user_factors
    V_loc = item_factors
    item_bias_loc = item_biases
    ratings_loc = ratings
    item_idx_loc = item_index_for_rating
    start_idx_loc = user_start_index

    for user_id in prange(num_users):

        start = start_idx_loc[user_id]
        end   = start_idx_loc[user_id + 1]
        num_ratings = end - start

        if num_ratings == 0:
            user_biases[user_id] = 0.0
            continue

        residual_sum = 0.0
        user_vec = U_loc[user_id]

        for rating_pos in range(start, end):
            item_id = item_idx_loc[rating_pos]
            residual_sum += ratings_loc[rating_pos] - (np.dot(user_vec, V_loc[item_id]) + item_bias_loc[item_id])

        user_biases[user_id] = (lambda_f * residual_sum) / (lambda_f * num_ratings + gamma_f)

@njit(parallel=True)
def update_item_biases(
    item_biases, user_biases,
    user_factors, item_factors,
    user_index_for_rating,
    ratings,
    item_start_index,
    lambda_reg,
    gamma_reg
):
    num_items = item_biases.shape[0]

    lambda_f = np.float32(lambda_reg)
    gamma_f  = np.float32(gamma_reg)

    # local variables to help numba
    U_loc = user_factors
    V_loc = item_factors
    ratings_loc = ratings
    user_idx_loc = user_index_for_rating
    start_idx_loc = item_start_index
    user_bias_loc = user_biases

    for item_id in prange(num_items):

        start = start_idx_loc[item_id]
        end   = start_idx_loc[item_id + 1]
        num_ratings_item = end - start

        if num_ratings_item == 0:
            item_biases[item_id] = 0.0
            continue

        residual_sum = 0.0
        item_vec = V_loc[item_id]

        for rating_pos in range(start, end):
            user_id = user_idx_loc[rating_pos]
            residual_sum += ratings_loc[rating_pos] - (np.dot(U_loc[user_id], item_vec) + user_bias_loc[user_id])

        item_biases[item_id] = (lambda_f * residual_sum) / (lambda_f * num_ratings_item + gamma_f)

@njit(parallel=True)
def compute_sse(
    user_factors, item_factors,
    user_biases, item_biases,
    item_index_for_rating,
    ratings,
    user_start_index
):
    num_users = user_start_index.shape[0] - 1
    sse_per_user = np.zeros(num_users, dtype=np.float64)

    # Alias locaux pour Numba
    U_loc = user_factors
    V_loc = item_factors
    ratings_loc = ratings
    item_idx_loc = item_index_for_rating
    start_idx_loc = user_start_index
    user_bias_loc = user_biases
    item_bias_loc = item_biases

    for user_id in prange(num_users):

        start = start_idx_loc[user_id]
        end   = start_idx_loc[user_id + 1]
        sse_local = 0.0
        user_vec = U_loc[user_id]

        for rating_pos in range(start, end):
            item_id = item_idx_loc[rating_pos]
            residual = ratings_loc[rating_pos] - (
                np.dot(user_vec, V_loc[item_id]) + user_bias_loc[user_id] + item_bias_loc[item_id]
            )
            sse_local += residual * residual

        sse_per_user[user_id] = sse_local

    total_sse = 0.0
    for user_id in range(num_users):
        total_sse += sse_per_user[user_id]

    return total_sse



@njit()
def compute_rmse_als(U, V, user_biases, item_biases, movie_idx, rating, starts):
    sse = compute_sse(U, V, user_biases, item_biases, movie_idx, rating, starts)
    total = rating.shape[0]
    if total == 0:
        return 0.0
    return np.sqrt(sse / total)

# ---------------- ALS training ----------------
def train_als(data_by_user_train, data_by_item_train, data_by_user_test,
              factor_number=10, lambda_val=0.1, gamma=0.05, tau=0.1, n_iters=20, verbose=True):

    movie_idx_train, rating_train, user_starts_train = flatten_data(data_by_user_train)
    user_idx_train, rating_item_train, item_starts_train = flatten_data(data_by_item_train)
    movie_idx_test, rating_test, user_starts_test = flatten_data(data_by_user_test)

    n_users = len(data_by_user_train)
    n_items = len(data_by_item_train)

    U = np.random.normal(0, 0.1, (n_users, factor_number)).astype(np.float32)
    V = np.random.normal(0, 0.1, (n_items, factor_number)).astype(np.float32)
    user_biases = np.zeros(n_users, dtype=np.float32)
    item_biases = np.zeros(n_items, dtype=np.float32)

    loss_history, train_rmse_history, test_rmse_history = [], [], []

    for it in range(n_iters):
        update_user_biases(user_biases, item_biases, U, V, movie_idx_train, rating_train, user_starts_train, lambda_val, gamma)
        update_user_factors(U, V, user_biases, item_biases, movie_idx_train, rating_train, user_starts_train,
                            factor_number, lambda_val, tau)
        update_item_biases(item_biases, user_biases, U, V, user_idx_train, rating_item_train, item_starts_train, lambda_val, gamma)
        update_item_factors(V, U, item_biases, user_biases, user_idx_train, rating_item_train, item_starts_train,
                            factor_number, lambda_val, tau)
        sse = compute_sse(U, V, user_biases, item_biases, movie_idx_train, rating_train, user_starts_train)
        reg_factors = np.sum(U.astype(np.float64)**2) + np.sum(V.astype(np.float64)**2)
        reg_biases = np.sum(user_biases.astype(np.float64)**2) + np.sum(item_biases.astype(np.float64)**2)
        loss = 0.5*lambda_val*sse + 0.5*tau*reg_factors + 0.5*gamma*reg_biases

        train_rmse = compute_rmse_als(U, V, user_biases, item_biases, movie_idx_train, rating_train, user_starts_train)
        test_rmse = compute_rmse_als(U, V, user_biases, item_biases, movie_idx_test, rating_test, user_starts_test)

        loss_history.append(loss)
        train_rmse_history.append(train_rmse)
        test_rmse_history.append(test_rmse)

        if verbose:
            print(f"Iter {it+1}: Train RMSE={train_rmse:.4f}, Test RMSE={test_rmse:.4f}, Loss={loss:.4f}")

    return U, V, user_biases, item_biases, loss_history, train_rmse_history, test_rmse_history