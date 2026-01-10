import numpy as np
from data.flatten import flatten_data
from numba import njit, prange


@njit(parallel=True)
def update_user_biases(user_biases, item_biases, item_index_for_rating, ratings,
                       user_start_index, lambda_reg, gamma_reg):

    num_users = user_biases.shape[0]

    for user_id in prange(num_users):

        residual_sum = 0.0
        start = user_start_index[user_id]
        end = user_start_index[user_id + 1]
        num_ratings_user = end - start

        for rating_pos in range(start, end):
            item_id = item_index_for_rating[rating_pos]
            residual_sum += ratings[rating_pos] - item_biases[item_id]

        if num_ratings_user > 0:
            user_biases[user_id] = (lambda_reg * residual_sum) / (lambda_reg * num_ratings_user + gamma_reg)

@njit(parallel=True)
def update_item_biases(item_biases, user_biases, user_index_for_rating, ratings,
                       item_start_index, lambda_reg, gamma_reg):

    num_items = item_biases.shape[0]

    for item_id in prange(num_items):

        residual_sum = 0.0
        start = item_start_index[item_id]
        end = item_start_index[item_id + 1]
        num_ratings_for_item = end - start

        for rating_pos in range(start, end):
            user_id = user_index_for_rating[rating_pos]
            residual_sum += ratings[rating_pos] - user_biases[user_id]

        if num_ratings_for_item > 0:
            item_biases[item_id] = (lambda_reg * residual_sum) / (lambda_reg * num_ratings_for_item + gamma_reg)

@njit(parallel=True)
def compute_sse(item_index_for_rating, ratings, user_biases, item_biases, user_start_index):
    total_squared_error = 0.0
    num_users = user_start_index.shape[0] - 1

    for user_id in prange(num_users):

        start = user_start_index[user_id]
        end = user_start_index[user_id + 1]
        user_bias = user_biases[user_id]

        for rating_pos in range(start, end):
            item_id = item_index_for_rating[rating_pos]
            prediction_error = ratings[rating_pos] - (user_bias + item_biases[item_id])
            total_squared_error += prediction_error * prediction_error

    return total_squared_error


@njit()
def compute_rmse(movie_idx, ratings, user_biases, item_biases, starts):
    sse = compute_sse(movie_idx, ratings, user_biases, item_biases, starts)
    total = ratings.shape[0]
    if total == 0:
        return 0.0
    return np.sqrt(sse / total)


def train_bias_only(data_by_user_train, data_by_item_train, data_by_user_test,
                    lambda_val=0.3, gamma=0.002, n_iters=20, verbose=True):
    #  to flatten data
    movie_idx_train, rating_train, user_starts_train = flatten_data(data_by_user_train)
    user_idx_train, rating_item_train, item_starts_train = flatten_data(data_by_item_train)
    movie_idx_test, rating_test, user_starts_test = flatten_data(data_by_user_test)

    n_users = len(data_by_user_train)
    n_items = len(data_by_item_train)

    user_biases = np.zeros(n_users, dtype=np.float32)
    item_biases = np.zeros(n_items, dtype=np.float32)

    loss_history = []
    train_rmse_history = []
    test_rmse_history = []

    for itt in range(n_iters):
        update_user_biases(user_biases, item_biases, movie_idx_train, rating_train, user_starts_train, lambda_val, gamma)
        update_item_biases(item_biases, user_biases, user_idx_train, rating_item_train, item_starts_train, lambda_val, gamma)
        sse = compute_sse(movie_idx_train, rating_train, user_biases, item_biases, user_starts_train)
        train_rmse = compute_rmse(movie_idx_train, rating_train, user_biases, item_biases, user_starts_train)
        test_rmse = compute_rmse(movie_idx_test, rating_test, user_biases, item_biases, user_starts_test)
        reg_sum = np.sum(user_biases.astype(np.float64)**2) + np.sum(item_biases.astype(np.float64)**2)
        loss = 0.5 * lambda_val * sse + 0.5 * gamma * reg_sum
        loss_history.append(loss)
        train_rmse_history.append(train_rmse)
        test_rmse_history.append(test_rmse)

        if verbose:
            print(f"Iter {itt+1:2d}: Loss={loss:.6f}, Train RMSE={train_rmse:.6f}, Test RMSE={test_rmse:.6f}")

    return user_biases, item_biases, loss_history, train_rmse_history, test_rmse_history

