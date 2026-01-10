import numpy as np
import pandas as pd


def compute_polarization(item_factors):
    """Compute polarization as the L2 norm of latent item vectors."""
    return np.linalg.norm(item_factors, axis=1)


def select_movies_by_polarization(
    polarization,
    idx_to_movie,
    movieid_to_idx,
    data_by_movie_train,
    top_k=10,
    min_ratings=1,
    most_polarizing=True,
    top_candidates=200_000_000
):
    """
    Select top-k polarizing or non-polarizing movies
    based on polarization score and minimum ratings.
    """
    if most_polarizing:
        sorted_idx = np.argsort(-polarization)[:top_candidates]
    else:
        sorted_idx = np.argsort(polarization)[:top_candidates]

    selected = []
    for idx in sorted_idx:
        idx = int(idx)
        movie_id = idx_to_movie[idx]
        movie_idx = movieid_to_idx[movie_id]

        if len(data_by_movie_train[movie_idx]) >= min_ratings:
            selected.append(idx)

        if len(selected) == top_k:
            break

    return selected


def compute_movie_statistics(
    movie_indices,
    polarization,
    idx_to_movie,
    movieid_to_idx,
    data_by_movie_train,
    movieid_to_title,
    label_type
):
    """Compute rating statistics for selected movies."""
    rows = []

    for idx in movie_indices:
        movie_id = idx_to_movie[idx]
        movie_idx = movieid_to_idx[movie_id]
        ratings = np.array([r[1] for r in data_by_movie_train[movie_idx]])

        rows.append({
            "Movie": movieid_to_title.get(movie_id, f"Movie {idx}"),
            "Polarization": polarization[idx],
            "Mean Rating": ratings.mean(),
            "Std Rating": ratings.std(),
            "Min Rating": ratings.min(),
            "Max Rating": ratings.max(),
            "Count": len(ratings),
            "Type": label_type
        })

    return rows
def build_polarization_dataframe(
    polarizing_idx,
    nonpolarizing_idx,
    polarization,
    idx_to_movie,
    movieid_to_idx,
    data_by_movie_train,
    movieid_to_title
):
    """Build a DataFrame with statistics for polarizing and non-polarizing movies."""
    rows = []

    rows += compute_movie_statistics(
        polarizing_idx, polarization,
        idx_to_movie, movieid_to_idx,
        data_by_movie_train, movieid_to_title,
        label_type="Polarizing"
    )

    rows += compute_movie_statistics(
        nonpolarizing_idx, polarization,
        idx_to_movie, movieid_to_idx,
        data_by_movie_train, movieid_to_title,
        label_type="Non-Polarizing"
    )

    return pd.DataFrame(rows)
