# ratings_utils.py
import numpy as np
import matplotlib.pyplot as plt

def flatten_ratings(data_by_user):
    """
    Flatten all user ratings into a 1D numpy array.
    """
    return np.array([float(rating_value) for user_data in data_by_user for _, rating_value in user_data])


def plot_rating_distribution(ratings, bins=None, save_path=None, title=None):
    """
    Plot histogram of ratings.
    """
    if bins is None:
        bins = np.arange(0.25, 5.5, 0.5)

    plt.figure(figsize=(7,5))
    counts, _, _ = plt.hist(ratings, bins=bins, edgecolor='black', alpha=0.7, rwidth=0.9)
    
    plt.xlabel("Rating Value")
    plt.ylabel("Frequency")
    plt.xticks(np.arange(0.5, 5.5, 0.5))
    plt.title(title if title else "Distribution of Movie Ratings")
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    
    plt.show()


def plot_ratings_by_user_movie(data_by_user, data_by_movie, save_path=None):
    """
    Plot histograms of ratings per user and ratings per movie.
    """
    ratings_per_user = [len(user_data) for user_data in data_by_user]
    ratings_per_item = [len(item_data) for item_data in data_by_movie]

    plt.figure(figsize=(12,5))

    # Ratings per user
    plt.subplot(1,2,1)
    plt.hist(ratings_per_user, bins=50, color='skyblue', edgecolor='black')
    plt.title("Ratings per User")
    plt.xlabel("Number of ratings")
    plt.ylabel("Number of users")
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.5)

    # Ratings per movie
    plt.subplot(1,2,2)
    plt.hist(ratings_per_item, bins=50, color='salmon', edgecolor='black')
    plt.title("Ratings per Movie")
    plt.xlabel("Number of ratings")
    plt.ylabel("Number of movies")
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    
    plt.show()
