import numpy as np
import matplotlib.pyplot as plt


def compute_degrees(data):
    """
    Compute degrees (number of ratings per entity)
    Args:
        data (list of lists): ratings per movie or per user
    Returns:
        degrees (np.array): array of counts (ignoring zero ratings)
    """
    return np.array([len(ratings) for ratings in data if len(ratings) > 0])

def get_distribution(degrees):
    """
    Compute the distribution function P(k)
    Args:
        degrees (np.array): array of counts
    Returns:
        unique_k (np.array): unique degrees
        freq (np.array): counts of each degree
    """
    unique_k, counts = np.unique(degrees, return_counts=True)
    return unique_k, counts

def plot_power_law(movie_k, movie_p, user_k, user_p, save_path=None):
    """
    Plot the power-law distribution for movies and users.
    Args:
        movie_k, movie_p: degrees and counts for movies
        user_k, user_p: degrees and counts for users
        save_path (str, optional): path to save the figure as PDF
    """
    plt.figure(figsize=(7,5))
    plt.scatter(movie_k, movie_p, s=10, color='tab:blue', label='Movies (ratings per movie)', alpha=0.7)
    plt.scatter(user_k, user_p, s=10, color='tab:green', label='Users (ratings per user)', alpha=0.7)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of ratings (k)')
    plt.ylabel('P(k)')
    plt.title('Power-law Distribution (Movies vs Users)')
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    
    plt.show()



