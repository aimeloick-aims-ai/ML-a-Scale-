import pandas as pd
import matplotlib.pyplot as plt
movies_df = pd.read_csv("ml-32m/movies.csv")
movieid_to_genres = dict(zip(movies_df['movieId'], movies_df['genres']))


def idx_to_title(idx, idx_to_movie, movieid_to_title):
    movie_id = idx_to_movie[idx]
    return movieid_to_title.get(movie_id, f"Movie {idx}")

def idx_to_primary_genre(idx, idx_to_movie, movieid_to_title,movieid_to_genres):
    movie_id = idx_to_movie[idx]
    genres_str = movieid_to_genres.get(movie_id, "Unknown")
    return genres_str.split('|')[0] if '|' in genres_str else genres_str
def plot_embedding_movie(item_factors,idx_to_movie,data_by_movie, movieid_to_title):
        genre_styles = {
            'Action': {'color': '#d62728', 'marker': 'x', 'label': 'Action'},
            'Adventure': {'color': '#ff7f0e', 'marker': 'D', 'label': 'Adventure'},
            'Animation': {'color': '#9467bd', 'marker': 'P', 'label': 'Animation'},
            'Children': {'color': '#e377c2', 'marker': 'H', 'label': 'Children'},
            'Comedy': {'color': '#bcbd22', 'marker': 'o', 'label': 'Comedy'},
            'Crime': {'color': '#8c564b', 'marker': 's', 'label': 'Crime'},
            'Documentary': {'color': '#17becf', 'marker': '^', 'label': 'Documentary'},
            'Drama': {'color': '#1f77b4', 'marker': 'v', 'label': 'Drama'},
            'Fantasy': {'color': '#2ca02c', 'marker': '*', 'label': 'Fantasy'},
            'Horror': {'color': '#000000', 'marker': 'X', 'label': 'Horror'},
            'Musical': {'color': '#ff6ec7', 'marker': 'p', 'label': 'Musical'},
            'Mystery': {'color': '#795548', 'marker': 'h', 'label': 'Mystery'},
            'Romance': {'color': '#f06292', 'marker': '<', 'label': 'Romance'},
            'Sci-Fi': {'color': '#00bcd4', 'marker': '>', 'label': 'Sci-Fi'},
            'Thriller': {'color': '#4a148c', 'marker': 'd', 'label': 'Thriller'},
            'War': {'color': '#5d4037', 'marker': '8', 'label': 'War'},
            'Western': {'color': '#ff9800', 'marker': 's', 'label': 'Western'},
        }

        default_style = {'color': '#999999', 'marker': 'o', 'label': 'Other'}

        n_labels = 100

        movie_popularity = []
        for idx in range(len(item_factors)):
            movie_id = idx_to_movie[idx]
            n_ratings = len(data_by_movie[movie_id]) if movie_id < len(data_by_movie) else 0
            movie_popularity.append((idx, n_ratings))

        movie_popularity.sort(key=lambda x: x[1], reverse=True)
        movies_to_label = [idx for idx, _ in movie_popularity[:n_labels]]

        movies_by_genre = {}
        for idx in movies_to_label:
            genre = idx_to_primary_genre(idx, idx_to_movie, movieid_to_title,movieid_to_genres)
            if genre not in movies_by_genre:
                movies_by_genre[genre] = []
            movies_by_genre[genre].append(idx)


        fig, ax = plt.subplots(figsize=(16, 12))

        ax.scatter(item_factors[:, 0], item_factors[:, 1], 
                alpha=0.1, s=5, c='lightgray', zorder=1)

        plotted_genres = set()
        for genre, movie_indices in movies_by_genre.items():
            style = genre_styles.get(genre, default_style)
            
            for idx in movie_indices:
                x, y = item_factors[idx, 0], item_factors[idx, 1]

                if genre not in plotted_genres:
                    ax.scatter(x, y, c=style['color'], marker=style['marker'], 
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                            label=style['label'], zorder=2)
                    plotted_genres.add(genre)
                else:
                    ax.scatter(x, y, c=style['color'], marker=style['marker'], 
                            s=100, alpha=0.7, edgecolors='black', linewidth=0.5, zorder=2)

                ax.text(x, y, idx_to_title(idx, idx_to_movie, movieid_to_title), 
                    fontsize=5, ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                alpha=0.6, edgecolor='none'), zorder=3)

        ax.axhline(0, linewidth=0.5, color='black', alpha=0.3)
        ax.axvline(0, linewidth=0.5, color='black', alpha=0.3)
        ax.set_xlabel("Latent Factor 1", fontsize=13)
        ax.set_ylabel("Latent Factor 2", fontsize=13)
        ax.grid(alpha=0.2)
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                frameon=True, fontsize=10, markerscale=1.2)

        plt.tight_layout()
        plt.savefig("als_2d_by_genre.png", dpi=300, bbox_inches='tight')
        plt.show()

