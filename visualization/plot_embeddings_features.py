import matplotlib.pyplot as plt
import pandas as pd

movies_df = pd.read_csv("ml-32m/movies.csv")
def plot_genre_embeddings(f, save_path="plot_feature_embedding.pdf"):
    all_genres = set()
    for genres_str in movies_df['genres']:
        if isinstance(genres_str, str):
            all_genres.update(genres_str.split('|'))
    all_genres.discard('(no genres listed)')

    feature_names = sorted(list(all_genres))

    # --- 2) Genre styles ---
    genre_styles = {
        'Action': {'color': '#e74c3c', 'marker': 'x'},
        'Adventure': {'color': '#e67e22', 'marker': 'D'},
        'Animation': {'color': '#9b59b6', 'marker': 'P'},
        'Children': {'color': '#3498db', 'marker': 'o'},
        'Comedy': {'color': '#f39c12', 'marker': 'o'},
        'Crime': {'color': '#34495e', 'marker': 's'},
        'Documentary': {'color': '#16a085', 'marker': '^'},
        'Drama': {'color': '#2980b9', 'marker': 'v'},
        'Fantasy': {'color': '#27ae60', 'marker': '*'},
        'Horror': {'color': '#c0392b', 'marker': 'X'},
        'Musical': {'color': '#e91e63', 'marker': 'p'},
        'Mystery': {'color': '#795548', 'marker': 'h'},
        'Romance': {'color': '#ec407a', 'marker': '<'},
        'Sci-Fi': {'color': '#00bcd4', 'marker': '>'},
        'Thriller': {'color': '#7e57c2', 'marker': 'd'},
        'War': {'color': '#8d6e63', 'marker': '8'},
        'Western': {'color': '#ff9800', 'marker': 's'},
    }
    default_style = {'color': '#95a5a6', 'marker': 'o'}

    # --- 3) Plot ---
    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')

    for i, genre_name in enumerate(feature_names[:len(f)]):
        x, y = f[i, 0], f[i, 1]
        style = genre_styles.get(genre_name, default_style)

        ax.scatter(x, y, c=style['color'], marker=style['marker'],
                   s=80, alpha=0.75, edgecolors='none', zorder=2)
        ax.text(x, y, f' {genre_name}', fontsize=8, ha='left', va='center',
                color='#2c3e50', alpha=0.85, zorder=3)

    ax.axhline(0, linewidth=0.8, color='#bdc3c7', alpha=0.5, zorder=1)
    ax.axvline(0, linewidth=0.8, color='#bdc3c7', alpha=0.5, zorder=1)

    ax.set_title("Embeddings", fontsize=24, weight='normal', color='#2c3e50', pad=20, loc='left')
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor('#ecf0f1')
        spine.set_linewidth(1)

    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.show()

