import matplotlib.pyplot as plt
import numpy as np
from prediction.recommend import compute_overlap_stats

def extract_plot_data(top_list, idx_to_movie, movieid_to_title, data_by_movie_train, dummy_movie_idx):
    titles = []
    proportions = []
    for movie_idx, _ in top_list[:10]:
        mid = idx_to_movie[movie_idx]
        title = movieid_to_title.get(mid, f"MovieIdx {movie_idx}")
        stats = compute_overlap_stats(movie_idx, data_by_movie_train, dummy_movie_idx)
        titles.append(title[:20] + "…")
        proportions.append(100 * stats["prop_lotr_overlap"])
    return titles, proportions

def plot_dummy_user_predictions(predictions_std, predictions_005, idx_to_movie, movieid_to_title, data_by_movie_train, dummy_movie_idx, filename_prefix="plot_dummy_user"):
    titles_std, prop_std = extract_plot_data(predictions_std, idx_to_movie, movieid_to_title, data_by_movie_train, dummy_movie_idx)
    titles_005, prop_005 = extract_plot_data(predictions_005, idx_to_movie, movieid_to_title,  data_by_movie_train, dummy_movie_idx)

    # --- Barplot top 10 ---
    plt.figure(figsize=(14, 6))
    x = np.arange(10)
    width = 0.35

    plt.bar(x - width/2, prop_std, width, label='Standard')
    plt.bar(x + width/2, prop_005, width, label='0.05 bias')

    xticks_labels = [f"{titles_std[i]}\n{titles_005[i]}" for i in range(len(x))]
    plt.xticks(x, xticks_labels, rotation=90, ha='center', fontsize=12)
    plt.ylabel("% overlap with LOTR-lovers")
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(f"pdf_reports/{filename_prefix}_top10_overlap.pdf", format="pdf")
    plt.show()

    # --- Histogramme  scores ---
    predicted_scores = [score for movie_idx, score in predictions_std]
    plt.figure(figsize=(10, 6))
    plt.hist(predicted_scores, bins=50, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Predicted Scores for Dummy User')
    plt.xlabel('Predicted Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(f"pdf_reports/{filename_prefix}_score_distribution.pdf", format="pdf")
    plt.show()