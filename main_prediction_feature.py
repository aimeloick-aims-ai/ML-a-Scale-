import pickle
from data.load_ratings import structure_data
from data.split import data_split
from data.features import load_features
from models.als_features import train_als_features
from prediction.dummy_user import build_dummy_rating
from prediction.recommend import compute_dummy_user,predict_dummy_user
from visualization.loss_rmse_plots import loss_rmse
from polarisation.top_polarisation import compute_polarization, select_movies_by_polarization, build_polarization_dataframe
from visualization.polarisation import plot_top_polarization, plot_polarization_vs_count, plot_polarization_with_std
from visualization.prediction import  plot_dummy_user_predictions

idx_to_movie, _, movieid_to_idx, _, data_by_movie, data_by_user = structure_data()
data_by_user_train, data_by_user_test, data_by_movie_train, data_by_movie_test = data_split(data_by_movie, data_by_user)
item_features, features_having_items = load_features(data_by_movie, movieid_to_idx)


U, V, f, user_biases, item_biases, loss_hist, train_rmse_history, test_rmse_history = train_als_features(
    data_by_user_train, data_by_movie_train, data_by_user_test,
    item_features,
    features_having_items,
    lambda_val=0.1,
    gamma=0.04,
    tau= 2.1,
    n_iters=20,
    verbose=True
)


loss_rmse(loss_hist, train_rmse_history, test_rmse_history, filename="pdf_reports/rmse_and_loss_full_features_model.pdf")


movie_title = "Lord of the Rings: The Two Towers, The (2002)"

dummy_ratings, dummy_movie_idx, _, movieid_to_title = build_dummy_rating(
    movie_title,
    movieid_to_idx,
    rating_value=5.0,
    movies_csv_path="ml-32m/movies.csv"
)

print("Dummy ratings:", dummy_ratings)

dummy_user_factors, dummy_user_bias = compute_dummy_user(V, item_biases, dummy_ratings, K=15,
    lambda_val=0.1,
    tau= 2.1,
    gamma_bias=0.04
)

print("Dummy user latent factors:", dummy_user_factors)
print("Dummy user bias:", dummy_user_bias)

predictions_full_bias, predictions_scaled_bias = predict_dummy_user(
    dummy_user_factors,
    V,
    item_biases,
    bias_scale=0.05
)

print("\n🎬 Top 10 Recommendations (bias complet)")
for item_idx, score in predictions_full_bias[:10]:
    movie_id = list(movieid_to_idx.keys())[list(movieid_to_idx.values()).index(item_idx)]
    title = movieid_to_title.get(movie_id, f"MovieIdx {item_idx}")
    print(f"- {title} | Score prédit : {score:.4f}")

print("\n🎬 Top 10 Recommendations (bias pondéré 0.05)")
for item_idx, score in predictions_scaled_bias[:10]:
    movie_id = list(movieid_to_idx.keys())[list(movieid_to_idx.values()).index(item_idx)]
    title = movieid_to_title.get(movie_id, f"MovieIdx {item_idx}")
    print(f"- {title} | Score prédit : {score:.4f}")



MIN_RATINGS_POLARIZING = 1
MIN_RATINGS_NONPOLARIZING = 1
TOP_CANDIDATES = 200_000_000
TOP_K = 10

plot_dummy_user_predictions(predictions_full_bias, predictions_scaled_bias, idx_to_movie, movieid_to_title, data_by_movie_train, dummy_movie_idx, filename_prefix="plot_dummy_user")

polarization = compute_polarization(V)

polarizing_idx = select_movies_by_polarization(
    polarization,
    idx_to_movie,
    movieid_to_idx,
    data_by_movie_train,
    top_k=TOP_K,
    min_ratings=MIN_RATINGS_POLARIZING,
    most_polarizing=True
)

nonpolarizing_idx = select_movies_by_polarization(
    polarization,
    idx_to_movie,
    movieid_to_idx,
    data_by_movie_train,
    top_k=TOP_K,
    min_ratings=MIN_RATINGS_NONPOLARIZING,
    most_polarizing=False
)

df_stats = build_polarization_dataframe(
    polarizing_idx,
    nonpolarizing_idx,
    polarization,
    idx_to_movie,
    movieid_to_idx,
    data_by_movie_train,
    movieid_to_title
)

print(df_stats)

plot_top_polarization(df_stats)
plot_polarization_vs_count(polarization, data_by_movie_train)
plot_polarization_with_std(polarization, data_by_movie_train)


als_model = {
    "user_factors":     U,
    "item_factors": V,
    "user_biases": user_biases,
    "movieid_to_idx" : movieid_to_idx,
    "item_biases": item_biases,
    "loss_history": loss_hist,
    "train_rmse_history": train_rmse_history,
    "test_rmse_history": test_rmse_history,
    "idx_to_movie": idx_to_movie
}

with open("als_model.pkl", "wb") as f:  # noqa: F811
    pickle.dump(als_model, f)

print("✅ ALS model saved as als_model.pkl")
