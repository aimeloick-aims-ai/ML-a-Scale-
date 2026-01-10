from data.load_ratings import structure_data
from data.split import data_split
from models.als_features import  train_als_features
from data.features import load_features
from visualization.loss_rmse_plots import loss_rmse
#from visualization.rmse_plots import plot_rmse_history

_, _, movieid_to_idx, _, data_by_movie, data_by_user = structure_data()
data_by_user_train, data_by_user_test, data_by_movie_train, data_by_movie_test  = data_split(data_by_movie, data_by_user) 
item_features, features_having_items = load_features(data_by_movie, movieid_to_idx)

U, V, f, user_biases, item_biases, loss_hist, train_rmse_history, test_rmse_history = train_als_features(
    data_by_user_train, data_by_movie_train, data_by_user_test,
    item_features,
    features_having_items,
    factor_number= 15, 
    lambda_val=0.1,   
    gamma= 0.04,       
    tau= 2.1,         
    n_iters=20, 
    verbose=True
)

loss_rmse(loss_hist, train_rmse_history, test_rmse_history, filename ="pdf_reports/rmse_and_loss_full_features_model.pdf")