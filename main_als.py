from data.load_ratings import structure_data
from data.split import data_split
from visualization.loss_rmse_plots import loss_rmse
from models.als_core import   train_als
#from visualization.rmse_plots import plot_rmse_history

_, _, _, _, data_by_movie, data_by_user = structure_data()
data_by_user_train, data_by_user_test, data_by_movie_train, data_by_movie_test  = data_split(data_by_movie, data_by_user) 


user_vectors, biases_vectors , user_biases, item_biases, loss_hist, train_rmse, test_rmse = train_als(
    data_by_user_train, data_by_movie_train, data_by_user_test,
    factor_number= 15, lambda_val= 0.1, gamma=0.04, tau= 1.9, n_iters=20
)
loss_rmse(loss_hist, train_rmse, test_rmse, filename ="pdf_reports/rmse_and_loss_full_model.pdf")
