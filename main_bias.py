from data.load_ratings import structure_data
from data.split import data_split
from visualization.loss_rmse_plots import loss_rmse
from models.bias_only import   train_bias_only

_, _, _, _,data_by_movie, data_by_user = structure_data()
data_by_user_train, data_by_user_test, data_by_movie_train, data_by_movie_test  = data_split(data_by_movie, data_by_user) 


user_biases, item_biases, loss_hist, train_rmse_hist, test_rmse_hist = train_bias_only(
    data_by_user_train, data_by_movie_train, data_by_user_test,
    lambda_val=0.3, gamma=0.002, n_iters=20)
loss_rmse(loss_hist, train_rmse_hist, test_rmse_hist,filename ="pdf_reports/rmse_and_loss_bias_model.pdf")