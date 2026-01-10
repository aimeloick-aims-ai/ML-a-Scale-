from models.als_core import train_als
from visualization.loss_rmse_plots import plot_rmse_ks,plot_overfitting_comparison


def run_als_experiment_ks(data_by_user_train, data_by_user_test, \
    data_by_movie_train, data_by_movie_test,
    Ks,
    lambda_val=0.0630957344480193,
    gamma=0.0630957344480193,
    tau=1.5848931924611136,
    n_iters=10,
    plot_filename="k_evolution_checking",
    verbose=True
):

    rmse_history = {}

    for K in Ks:
        if verbose:
            print(f"\nTraining ALS with K = {K}")

        U, V, ub, ib, loss_hist, tr_rmse_history, te_rmse_history = train_als(
            data_by_user_train,
            data_by_movie_train,
            data_by_user_test,
            factor_number=K,
            lambda_val=lambda_val,
            gamma=gamma,
            tau=tau,
            n_iters=n_iters,
            verbose=verbose
        )

        rmse_history[K] = {
            "train": tr_rmse_history,
            "test": te_rmse_history
        }

    plot_rmse_ks(rmse_history, Ks, filename=plot_filename)
    
    # Comparaison K=10 vs K=20
    if 10 in rmse_history and 20 in rmse_history:
        plot_overfitting_comparison(rmse_history, K1=10, K2=20, 
                                   filename="overfitting_K10_vs_K20.pdf")
    return rmse_history
