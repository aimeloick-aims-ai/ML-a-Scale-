import optuna
from tqdm import tqdm
from data.load_ratings import structure_data
from data.split import data_split
from models.als_core import   train_als
from visualization.heatmaps import heatmaps_3params_from_study, plot_all_params
from visualization.overfitting_checking import plot_gap_vs_counts 
from tuning.overfitting_checking import evaluate_over_K
from tuning.hierarchical import run_als_experiment_ks

# load data split
_, _, _, _, data_by_movie, data_by_user = structure_data()
data_by_user_train, data_by_user_test, data_by_movie_train, data_by_movie_test  = data_split(data_by_movie, data_by_user)

# Store results  plotting
results_list = []

def objective(trial):
    tau   = trial.suggest_float("tau", 1e-2, 2e1, log=True)
    gamma = trial.suggest_float("gamma", 1e-3, 1e1, log=True)
    K     = trial.suggest_int("K", 10, 20)

    n_iters = 0
    patience = 2
    best_val_rmse = float("inf")
    best_iter = 0
    train_rmse_history = []
    val_rmse_history = []

    # Early stopping
    for it in range(n_iters):
        user_vectors, biases_vectors, user_biases, item_biases, loss_hist, train_rmse, val_rmse = train_als(
                                                                                                            data_by_user_train=data_by_user_train,
                                                                                                            data_by_item_train=data_by_movie_train,
                                                                                                            data_by_user_test=data_by_user_test,
                                                                                                            factor_number=int(K),   
                                                                                                            lambda_val=1,
                                                                                                            gamma=gamma,
                                                                                                            tau=tau,
                                                                                                            n_iters=15,
                                                                                                            verbose=False
                                                                                                        )

        train_rmse_history.append(train_rmse[-1])
        val_rmse_history.append(val_rmse[-1])

        if val_rmse[-1] < best_val_rmse:
            best_val_rmse = val_rmse[-1]
            best_iter = it
        elif it - best_iter >= patience:
            break

    results_list.append({
        "tau": tau,
        "gamma": gamma,
        "K": K,
        "train_rmse": train_rmse_history,
        "val_rmse": val_rmse_history,
        "best_val_rmse": best_val_rmse
    })

    return best_val_rmse


study = optuna.create_study(direction="minimize")
n_trials = 50
for _ in tqdm(range(n_trials), desc="Optimisation Optuna"):
    study.optimize(objective, n_trials=1)

print("Meilleur tau:", study.best_params["tau"])
print("Meilleur gamma:", study.best_params["gamma"])
print("Meilleur K:", study.best_params["K"])
print("RMSE validation:", study.best_value)



heatmaps_3params_from_study(results_list, study)
plot_all_params(results_list)



Ks = [0, 4, 10, 15, 20 , 32]

rmse_history = run_als_experiment_ks(data_by_user_train, data_by_user_test, \
    data_by_movie_train, data_by_movie_test,
    Ks=Ks,
    n_iters=20,
    plot_filename="k_evolution_checking"
)

results = evaluate_over_K(train_als, 20, data_by_user_train, data_by_movie_train, data_by_user_test,
                        lambda_val=0.1, gamma=0.04, tau=1.9, n_iters=20)

plot_gap_vs_counts(results, 20)





