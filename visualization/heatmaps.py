import matplotlib.pyplot as plt
import numpy as np
def heatmaps_3params_from_study(results_list, study):
    import numpy as np
    import matplotlib.pyplot as plt

    #  Retrieve explored unique values  
    taus   = sorted(list(set(r['tau'] for r in results_list)))
    gammas = sorted(list(set(r['gamma'] for r in results_list)))
    Ks     = sorted(list(set(r['K'] for r in results_list)))

    # Retrieve the best Optuna settings
    tau_best   = study.best_params['tau']
    gamma_best = study.best_params['gamma']
    K_best     = study.best_params['K']

    # Heatmap tau vs gamma (fixed K) 
    RMSE1 = np.zeros((len(taus), len(gammas)))
    for r in results_list:
        if r['K'] == K_best:
            i = taus.index(r['tau'])
            j = gammas.index(r['gamma'])
            RMSE1[i,j] = r['best_val_rmse']

    # Heatmap tau vs K (fixed gamma) 
    RMSE2 = np.zeros((len(taus), len(Ks)))
    for r in results_list:
        if r['gamma'] == gamma_best:
            i = taus.index(r['tau'])
            j = Ks.index(r['K'])
            RMSE2[i,j] = r['best_val_rmse']

    # Heatmap gamma vs K (fixed tau)
    RMSE3 = np.zeros((len(gammas), len(Ks)))
    for r in results_list:
        if r['tau'] == tau_best:
            i = gammas.index(r['gamma'])
            j = Ks.index(r['K'])
            RMSE3[i,j] = r['best_val_rmse']

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18,5))

    im = axes[0].imshow(RMSE1, origin='lower', cmap='viridis',
                        extent=[min(gammas), max(gammas), min(taus), max(taus)],
                        aspect='auto')
    axes[0].set_xlabel('gamma')
    axes[0].set_ylabel('tau')
    axes[0].set_title(f'Test RMSE (K={K_best})')
    fig.colorbar(im, ax=axes[0], label='RMSE')

    im = axes[1].imshow(RMSE2, origin='lower', cmap='viridis',
                        extent=[min(Ks), max(Ks), min(taus), max(taus)],
                        aspect='auto')
    axes[1].set_xlabel('K')
    axes[1].set_ylabel('tau')
    axes[1].set_title(f'Test RMSE (gamma={gamma_best})')
    fig.colorbar(im, ax=axes[1], label='RMSE')

    im = axes[2].imshow(RMSE3, origin='lower', cmap='viridis',
                        extent=[min(Ks), max(Ks), min(gammas), max(gammas)],
                        aspect='auto')
    axes[2].set_xlabel('K')
    axes[2].set_ylabel('gamma')
    axes[2].set_title(f'Test RMSE (tau={tau_best})')
    fig.colorbar(im, ax=axes[2], label='RMSE')

    plt.tight_layout()
    plt.savefig("pdf_reports/optimisation3params_best.pdf", format="pdf")
    plt.show()


def plot_all_params(results_list):
    fig, axes = plt.subplots(1,3, figsize=(15,4))
    params = ["tau", "gamma", "K"]
    for i, param in enumerate(params):
        vals = sorted(list(set(r[param] for r in results_list)))
        rmse = []
        for v in vals:
            rmse.append(np.mean([r['best_val_rmse'] for r in results_list if r[param]==v]))
        ax = axes[i]
        if param in ["tau","gamma"]:
            ax.semilogx(vals, rmse, 'o-', color='black', markerfacecolor='white', markeredgewidth=1.5)
        else:
            ax.plot(vals, rmse, 'o-', color='black', markerfacecolor='white', markeredgewidth=1.5)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_xlabel(param)
        ax.set_ylabel("Test RMSE")
        ax.set_title(f"{param} tuning")
        min_rmse = min(rmse)
        best_val = vals[rmse.index(min_rmse)]
        ax.axvline(best_val, color='red', linestyle='--', alpha=0.7)
        ax.text(best_val, min_rmse, f"  best={best_val:.3g}", color='red', fontsize=9, verticalalignment='bottom')
    plt.tight_layout()
    plt.savefig("pdf_reports/allparams_best.pdf", format="pdf")
    plt.show()