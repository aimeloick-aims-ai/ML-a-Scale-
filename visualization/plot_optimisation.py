import numpy as np
import matplotlib.pyplot as plt
from models.als_core import train_als
from data.load_ratings import structure_data
from data.split import data_split

idx_to_movie, _, movieid_to_idx, _, data_by_movie, data_by_user = structure_data()
data_by_user_train, data_by_user_test, data_by_movie_train, data_by_movie_test = data_split(data_by_movie, data_by_user)

def heatmap_around_best(data_by_user_train, data_by_movie_train, 
                        data_by_user_test, data_by_movie_test,
                        fixed_param='K', fixed_value=15,
                        tau_center=1.9, gamma_center=0.04,
                        n_points=5, range_factor=3):
    """
    Fix one parameter and explore the other 2 around given values.
    
    Args:
        fixed_param: 'K', 'tau' or 'gamma' - the parameter to fix
        fixed_value: value of the fixed parameter
        tau_center, gamma_center: central values for exploration
        n_points: number of points in each direction (e.g., 5x5 grid)
        range_factor: multiplicative factor for the exploration range
    """
    
    if fixed_param == 'K':
        # Fix K, explore tau and gamma
        tau_range = np.logspace(np.log10(tau_center/range_factor), 
                                np.log10(tau_center*range_factor), n_points)
        gamma_range = np.logspace(np.log10(gamma_center/range_factor), 
                                  np.log10(gamma_center*range_factor), n_points)
        
        rmse_grid = np.zeros((n_points, n_points))
        
        for i, tau in enumerate(tau_range):
            for j, gamma in enumerate(gamma_range):
                _, _, _, _, _, _, val_rmse = train_als(
                    data_by_user_train=data_by_user_train,
                    data_by_item_train=data_by_movie_train,
                    data_by_user_test=data_by_user_test,
                    factor_number=int(fixed_value),
                    lambda_val=1,
                    gamma=gamma,
                    tau=tau,
                    n_iters=15,
                    verbose=False
                )
                rmse_grid[i, j] = val_rmse[-1]
        
        # Plot
        plt.figure(figsize=(10, 8))
        im = plt.imshow(rmse_grid, aspect='auto', origin='lower', cmap='RdYlGn_r')
        plt.colorbar(im, label='RMSE Test')
        
        plt.xticks(range(n_points), [f'{g:.3f}' for g in gamma_range], rotation=45)
        plt.yticks(range(n_points), [f'{t:.2f}' for t in tau_range])
        
        plt.xlabel('γ (gamma)', fontsize=12)
        plt.ylabel('τ (tau)', fontsize=12)
        plt.title(f'RMSE with K={fixed_value} fixed', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'heatmap_K{fixed_value}_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    elif fixed_param == 'tau':
        # Fix tau, explore K and gamma
        K_range = np.linspace(max(5, fixed_value-10), fixed_value+10, n_points, dtype=int)
        gamma_range = np.logspace(np.log10(gamma_center/range_factor), 
                                  np.log10(gamma_center*range_factor), n_points)
        
        rmse_grid = np.zeros((n_points, n_points))
        
        for i, K in enumerate(K_range):
            for j, gamma in enumerate(gamma_range):
                _, _, _, _, _, _, val_rmse = train_als(
                    data_by_user_train=data_by_user_train,
                    data_by_item_train=data_by_movie_train,
                    data_by_user_test=data_by_user_test,
                    factor_number=int(K),
                    lambda_val=1,
                    gamma=gamma,
                    tau=fixed_value,
                    n_iters=15,
                    verbose=False
                )
                rmse_grid[i, j] = val_rmse[-1]
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(rmse_grid, aspect='auto', origin='lower', cmap='RdYlGn_r')
        plt.colorbar(im, label='RMSE Test')
        
        plt.xticks(range(n_points), [f'{g:.3f}' for g in gamma_range], rotation=45)
        plt.yticks(range(n_points), K_range)
        
        plt.xlabel('γ (gamma)', fontsize=12)
        plt.ylabel('K (latent factors)', fontsize=12)
        plt.title(f'RMSE with τ={fixed_value:.2f} fixed', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'heatmap_tau{fixed_value:.2f}_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    elif fixed_param == 'gamma':
        # Fix gamma, explore K and tau
        K_range = np.linspace(max(5, fixed_value-10), fixed_value+10, n_points, dtype=int)
        tau_range = np.logspace(np.log10(tau_center/range_factor), 
                               np.log10(tau_center*range_factor), n_points)
        
        rmse_grid = np.zeros((n_points, n_points))
        
        for i, K in enumerate(K_range):
            for j, tau in enumerate(tau_range):
                _, _, _, _, _, _, val_rmse = train_als(
                    data_by_user_train=data_by_user_train,
                    data_by_item_train=data_by_movie_train,
                    data_by_user_test=data_by_user_test,
                    factor_number=int(K),
                    lambda_val=1,
                    gamma=fixed_value,
                    tau=tau,
                    n_iters=15,
                    verbose=False
                )
                rmse_grid[i, j] = val_rmse[-1]
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(rmse_grid, aspect='auto', origin='lower', cmap='RdYlGn_r')
        plt.colorbar(im, label='RMSE Test')
        
        plt.xticks(range(n_points), [f'{t:.2f}' for t in tau_range], rotation=45)
        plt.yticks(range(n_points), K_range)
        
        plt.xlabel('τ (tau)', fontsize=12)
        plt.ylabel('K (latent factors)', fontsize=12)
        plt.title(f'RMSE with γ={fixed_value:.3f} fixed', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'heatmap_gamma{fixed_value:.3f}_fixed.png', dpi=300, bbox_inches='tight')
        plt.show()

