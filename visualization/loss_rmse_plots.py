import matplotlib.pyplot as plt
import numpy as np
def loss_rmse(loss_hist, train_rmse_hist, test_rmse_hist, filename="plot.pdf"):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(loss_hist)
    plt.title("Loss")
    plt.grid(False)
    plt.subplot(1,2,2)
    plt.plot(train_rmse_hist, label="Train")
    plt.plot(test_rmse_hist, label="Test")
    plt.legend()
    #plt.title("RMSE (Train vs Test)")
    plt.grid(False)
    plt.savefig(
    filename,
    format="pdf"
    )
    plt.tight_layout()
    plt.show()
    
def plot_rmse_ks(rmse_history, Ks, filename=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(Ks)))
    
    for i, K in enumerate(Ks):
        train_rmse = rmse_history[K]["train"]
        test_rmse = rmse_history[K]["test"]
        iterations = range(1, len(train_rmse) + 1)
        
        ax1.plot(iterations, train_rmse, 
                linewidth=2, color=colors[i], 
                label=f"K={K} Train", alpha=0.8)
        
        ax1.plot(iterations, test_rmse, 
                linewidth=2, color=colors[i], 
                linestyle='--', label=f"K={K} Test", alpha=0.8)
    
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("RMSE", fontsize=12)
    ax1.set_title("Train/Test RMSE Evolution\n(gap shows overfitting)", fontsize=13, pad=10)
    ax1.legend(fontsize=8, loc='upper right', ncol=2)
    ax1.grid(alpha=0.3, linestyle='--')
    
    ax2 = axes[1]
    
    final_train = [rmse_history[K]["train"][-1] for K in Ks]
    final_test = [rmse_history[K]["test"][-1] for K in Ks]
    overfitting_gap = [test - train for train, test in zip(final_train, final_test)]
    
    x = np.arange(len(Ks))
    width = 0.35
    
    # Barres pour Train et Test
    bars1 = ax2.bar(x - width/2, final_train, width,  # noqa: F841
                    label='Train RMSE', color='#3498db', alpha=0.8)
    bars2 = ax2.bar(x + width/2, final_test, width,  # noqa: F841
                    label='Test RMSE', color='#e74c3c', alpha=0.8)
    
    # Annotations du gap (overfitting)
    for i, (train, test, gap) in enumerate(zip(final_train, final_test, overfitting_gap)):
        # Flèche montrant le gap
        ax2.annotate('', xy=(i + width/2, test), xytext=(i + width/2, train),
                    arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        # Valeur du gap
        ax2.text(i + width/2 + 0.15, (train + test) / 2, 
                f'+{gap:.3f}', fontsize=8, color='black', weight='bold')
    
    ax2.set_xlabel("Number of Latent Dimensions (K)", fontsize=12)
    ax2.set_ylabel("Final RMSE", fontsize=12)
    ax2.set_title("Final RMSE vs K\n(arrow = overfitting gap)", fontsize=13, pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'K={k}' for k in Ks])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"pdf_reports/{filename}", format="pdf", dpi=300, bbox_inches='tight')
    
    plt.show()
    

    print("\n" + "="*60)
    print("OVERFITTING ANALYSIS")
    print("="*60)
    
    for K in Ks:
        train = rmse_history[K]["train"][-1]
        test = rmse_history[K]["test"][-1]
        gap = test - train
        gap_percent = (gap / train) * 100
        print(f"K={K:2d} → Train: {train:.4f} | Test: {test:.4f} | Gap: {gap:.4f} ({gap_percent:.1f}%)")
    

    best_K = min(Ks, key=lambda k: rmse_history[k]["test"][-1])
    best_test_rmse = rmse_history[best_K]["test"][-1]
    
    print("\n" + "="*60)
    print("DIMINISHING RETURNS ANALYSIS")
    print("="*60)
    
    for i in range(1, len(Ks)):
        K_prev = Ks[i-1]
        K_curr = Ks[i]
        test_prev = rmse_history[K_prev]["test"][-1]
        test_curr = rmse_history[K_curr]["test"][-1]
        improvement = test_prev - test_curr
        improvement_percent = (improvement / test_prev) * 100
        
        status = "✅ Worth it" if improvement_percent > 0.5 else "⚠️ Diminishing"
        print(f"{K_prev}→{K_curr}: Δ={improvement:.4f} ({improvement_percent:.2f}%) {status}")
    
    print(f"\n🎯 Best K: {best_K} (Test RMSE: {best_test_rmse:.4f})")



def plot_overfitting_comparison(rmse_history, K1, K2, filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for K, color, marker in [(K1, '#3498db', 'o'), (K2, '#e74c3c', 's')]:
        train = rmse_history[K]["train"]
        test = rmse_history[K]["test"]
        iterations = range(1, len(train) + 1)
        
        # Train
        ax.plot(iterations, train, linewidth=2.5, color=color, 
               label=f"K={K} Train", marker=marker, markersize=4, markevery=2)
        
        # Test
        ax.plot(iterations, test, linewidth=2.5, color=color, 
               linestyle='--', label=f"K={K} Test", marker=marker, 
               markersize=4, markevery=2, alpha=0.7)
        
        # Zone de shading pour montrer le gap
        ax.fill_between(iterations, train, test, 
                        color=color, alpha=0.15, 
                        label=f"K={K} Overfitting Gap")
    
    ax.set_xlabel("Iteration", fontsize=13)
    ax.set_ylabel("RMSE", fontsize=13)
    ax.set_title(f"Overfitting Comparison: K={K1} vs K={K2}", 
                fontsize=14, weight='bold', pad=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')
    
    # Annotations finales
    for K, color, y_pos in [(K1, '#3498db', 0.95), (K2, '#e74c3c', 0.90)]:
        gap = rmse_history[K]["test"][-1] - rmse_history[K]["train"][-1]
        ax.text(0.02, y_pos, f"K={K} final gap: {gap:.4f}", 
               transform=ax.transAxes, fontsize=10, 
               bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(f"pdf_reports/{filename}", format="pdf", dpi=300, bbox_inches='tight')
    
    plt.show()
