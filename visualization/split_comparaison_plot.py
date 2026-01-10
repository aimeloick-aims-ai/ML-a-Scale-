import matplotlib.pyplot as plt

def plot_split_comparaison(data, splits, figsize=(12, 4)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Train RMSE ---
    for key, cfg in splits.items():
        axes[0].plot(
            data[key]["train_rmse"],
            label=cfg["label"],
        )

    axes[0].set_title("Train RMSE")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("RMSE")
    axes[0].legend()

    # --- Test RMSE ---
    for key, cfg in splits.items():
        axes[1].plot(
            data[key]["test_rmse"],
            label=cfg["label"],
        )

    axes[1].set_title("Test RMSE")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()
    plt.savefig("pdf_report/plot_comparaison_split.pdf", format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.tight_layout()
    plt.show()
