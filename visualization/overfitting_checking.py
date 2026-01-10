
import matplotlib.pyplot as plt
import numpy as np
def plot_gap_vs_counts(results, K):
    g = results['gap_u']
    counts = results['counts_train']
    mask = (~np.isnan(g)) & (counts > 0)
    plt.figure(figsize=(6,4))
    plt.scatter(counts[mask], g[mask], alpha=0.4)
    plt.xscale('log')
    plt.xlabel('Number of ratings (log scale)')
    plt.ylabel('gap (test - train)')
    plt.title(f'gap vs number of ratings per user (K={K})')
    plt.axhline(0, color='k', linestyle='--')
    plt.tight_layout()
    plt.savefig("pdf_reports/plot_gap_vs_counts.pdf", format="pdf")
    plt.show()


def plot_avg_gap_by_degree(results, K, max_bin=10):
    counts = results['counts_train']
    gap = results['gap_u']

    # Binning
    counts_binned = np.minimum(counts, max_bin)
    bins = np.arange(max_bin + 2)
    avg_gap = np.array([np.nanmean(gap[counts_binned == b]) if np.any(counts_binned == b) else np.nan
                        for b in bins])

    plt.figure(figsize=(8,5))
    plt.plot(bins, avg_gap, marker='o', linestyle='-')
    plt.xlabel("User degree (train ratings, binned)")
    plt.ylabel("Average Test RMSE - Train RMSE (gap)")
    plt.title(f"Average Overfitting Gap by User Degree (K={K})")
    plt.tight_layout()
    plt.savefig("pdf_reports/plot_avg_gap_by_degree.pdf", format="pdf")
    plt.show()
