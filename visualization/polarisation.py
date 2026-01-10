
import matplotlib.pyplot as plt
import numpy as np


def plot_top_polarization(df, filename="pdf_reports/Plot_top_polarization"):
    """Horizontal bar plot of polarization scores."""
    colors = df["Type"].map({
        "Polarizing": "red",
        "Non-Polarizing": "green"
    })

    plt.figure(figsize=(14, 6))
    plt.barh(range(len(df)), df["Polarization"], color=colors)
    plt.yticks(range(len(df)), df["Movie"])
    plt.xlabel("Polarization (||v||)")
    plt.title("Top 10 Polarizing vs Top 10 Non-Polarizing Movies")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if filename:
        plt.savefig(
    filename,
    format="pdf",
    dpi=300,
    bbox_inches='tight',
    pad_inches=0
    )
    plt.show()


def plot_polarization_vs_count(polarization, data_by_movie_train, filename="pdf_reports/Plot_scatter_polarization"):
    counts = np.array([len(r) for r in data_by_movie_train])

    plt.figure(figsize=(10, 6))
    plt.scatter(counts, polarization, alpha=0.6, edgecolor='k')
    plt.xscale('log')
    plt.xlabel("Number of ratings")
    plt.ylabel("Polarization (||v_i||)")
    plt.title("Polarization vs Number of Ratings")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if filename:
        plt.savefig(
    filename,
    format="pdf",
    dpi=300,
    bbox_inches='tight',
    pad_inches=0
    )
    plt.show()


def plot_polarization_with_std(
    polarization,
    data_by_movie_train,
    filename="pdf_reports/Plot_scatter_std_polarization"
):
    counts = np.array([len(r) for r in data_by_movie_train])

    std_ratings = np.array([
        np.std([x[1] for x in r]) if len(r) > 0 else 0
        for r in data_by_movie_train
    ])

    plt.figure(figsize=(10, 6))

    sc = plt.scatter(
        counts,
        polarization,
        c=std_ratings,
        cmap='viridis',
        alpha=0.7,
        edgecolor='k'
    )

    plt.axhline(
        y=2,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Polarization = 2'
    )

    plt.axvline(
        x=1e3,
        color='red',
        linestyle='--',
        linewidth=2,
        label='Ratings = 10³'
    )

    plt.xscale('log')
    plt.xlabel("Number of ratings")
    plt.ylabel("Polarization (||v_i||)")
    plt.title("Polarization vs Number of Ratings (color = rating std)")

    plt.colorbar(sc, label="Rating standard deviation")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(
            filename,
            format="pdf",
            dpi=300,
            bbox_inches='tight',
            pad_inches=0
        )

    plt.show()
