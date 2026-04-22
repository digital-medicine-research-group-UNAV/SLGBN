import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
This script generates the figures used in the paper from the experimental results.

It reads the results stored in a CSV file, processes both "Fixed" and "Free"
experimental settings, and produces plots for different evaluation metrics.

The resulting figures are automatically saved in the "output" directory
in both EPS and SVG formats, ready for inclusion in the paper.
"""

os.makedirs("output", exist_ok=True)

df = pd.read_csv("results/results.csv")

df_fixed = df[df['Type'] == 'Fixed'].copy()
df_free = df[df['Type'] == 'Free'].copy()

ground_ll_5 = df_free[df_free['Data'] == 'G5']['ground_ll'].iloc[0]
ground_ll_8 = df_free[df_free['Data'] == 'G8']['ground_ll'].iloc[0]
ground_ll_insu = df_free[df_free['Data'] == 'G_insu']['ground_ll'].iloc[0]

ground_truth_map = {
    "G5": ground_ll_5,
    "G8": ground_ll_8,
    "G_insu": ground_ll_insu
}

title_map = {
    "G5": "B5",
    "G8": "B8",
    "G_insu": "B_insu"
}

sns.set(style="whitegrid")

# FIXED

fixed = ['best_ll', 'THD']

for variable in fixed:

    shd_mean = (
        df_fixed
        .groupby(['Data', 'Model', 'train_size'])[variable]
        .agg(['mean', 'min', 'max'])
        .reset_index()
    )

    datasets = shd_mean['Data'].unique()
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))

    if n_datasets == 1:
        axes = [axes]

    for ax, data_name in zip(axes, datasets):

        df_plot = shd_mean[shd_mean['Data'] == data_name]

        for model in df_plot['Model'].unique():

            df_model = df_plot[df_plot['Model'] == model].sort_values('train_size')

            ax.plot(
                df_model['train_size'],
                df_model['mean'],
                marker='o',
                label=model
            )

        if variable == "best_ll":
            ax.axhline(
                ground_truth_map[data_name],
                color='red',
                linestyle='--',
                linewidth=2,
                label="Ground truth"
            )

        ax.set_xscale('log')

        train_sizes = sorted(df_plot['train_size'].unique())

        ax.set_xticks(train_sizes)
        ax.set_xticklabels(train_sizes, fontsize = 14)

        if ax == axes[len(axes)//2]:
            ax.set_xlabel("Train size", fontsize=16)
        ax.set_title(title_map[data_name], fontsize = 18)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True)

    axes[0].set_ylabel(variable, fontsize=16)
    
    handles, labels = axes[-1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.07),
        ncol=len(labels),
        fontsize=16,
        title_fontsize=16
    )

    plt.tight_layout()

    plt.savefig(f"output/{variable}_fixed.eps", format="eps", bbox_inches="tight")
    plt.savefig(f"output/{variable}_fixed.svg", format="svg", bbox_inches="tight")

    plt.close()

# FREE

free = [
    'best_ll', 'Total_SHD', 'Precision_total', 'Recall_total'
]

node_map = {
    "G5": 5,
    "G8": 8,
    "G_insu": 27
}

df_free["n_nodes"] = df_free["Data"].map(node_map)
df_free['Total_SHD'] = df_free['THD'] + df_free['skeleton_SHD']
df_free["K_types"] = df_free["n_nodes"] - df_free["THD"]
df_free["E_true"] = df_free["skeleton_K"] + df_free["skeleton_FN"]
df_free["E_pred"] = df_free["skeleton_K"] + df_free["skeleton_FP"]
df_free["K_total"] = df_free["skeleton_K"] + df_free["K_types"]
df_free["Recall_total"] = df_free["K_total"] / (df_free["E_true"] + df_free["n_nodes"])
df_free["Precision_total"] = df_free["K_total"] / (df_free["E_pred"] + df_free["n_nodes"])

for variable in free:

    shd_mean = (
        df_free
        .groupby(['Data', 'Model', 'train_size'])[variable]
        .agg(['mean', 'min', 'max'])
        .reset_index()
    )

    datasets = shd_mean['Data'].unique()
    n_datasets = len(datasets)

    fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5), sharey=(variable in ['Precision_total', 'Recall_total']))

    if n_datasets == 1:
        axes = [axes]

    for ax, data_name in zip(axes, datasets):

        df_plot = shd_mean[shd_mean['Data'] == data_name]

        for model in df_plot['Model'].unique():

            df_model = df_plot[df_plot['Model'] == model].sort_values('train_size')

            ax.plot(
                df_model['train_size'],
                df_model['mean'],
                marker='o',
                label=model
            )

        if variable == "best_ll":
            ax.axhline(
                ground_truth_map[data_name],
                color='red',
                linestyle='--',
                linewidth=2,
                label="Ground truth"
            )

        ax.set_xscale('log')

        train_sizes = sorted(df_plot['train_size'].unique())

        ax.set_xticks(train_sizes)
        ax.set_xticklabels(train_sizes, fontsize = 14)

        if ax == axes[len(axes)//2]:
            ax.set_xlabel("Train size", fontsize=16)
        ax.set_title(title_map[data_name], fontsize=18)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(True)

    axes[0].set_ylabel(variable, fontsize=16)

    handles, labels = axes[-1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.07),
        ncol=len(labels),
        fontsize=16,
        title_fontsize=16
    )

    plt.tight_layout()

    plt.savefig(f"output/{variable}_free.eps", format="eps", bbox_inches="tight")
    plt.savefig(f"output/{variable}_free.svg", format="svg", bbox_inches="tight")

    plt.close()