import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

# ============================================================================
# CONFIG
# ============================================================================

INPUT_FILE = "results/results_final.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 5)
plt.rcParams['font.size'] = 12

title_map = {
    "G5": "B5",
    "G8": "B8",
    "G_insu": "B_insu"
}

color_map = {
    "LG": "#1f77b4",
    "SPBN": "#2ca02c",
    "SLGBN": "#ff7f0e"
}

marker_map = {
    "LG": "o",
    "SPBN": "s",
    "SLGBN": "^"
}

x_formatter = mticker.ScalarFormatter()
x_formatter.set_scientific(False)
x_formatter.set_useOffset(False)

# ============================================================================
# LOAD DATA
# ============================================================================

df = pd.read_csv(INPUT_FILE)
df["CHD"] = df["SHD"] + df["THD"]

datasets = sorted(df["Data"].unique())
models = sorted(df["Model"].unique())

# ============================================================================
# GLOBAL Y LIMITS
# ============================================================================

def compute_global_ylim(df, metric, clip_to_zero=False, include_ground_truth=False):
    ymin = np.inf
    ymax = -np.inf

    for dataset in datasets:
        df_d = df[df["Data"] == dataset]

        for model in models:
            df_m = df_d[df_d["Model"] == model]

            grouped = (
                df_m.groupby("train_size")[metric]
                .agg(["mean", "std"])
                .reset_index()
            )

            if grouped.empty:
                continue

            mean = grouped["mean"].values
            std = grouped["std"].values

            upper = mean + std

            if clip_to_zero:
                lower = np.maximum(0, mean - std)
            else:
                lower = mean - std

            ymin = min(ymin, np.min(lower))
            ymax = max(ymax, np.max(upper))

        if include_ground_truth:
            gt_vals = df_d["ground_ll"].dropna().unique()
            if len(gt_vals) > 0:
                gt = gt_vals[0]
                ymin = min(ymin, gt)
                ymax = max(ymax, gt)

    # Precision/Recall empiezan en 0
    if metric in ["Precision", "Recall"]:
        ymin = 0

    # Margen visual para THD/CHD
    if metric in ["THD", "CHD"]:
        ymin = -0.5
        ymax = ymax + 0.5

    return ymin, ymax

# ============================================================================
# CUSTOM ERROR BARS
# ============================================================================

def plot_with_custom_error(ax, x, mean, std, model, clip_to_zero=False):
    color = color_map.get(model, None)
    marker = marker_map.get(model, "o")

    ax.plot(x, mean, marker=marker, linewidth=2, label=model, color=color)

    for xi, m, s in zip(x, mean, std):
        upper = m + s

        if clip_to_zero:
            lower = max(0, m - s)
        else:
            lower = m - s

        ax.plot([xi, xi], [lower, upper], color=color, linewidth=1.5)
        ax.plot([xi * 0.95, xi * 1.05], [upper, upper], color=color, linewidth=1.5)
        ax.plot([xi * 0.95, xi * 1.05], [lower, lower], color=color, linewidth=1.5)

# ============================================================================
# MAIN PLOT FUNCTION
# ============================================================================

def plot_metric(df, metric, title, ylabel, filename,
                include_ground_truth=False,
                clip_to_zero=False,
                share_y_axis=True):

    fig, axes = plt.subplots(1, len(datasets), figsize=(16, 5))

    if share_y_axis:
        ymin, ymax = compute_global_ylim(
            df, metric,
            clip_to_zero=clip_to_zero,
            include_ground_truth=include_ground_truth
        )

    for i, dataset in enumerate(datasets):
        ax = axes[i]
        df_d = df[df["Data"] == dataset]

        if share_y_axis:
            ax.set_ylim(ymin, ymax)
            ax.margins(y=0)
        else:
            ax.margins(y=0.05)

        for model in models:
            df_m = df_d[df_d["Model"] == model]

            grouped = (
                df_m.groupby("train_size")[metric]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("train_size")
            )

            if grouped.empty:
                continue

            x = grouped["train_size"].values
            mean = grouped["mean"].values
            std = grouped["std"].values

            plot_with_custom_error(
                ax, x, mean, std,
                model=model,
                clip_to_zero=clip_to_zero
            )

        # Ground truth
        if include_ground_truth:
            gt_vals = df_d["ground_ll"].dropna().unique()
            if len(gt_vals) > 0:
                gt = gt_vals[0]
                ax.axhline(gt, linestyle='--', linewidth=2, color='red', label='Ground Truth')

                if share_y_axis:
                    ymin_ax, ymax_ax = ax.get_ylim()
                    ax.set_ylim(min(ymin_ax, gt), max(ymax_ax, gt))

        ax.set_xscale("log")
        ax.set_xticks([200, 1000, 5000, 10000])
        ax.xaxis.set_major_formatter(x_formatter)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_title(title_map[dataset], fontweight="bold")
        ax.set_xlabel("Sample Size")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend()

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/{filename}.svg", bbox_inches="tight")
    plt.savefig(f"{OUTPUT_DIR}/{filename}.eps", bbox_inches="tight")
    plt.close()

    print(f"Saved: {filename}")

# ============================================================================
# SPLIT DATA
# ============================================================================

df_fixed = df[df["Type"] == "Fixed"]
df_free = df[df["Type"] == "Free"]

# ============================================================================
# FIXED
# ============================================================================

plot_metric(
    df_fixed,
    metric="best_ll",
    title="Best Log-Likelihood vs Sample Size (Fixed)",
    ylabel="Log-Likelihood",
    filename="fixed_best_ll",
    include_ground_truth=True,
    clip_to_zero=False,
    share_y_axis=False
)

plot_metric(
    df_fixed,
    metric="THD",
    title="THD vs Sample Size (Fixed)",
    ylabel="THD",
    filename="fixed_thd",
    clip_to_zero=True,
    share_y_axis=True
)

# ============================================================================
# FREE
# ============================================================================

plot_metric(
    df_free,
    metric="best_ll",
    title="Best Log-Likelihood vs Sample Size (Free)",
    ylabel="Log-Likelihood",
    filename="free_best_ll",
    include_ground_truth=True,
    clip_to_zero=False,
    share_y_axis=False
)

plot_metric(
    df_free,
    metric="CHD",
    title="CHD vs Sample Size (Free)",
    ylabel="CHD (SHD + THD)",
    filename="free_chd",
    clip_to_zero=True,
    share_y_axis=True
)

plot_metric(
    df_free,
    metric="Precision",
    title="Precision vs Sample Size (Free)",
    ylabel="Precision",
    filename="free_precision",
    clip_to_zero=False,
    share_y_axis=True
)

plot_metric(
    df_free,
    metric="Recall",
    title="Recall vs Sample Size (Free)",
    ylabel="Recall",
    filename="free_recall",
    clip_to_zero=False,
    share_y_axis=True
)

print("\nAll figures generated successfully in 'output/'")