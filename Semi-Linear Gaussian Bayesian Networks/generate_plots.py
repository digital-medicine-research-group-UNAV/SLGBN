import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D

# =========================
# Load data
# =========================
df = pd.read_csv("results/results.csv")

required_columns = {
    "Data",
    "Model",
    "Score",
    "Type",
    "train_size",
    "best_ll",
    "ground_ll",
    "THD",
    "SHD",
    "Recall",
    "Precision",
}
missing_columns = sorted(required_columns - set(df.columns))
if missing_columns:
    raise ValueError(
        "results/results_ebic.csv is missing required columns: "
        + ", ".join(missing_columns)
    )

df["Model"] = df["Model"].replace({"LG": "LGBN"})
df["ModelScore"] = df["Model"] + "-" + df["Score"]

# =========================
# Output
# =========================
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# =========================
# STYLE
# =========================
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 7)
plt.rcParams["font.size"] = 11
TITLE_SIZE = 14
AXIS_LABEL_SIZE = 13
TICK_LABEL_SIZE = 12
LEGEND_SIZE = 11

# =========================
# COLORS (by MODEL)
# =========================
model_colors = {
    "LGBN": "#1f77b4",
    "SLGBN": "#ff7f0e",
    "SPBN": "#2ca02c"
}

data_order = ["G5", "G8", "G_insu"]

data_title = {
    "G5": "B5",
    "G8": "B8",
    "G_insu": "B_insu"
}

# =========================
# SCORE STYLE
# =========================
score_linestyle = {
    "BIC": "-",
    "EBIC": (0, (5, 3)),
    "CVLL": "-"
}

score_marker = {
    "BIC": "o",
    "EBIC": "^",
    "CVLL": "*"
}

# =========================
# helpers
# =========================
def get_model(ms):
    return ms.split("-")[0]

def get_score(ms):
    return ms.split("-")[1]

def get_ground_truth_ll(data_name):
    gt_values = df.loc[df["Data"] == data_name, "ground_ll"].dropna().unique()
    if len(gt_values) == 0:
        return None
    return gt_values[0]

def set_train_size_ticks(ax, values):
    train_sizes = sorted(pd.Series(values).dropna().astype(int).unique())
    if not train_sizes:
        return

    ax.xaxis.set_major_locator(mticker.FixedLocator(train_sizes))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mticker.NullLocator())

# =========================
# plotting function
# =========================
def plot_metric_by_type(metric, ylabel, type_name):

    datasets = [data for data in data_order if data in df["Data"].unique()]
    extra_datasets = sorted(set(df["Data"].unique()) - set(datasets))
    datasets.extend(extra_datasets)

    share_y_axis = metric != "best_ll"
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(5 * len(datasets), 5),
        sharey=share_y_axis,
    )
    axes = np.atleast_1d(axes)

    legend_handles = {}

    for ax, data_name in zip(axes, datasets):

        df_subset = df[(df["Data"] == data_name) & (df["Type"] == type_name)]

        for ms in df_subset["ModelScore"].unique():

            subset = df_subset[df_subset["ModelScore"] == ms]
            stats = (
                subset.groupby("train_size")[metric]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("train_size")
            )

            if stats.empty or stats["mean"].isna().all():
                continue

            model = get_model(ms)
            score = get_score(ms)

            color = model_colors.get(model, "black")
            linestyle = score_linestyle.get(score, "-")
            marker = score_marker.get(score, "o")

            # =========================
            # PLOT (no errorbars)
            # =========================
            ax.plot(
                stats["train_size"],
                stats["mean"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=8,
                linewidth=2,
                alpha=0.9
            )

            # =========================
            # LEGEND (clean representation)
            # =========================
            if ms not in legend_handles:
                legend_handles[ms] = Line2D(
                    [0], [0],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=9,
                    linewidth=2,
                    label=ms
                )

        if metric == "best_ll" and "ground_ll" in df.columns:
            ground_truth = get_ground_truth_ll(data_name)
            if ground_truth is not None:
                ax.axhline(
                    ground_truth,
                    color="red",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.9
                )

                if "Ground truth" not in legend_handles:
                    legend_handles["Ground truth"] = Line2D(
                        [0], [0],
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label="Ground truth"
                    )

        ax.set_title(data_title.get(data_name, data_name), fontsize=TITLE_SIZE)
        ax.set_xlabel("Train size", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        set_train_size_ticks(ax, df_subset["train_size"])
        ax.grid(alpha=0.3)

    if legend_handles:
        fig.legend(
            handles=list(legend_handles.values()),
            loc="lower center",
            ncol=min(len(legend_handles), 4),
            fontsize=LEGEND_SIZE,
            handlelength=3.5,
            numpoints=2
        )

    plt.tight_layout(rect=(0, 0.08, 1, 1))

    plt.savefig(output_dir / f"{metric.lower()}_{type_name}.svg")
    plt.savefig(output_dir / f"{metric.lower()}_{type_name}.eps", dpi=300)

    plt.close()

# =========================
# Run plots
# =========================
plot_metric_by_type("best_ll", "Log-Likelihood", "Fixed")
plot_metric_by_type("THD", "THD", "Fixed")

plot_metric_by_type("best_ll", "Log-Likelihood", "Free")
plot_metric_by_type("THD", "THD", "Free")
plot_metric_by_type("SHD", "SHD", "Free")
plot_metric_by_type("Recall", "Recall", "Free")
plot_metric_by_type("Precision", "Precision", "Free")

print("✓ All EBIC-MLE plots saved (grouped by metric and type)")
