import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os

def plot_task_completion_stats(csv_path="all_earliest_steps.csv", save_path=None):
    """
    Plot the mean and standard deviation of earliest step completion for each task across all rounds.
    Reads the cumulative CSV file, computes statistics, and displays a bar chart with error bars.
    :param csv_path: Path to the CSV file containing all rounds' earliest steps (default: 'all_earliest_steps.csv')
    :param save_path: If provided, saves the figure to this path instead of displaying it
    """
    

    # Read the cumulative results
    df = pd.read_csv(csv_path)
    if df.empty:
        print("No data to plot.")
        return

    # If only one row, mean and std will be scalars; convert to Series for consistent plotting
    if len(df) == 1:
        # Ensure task_means and task_std are pandas Series for consistent plotting
        task_means = pd.Series(df.iloc[0], index=df.columns)
        task_std = pd.Series([0]*len(df.columns), index=df.columns)
    else:
        task_means = df.mean(axis=0)
        task_std = df.std(axis=0)

    # Ensure task_means and task_std are pandas Series for consistent plotting
    if not isinstance(task_means, pd.Series):
        task_means = pd.Series(task_means, index=df.columns)
    if not isinstance(task_std, pd.Series):
        task_std = pd.Series(task_std, index=df.columns)

    # Use .to_numpy() for values and .index for labels
    plt.figure(figsize=(10, 6))
    plt.bar(task_means.index, task_means.to_numpy(), yerr=task_std.to_numpy(), color='lightcoral')
    plt.xlabel('Tasks')
    plt.ylabel('Mean Step Completion')
    plt.title('Average Step (AS)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_task_completion_and_success(
    csv_path="all_earliest_steps.csv",
    icon_paths=None,
    save_path=None
):
    """
    Plot the mean, std, and success rate for each task across all rounds, with task icons above the bars.
    :param csv_path: Path to the CSV file containing all rounds' earliest steps (default: 'all_earliest_steps.csv')
    :param icon_paths: List of icon image paths for each task, in order
    :param save_path: If provided, saves the figure to this path instead of displaying it
    """
    # Set default icon paths if not provided
    if icon_paths is None:
        icon_paths = [
            'icons/wood.png', 'icons/table.png', 'icons/wood_pickaxe.png', 'icons/stone.png',
            'icons/stone_pickaxe.png', 'icons/iron.png', 'icons/coal.png', 'icons/furnace.png',
            'icons/iron_pickaxe.png', 'icons/diamond.png'
        ]

    # Read the cumulative results
    df = pd.read_csv(csv_path)
    if df.empty:
        print("No data to plot.")
        return

    # Compute mean, std, and success rate for each task
    if len(df) == 1:
        task_means = pd.Series(df.iloc[0], index=df.columns)
        task_std = pd.Series([0]*len(df.columns), index=df.columns)
        success_rates = pd.Series([100 if not pd.isna(df.iloc[0][col]) else 0 for col in df.columns], index=df.columns)
    else:
        task_means = df.mean(axis=0)
        task_std = df.std(axis=0)
        # Success rate: percentage of non-NaN values per column
        success_rates = df.notna().sum(axis=0) / len(df) * 100

    # Ensure all stats are pandas Series for consistent plotting
    if not isinstance(task_means, pd.Series):
        task_means = pd.Series(task_means, index=df.columns)
    if not isinstance(task_std, pd.Series):
        task_std = pd.Series(task_std, index=df.columns)
    if not isinstance(success_rates, pd.Series):
        success_rates = pd.Series(success_rates, index=df.columns)

    # Set up the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for mean step completion with std as error bars
    ax1.bar(task_means.index, task_means.to_numpy(), yerr=task_std.to_numpy(), capsize=5, color='lightcoral', label='Mean Step Completion')
    ax1.set_xlabel('Tasks')
    ax1.set_ylabel('Average Steps (AS)', color='lightcoral')
    ax1.tick_params(axis='y', labelcolor='lightcoral')
    ax1.set_xticklabels(task_means.index, rotation=45, ha='right')

    # Line plot for success rate on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(success_rates.index, success_rates.to_numpy(), color='mediumseagreen', marker='o', label='Success Rate')
    ax2.set_ylabel('Success Rate (%)', color='mediumseagreen')
    ax2.tick_params(axis='y', labelcolor='mediumseagreen')
    ax2.set_ylim(0, 110)

    # Add icons above each bar
    zoom_ratios = [0.4, 0.4, 0.8, 0.4, 0.8, 0.4, 0.4, 0.4, 0.8, 0.4]
    for i, (task, zoom_ratio) in enumerate(zip(task_means.index, zoom_ratios)):
        if i < len(icon_paths):
            try:
                img = mpimg.imread(icon_paths[i])
                imagebox = OffsetImage(img, zoom=zoom_ratio)
                ab = AnnotationBbox(imagebox, (i, task_means[task]+50), frameon=False)
                ax1.add_artist(ab)
            except Exception as e:
                print(f"Could not load icon for {task}: {e}")

    plt.title('Average Step Completion Time and Success Rate for Each Task')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


class Plotter:
    """
    A class for plotting experiment results with icons above each task.
    """
    def __init__(self, picture_names=None, colors=None, figsize=(10, 4)):
        # Set default icon paths if not provided
        if picture_names is None:
            picture_names = [
                'icons/wood.png', 'icons/table.png', 'icons/wood_pickaxe.png', 'icons/stone.png',
                'icons/stone_pickaxe.png', 'icons/iron.png', 'icons/coal.png', 'icons/furnace.png',
                'icons/iron_pickaxe.png', 'icons/diamond.png'
            ]
        # Set default colors if not provided
        if colors is None:
            colors = [
                ('royalblue', 'lightskyblue'),
                ('firebrick', 'lightcoral'),
                ('darkgoldenrod', 'gold'),
                ('mediumseagreen', 'lightgreen')
            ]
        self.picture_names = picture_names
        self.colors = colors
        self.figsize = figsize

    def plot(self, data_list, legend_labels=None):
        """
        Read CSV files and plot mean and 95% confidence interval for multiple experiment groups, with icons above each task.
        :param csv_files: List of CSV file paths, each for one experiment group (rows: rounds, columns: tasks)
        :param legend_labels: List of legend labels for each group
        """
        dfs = []
        for data in data_list:
            if isinstance(data, str):
                df = pd.read_csv(data)
            else:
                df = data
            # Drop rows and columns that are completely empty
            df = df.dropna(how='all').dropna(axis=1, how='all')
            # If the first column is not numeric (e.g., an index or label), drop it
            if not np.issubdtype(df.iloc[:,0].dtype, np.number):
                try:
                    df = df.iloc[:, 1:]
                except Exception:
                    pass
            # Try to convert all data to float, non-convertible values become NaN
            df = df.apply(pd.to_numeric, errors='coerce')
            dfs.append(df)

        if legend_labels is None:
            legend_labels = [f'Group {i+1}' for i in range(len(dfs))]

        means_all = []
        ci_lower_all = []
        ci_upper_all = []

        # Calculate mean and 95% confidence interval for each group
        for df in dfs:
            means = df.mean(axis=0)
            stds = df.std(axis=0)
            n = df.shape[0]
            ci_lower = means - 1.96 * stds / np.sqrt(n)
            ci_upper = means + 1.96 * stds / np.sqrt(n)
            means_all.append(means)
            ci_lower_all.append(ci_lower)
            ci_upper_all.append(ci_upper)

        # For icon placement: get the max mean+ci_upper for each task
        means_max = np.max([ci_upper for ci_upper in ci_upper_all], axis=0)

        x_values = np.arange(len(dfs[0].columns))
        plt.figure(figsize=self.figsize)
        # Plot each group's mean and confidence interval
        for means, ci_lower, ci_upper, label, color in zip(means_all, ci_lower_all, ci_upper_all, legend_labels, self.colors):
            plt.plot(x_values, means, '-o', label=label, color=color[0])
            plt.fill_between(x_values, ci_lower, ci_upper, color=color[1], alpha=0.3)

        # Add icons above each task
        for i, img_path in enumerate(self.picture_names[:len(x_values)]):
            try:
                full_path = os.path.join('/Users/brenda/Desktop/AI_Agents/1012_6_agent', img_path)
                img = mpimg.imread(full_path)
                imagebox = OffsetImage(img, zoom=0.3)  # Adjust zoom as needed
                ab = AnnotationBbox(imagebox, (x_values[i], means_max[i]), frameon=False, zorder=10)
                plt.gca().add_artist(ab)
            except Exception as e:
                print(f"Could not load icon for {dfs[0].columns[i]}: {e}")

        plt.xticks(x_values, dfs[0].columns, rotation=45, ha='right')
        plt.ylabel('Average Steps (AS)')
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig('results/result_fig.png', bbox_inches='tight', dpi=300)
        plt.show()