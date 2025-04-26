"""
metric_plot.py

Description
-----------
This module contains the metric_plot function which is used to generate the
graph for a designated metric comparing multiple classifiers throughout time
considering the subject 'all'. In the main function, the configuration file is
read and the function is called with the specified parameters.


Dependencies
------------
sys
json
typing
pandas
matplotlib
metric_functions


"""

import sys
import json
from typing import Optional, Callable

import pandas as pd
import matplotlib.pyplot as plt

from metric_functions import accuracy, kappa, logloss, rmse


def metric_plot(
    file: str,
    metric: Callable[[pd.Series, pd.DataFrame], float] = accuracy,
    window_size: float = 0,
    v_lines: Optional[dict] = None,
) -> None:
    """Plots the metric values over time for each classifier.

    Description
    -----------
    Plots the metric values over time for each classifier.

    Parameters
    ----------
    file : str
            Path to the .csv file containing the classifier names, subjects,
            and the path to the .csv files with the obtained results.
    metric : Callable[[pd.Series, pd.DataFrame], float], optional
            Metric function used.
    window_size : float, optional
            Size of experiment window.
    v_lines: dict, optional
            Dictionary containing the vertical lines to be plotted. It should
            have the following keys: x, color, linestyle, and label. Each key
            containing a list of values.

    Returns
    -------
    None

    """

    config_df = pd.read_csv(file)
    classifiers = config_df["classifier"].unique()

    # Hard coded line styles and colors
    poss_lines = ["-", "--", ":"]
    poss_colors = ["#59a14f", "#6a3d9a", "#f28e2b"]
    lines = [poss_lines[i % len(poss_lines)] for i in range(len(classifiers))]
    colors = [
        poss_colors[i // len(poss_colors)] for i in range(len(classifiers))
    ]

    # Plotting configuration
    plt.figure(figsize=(3.4, 3))
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": "Times New Roman",
            "mathtext.fontset": "custom",
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    # Concatenate all the dataframes for each classifier and plot
    for i, classifier in enumerate(classifiers):
        df = pd.DataFrame()
        filtered = config_df[config_df["classifier"] == classifier]
        file_names = filtered["path"]
        for f in file_names:
            df = pd.concat([df, pd.read_csv(f)])
        df["tmin"] += window_size
        times = df["tmin"].unique()
        metrics = []
        # For each time, calculate and store the metric
        for time in times:
            filtered = df[df["tmin"] == time]
            metric_value = metric(filtered["true_label"], filtered.iloc[:, 3:])
            metrics.append(metric_value)
        plt.plot(
            times,
            metrics,
            label=classifier,
            color=colors[i],
            linestyle=lines[i],
        )

    # Plot vertical lines
    if v_lines is not None:
        for i in range(len(v_lines["x"])):
            plt.axvline(
                x=v_lines["x"][i],
                color=v_lines["color"][i],
                linestyle=v_lines["linestyle"][i],
                label=v_lines["label"][i],
            )

    # Other plot configurations
    plt.xlabel(r"Time [s]")
    if metric == rmse:
        plt.ylabel("RMSE")
    elif metric == logloss:
        plt.ylabel("Cross Entropy Loss")
    else:
        plt.ylabel(metric.__name__.capitalize())
    plt.tight_layout(pad=0.8, rect=(0, 0, 1, 0.96))
    plt.legend()
    # Save the plot
    plt.savefig(metric.__name__ + "_plot.pdf")


if __name__ == "__main__":
    # Check if the configuration file was passed as argument
    try:
        config = sys.argv[1]
    except IndexError:
        print("Usage: python metric_plot.py config_file.json")
        exit(1)

    # Read configuration .json file if it exists
    try:
        with open(config) as json_file:
            config_read = json.load(json_file)
    except FileNotFoundError:
        print("Error: configuration file not found")
        exit(1)

    # Check if the required .csv file is in the configuration file
    try:
        file = config_read["file"]
        with open(file):
            pass
    except FileNotFoundError:
        print("Error: .csv file not found")
        exit(1)
    except KeyError:
        print("Error: .csv file not in configuration file")
        exit(1)

    # Check all the other parameters in the configuration file which are used
    if "metric" in config_read:
        metric = config_read["metric"]
        if metric == "accuracy":
            metric = accuracy
        elif metric == "kappa":
            metric = kappa
        elif metric == "logloss":
            metric = logloss
        elif metric == "rmse":
            metric = rmse
        else:
            print("Error: invalid metric not found")
            exit(1)
    else:
        metric = accuracy

    if "v_lines" in config_read:
        v_lines = config_read["v_lines"]
    else:
        v_lines = None

    if "window_size" in config_read:
        window_size = config_read["window_size"]
    else:
        window_size = 0.0

    metric_plot(file, v_lines=v_lines, metric=metric, window_size=window_size)
