"""
performance_profile.py

Description
-----------
This module contains the performance_profile function which is used to
generate the performance profile graphic comparing multiple classifiers. In
the graphic, each line represents a classifier. In the main function, the
configuration file is read and the function is called with the specified
parameters.


Dependencies
------------
sys
json
typing
numpy
pandas
matplotlib
metric_functions


"""

import sys
import json
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metric_functions import accuracy, kappa, logloss, rmse


def performance_profile(
    file: str,
    metric: Callable[[pd.Series, pd.DataFrame], float] = accuracy,
    method: str = "wd_score",
    window_size: float = 0,
    fixed_time: float = 2.5,
) -> None:
    """Creates the performance profile graphic with one line for each
    classifier.

    Description
    -----------
    Creates the performance profile graphic with one line for each classifier.

    Parameters
    ----------
    file : str
            Path to the .csv file containing the classifier names, subjects,
            and the path to the .csv files with the obtained results.
    metric : Callable[[pd.Series, pd.DataFrame], float], optional
            Metric function used.
    method : str, optional
            Method used to analyze the metric values. Options are 'wd_score',
            'fixed_time', 'integral', and 'oscillation'.
    window_size : float, optional
            Size of experiment window.
    fixed_time : float, optional
            Fixed time to be used in the 'fixed_time' method.

    Returns
    -------
    None

    """

    config_df = pd.read_csv(file)
    subjects = config_df["subject"].unique()
    classifiers = config_df["classifier"].unique()

    # Dataframe table
    table = pd.DataFrame(columns=classifiers).astype(
        {column: "float" for column in classifiers}
    )

    # Iterate over subjects
    for subject in subjects:
        filtered_sub = config_df[config_df["subject"] == subject]
        new_row = []
        # Iterate over classifiers
        for classifier in classifiers:
            filtered = filtered_sub[filtered_sub["classifier"] == classifier]
            # Check if there is a file for the subject and classifier
            if filtered.empty:
                new_row.append(np.nan)
            else:
                file = filtered["path"]
                if len(file) > 1:
                    print(
                        f"Error: more than one file for subject {subject} and"
                        f" classifier {classifier}"
                    )
                    exit(1)
                df = pd.read_csv(file.values[0])
                df["tmin"] += window_size
                times = df["tmin"].unique()
                metrics = []
                # For each time, calculate and store the metric
                for time in times:
                    filtered = df[df["tmin"] == time]
                    metric_value = metric(
                        filtered["true_label"], filtered.iloc[:, 3:]
                    )
                    metrics.append(metric_value)
                # Use the metrics list according to the method chosen
                if method == "wd_score":
                    if metric == accuracy or metric == kappa:
                        max_metric = max(metrics)
                        max_time = times[metrics.index(max_metric)]
                        new_row.append((max_time, round(max_metric, 4)))
                    else:
                        min_metric = min(metrics)
                        min_time = times[metrics.index(min_metric)]
                        new_row.append((min_time, round(min_metric, 4)))
                elif method == "fixed_time":
                    new_row.append(
                        round(metrics[times.tolist().index(fixed_time)], 4)
                    )
                elif method == "integral":
                    new_row.append(round(sum(metrics), 4))
                elif method == "oscillation":
                    oscillation = sum(
                        [
                            abs(metrics[i] - metrics[i - 1])
                            for i in range(1, len(metrics))
                        ]
                    )
                    new_row.append(round(oscillation, 4))
        # Add the new row to the table
        table = pd.concat(
            [table, pd.DataFrame([new_row], columns=classifiers)],
            ignore_index=True,
        )

    print(table.dtypes)

    # Fix negative values
    if metric == kappa:
        table += 1

    # Normalize the table
    table = table.T
    table = table.div(table.max(axis=0), axis=1)
    table = table.T
    if metric == accuracy or metric == kappa:
        table = table**-1
    x_max = table.max().max()

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

    # Plotting the performance profiles
    for classifier in classifiers:
        column = table[classifier]
        nan_count = column.isna().sum()
        column = column.dropna()

        x = np.array(column.sort_values())
        y = np.linspace(0, 1, len(subjects) - nan_count)

        x = np.insert(x, 0, 1)
        y = np.insert(y, 0, 0)

        size = len(x)
        i = 1
        while i < size:
            if x[i] != x[i - 1]:
                x = np.insert(x, i, x[i])
                y = np.insert(y, i, y[i - 1])
                size += 1
            i += 1
        x = np.append(x, x_max)
        y = np.append(y, 1)

        plt.plot(x, y, label=classifier)

    # Other plot configurations
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\rho(\tau)$")
    plt.tight_layout(pad=0.8, rect=[0, 0, 1, 0.96])
    plt.legend()
    # Save the plot
    plt.savefig(metric.__name__ + "_" + method + "_figure.pdf")


if __name__ == "__main__":
    # Check if the configuration file was passed as argument
    try:
        config = sys.argv[1]
    except IndexError:
        print("Usage: python performance_profile.py config_file.json")
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

    if "method" in config_read:
        method = config_read["method"]
    else:
        method = "wd_score"

    if "window_size" in config_read:
        window_size = config_read["window_size"]
    else:
        window_size = 0.0

    if "fixed_time" in config_read:
        fixed_time = config_read["fixed_time"]
    else:
        fixed_time = 2.5

    performance_profile(
        file,
        metric=metric,
        method=method,
        window_size=window_size,
        fixed_time=fixed_time,
    )
