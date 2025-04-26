"""
metric_table.py

Description
-----------
This module contains the metric_table function which is used to generate the
table comparing multiple classifiers and subjects. The table has a column for
each classifier and each line represents a subject, whereas the two last lines
refer to the 'all' subject and the mean of individual subjects. In the main
function, the configuration file is read and the function is called with the
specified parameters.


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
from typing import Callable, Optional

import numpy as np
import pandas as pd
from tabulate import tabulate

from metric_functions import accuracy, kappa, logloss, rmse

tabulate.PRESERVE_WHITESPACE = True


def calculate_metric(
    df: pd.DataFrame,
    metric: Callable[[pd.Series, pd.DataFrame], float],
    classifier: str,
    subject: Optional[str] = None,
    window_size: float = 0,
):
    filtered = df[df["classifier"] == classifier]
    if subject is not None:
        filtered = filtered[filtered["subject"] == subject]["path"]
        if len(filtered) > 1:
            print(
                f"Error: more than one file for subject {subject} and"
                f" classifier {classifier}"
            )
            exit(1)
        data = pd.read_csv(filtered.values[0])
    else:
        filtered = filtered["path"]
        for f in filtered:
            df = pd.concat([df, pd.read_csv(f)])

                


    # Check if there is a file for the subject and classifier
    if filtered.empty:
        new_row.append(np.nan)  # does not work with pandas nan
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


def metric_table(
    file: str,
    metric: Callable[[pd.Series, pd.DataFrame], float] = accuracy,
    method: str = "wd_score",
    window_size: float = 0,
    fixed_time: float = 2.5,
) -> None:
    """Creates a LaTeX table with classifiers as columns and lines representing
    each subject, as well as additional lines for the mean of each subjects and
    the mean considering an 'all' subject.

    Description
    -----------
    Creates a LaTeX table with classifiers as columns and lines representing
    each subject, as well as additional lines for the mean of each subjects and
    the mean considering an 'all' subject.

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
    columns = ["subjects"] + list(classifiers)
    types = {
        column: "object" if column == "subjects" else "float"
        for column in columns
    }
    table = pd.DataFrame(columns=columns).astype(types)

    # Iterate over subjects
    for subject in subjects:
        filtered_sub = config_df[config_df["subject"] == subject]
        new_row = [subject]
        # Iterate over classifiers
        for classifier in classifiers:
            filtered = filtered_sub[filtered_sub["classifier"] == classifier]
            # Check if there is a file for the subject and classifier
            if filtered.empty:
                new_row.append(np.nan)  # does not work with pandas nan
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
        # Add new row to the table
        table = pd.concat(
            [table, pd.DataFrame([new_row], columns=columns)],
            ignore_index=True,
        )

    # Calculate the mean row
    new_row = ["mean"]
    # Iterate over classifiers
    for classifier in classifiers:
        mean_value = table[classifier]
        if method == "wd_score":
            mean_value = [x[1] for x in mean_value if x is not np.nan]
        else:
            mean_value = [x for x in mean_value if x is not np.nan]
        mean_value = round(sum(mean_value) / len(mean_value), 4)
        new_row.append(mean_value)
    # Add new row to the table
    table = pd.concat(
        [table, pd.DataFrame([new_row], columns=columns)], ignore_index=True
    )

    # Calculate the subject 'all' row
    new_row = ["all"]
    # Concatenate all the dataframes for each classifier and plot
    for classifier in classifiers:
        df = pd.DataFrame()
        filtered = config_df[config_df["classifier"] == classifier]
        if filtered.empty:
            new_row.append(np.nan)
        else:
            file_names = filtered["path"]
            for f in file_names:
                df = pd.concat([df, pd.read_csv(f)])
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
                integral = sum(metrics)
                new_row.append(round(integral, 4))
            elif method == "oscillation":
                oscillation = sum(
                    [
                        abs(metrics[i] - metrics[i - 1])
                        for i in range(1, len(metrics))
                    ]
                )
                new_row.append(round(oscillation, 4))
    table = pd.concat(
        [table, pd.DataFrame([new_row], columns=columns)], ignore_index=True
    )

    # Convert values to strings with 4 decimal places (keeps final zeros)
    for column in table.columns:
        if column != "subjects":
            table[column] = table[column].apply(
                lambda x: (
                    f"({x[0]:.1f}, {x[1]:.4f})" if type(x) is tuple else x
                )
            )
            table[column] = table[column].apply(
                lambda x: f"{x:.4f}" if type(x) is float else x
            )
    # Swap the last two rows
    aux = table.iloc[-1]
    table.iloc[-1] = table.iloc[-2]
    table.iloc[-2] = aux

    # Convert the table to a LaTeX table
    latex_table = tabulate(
        table,
        tablefmt="latex_booktabs",
        headers="keys",
        showindex=False,
        colalign=tuple(["center" for i in range(len(classifiers) + 1)]),
        floatfmt=tuple(
            ["" if i == 0 else ".4f" for i in range(len(classifiers) + 1)]
        ),
    )
    # Adds horizontal lines to the LaTeX table to improve readability
    lines = latex_table.split("\n")
    lines.insert(0, "\\begin{table}[ht!]")
    lines.append("\\end{table}")
    lines.insert(-5, "\\midrule")
    latex_table = "\n".join(lines)
    # Save the LaTeX table to a file
    with open(metric.__name__ + "_" + method + "_table.tex", "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    # Check if the configuration file was passed as argument
    try:
        config = sys.argv[1]
    except IndexError:
        print("Usage: python metric_table.py config_file.json")
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

    metric_table(
        file,
        metric=metric,
        method=method,
        window_size=window_size,
        fixed_time=fixed_time,
    )
