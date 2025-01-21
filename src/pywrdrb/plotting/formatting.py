import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_subplot_handles_and_labels(axs):
    # Gather legend handles and labels from all subplots and combine into single legend
    handles, labels = [], []
    for ax in axs:
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    # get only unique handles and labels
    handles, labels = np.array(handles), np.array(labels)
    idx = np.unique(labels, return_index=True)[1]
    handles, labels = handles[idx], labels[idx]
    return handles, labels


def clean_xtick_labels(
    axes,
    start_date,
    end_date,
    fontsize=10,
    date_format="%Y",
    max_ticks=10,
    rotate_labels=False,
):
    """
    Clean up x-axis tick labels for time series data.
    """
    try:
        start_date = (
            pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
        )
        end_date = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date

        if start_date >= end_date:
            raise ValueError(
                f"Start date must be before end date. Start: {start_date}, End: {end_date}"
            )

        total_days = (end_date - start_date).days

        if total_days <= 30:
            date_format = "%Y-%m-%d"
            tick_spacing = "D"
        elif total_days <= 365 * 2:
            date_format = "%Y-%m"
            tick_spacing = "MS"
        elif total_days <= 365 * 6:
            date_format = "%Y"
            tick_spacing = "1YS"
        elif total_days <= 365 * 10:
            date_format = "%Y"
            tick_spacing = "2YS"
        elif total_days <= 365 * 20:
            # Space every 5 years
            date_format = "%Y"
            tick_spacing = "5YS"
        else:
            # Space every 10 years
            date_format = "%Y"
            tick_spacing = "10YS"

        use_ticks = pd.date_range(start_date, end_date, freq=tick_spacing)
        tick_labels = [t.strftime(date_format) for t in use_ticks]

        for i in range(len(axes)):
            ax = axes[i]
            ax.set_xticks(use_ticks)
            ax.set_xticklabels(
                tick_labels,
                rotation=45 if rotate_labels else 0,
                fontsize=fontsize,
                ha="center",
            )
            ax.tick_params(axis="x", which="minor", length=0)
            ax.xaxis.set_minor_locator(plt.NullLocator())

            # Adjust layout to ensure labels are not cut off
            ax.figure.tight_layout()

    except Exception as e:
        print(f"Error in setting tick labels: {e}")

    return axes
