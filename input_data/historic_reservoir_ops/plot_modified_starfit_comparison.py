"""
This script is used to plot a comparison of the original
and modified STARFIT parameters for the reservoirs beltzvilleCombined and fewalter.

Elevation data and storage curves are used to compare the original and modified STARFIT parameters
to observed data.

Elevation timeseries were obtained from the USGS NWIS website.
The following sites were used:

beltzvilleCombined: 01449790
blueMarsh: 01470870
fewalter: 01447780

Storage curves were obtained from the USACE water control manuals:

- USACE, 2014, Water Control Manual (Revised) Francis E. Walter Dam and Reservoir Lehigh River Basin Pennsylvania [PDF].
    Retrieved from https://water.usace.army.mil/a2w/CWMS_CRREL.cwms_util_api.download_dcp?p_dcp_document_id=2564
- USACE, 2016, Water Control Manual (Revised) Beltzville Lake and Dam Lehigh River Basin Pennsylvania [PDF].
    Retrieved from https://water.usace.army.mil/a2w/CWMS_CRREL.cwms_util_api.download_dcp?p_dcp_document_id=2565
- USACE, 2018, Water Control Manual (Revised) Blue Marsh Lake and Dam Schuylkill Basin Pennsylvania [PDF].
    Retrieved from https://water.usace.army.mil/a2w/CWMS_CRREL.cwms_util_api.download_dcp?p_dcp_document_id=2424

"""

import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("./")
# sys.path.append('../../pywrdrb/')

# Custom modules
from pywrdrb.utils.lists import reservoir_list_nyc
from pywrdrb.utils.directories import input_dir, fig_dir, model_data_dir


def get_reservoir_capacity(reservoir, modified=False):
    if modified:
        return float(
            istarf["Adjusted_CAP_MG"]
            .loc[istarf["reservoir"] == f"modified_{reservoir}"]
            .iloc[0]
        )
    else:
        return float(
            istarf["Adjusted_CAP_MG"].loc[istarf["reservoir"] == reservoir].iloc[0]
        )


def get_starfit_params(reservoir, modified=False):
    # Load ISTARF data
    istarf = pd.read_csv(f"{model_data_dir}/drb_model_istarf_conus.csv", index_col=0)

    # Get the ISTARF data for the reservoir
    if modified:
        params = istarf.loc[f"modified_{reservoir}"]
    else:
        params = istarf.loc[reservoir]

    return params


def get_NOR(starfit_params, which_NOR, timestep):
    """
    Get the normalized reservoir storage of the Normal Operating Range (NORlo) for a given timestep.

    Args:
        starfit_params (pd.Series): The starfit parameters for the reservoir.
        timestep (pd.DateTime): The timestep for current day.

    Returns:
        float: The NOR value.
    """

    WATER_YEAR_OFFSET = 0
    assert which_NOR in ["NORlo", "NORhi"]
    use_params = ["mu", "alpha", "max", "min", "beta"]
    use_params = [f"{which_NOR}_{param}" for param in use_params]

    NOR_mu = starfit_params[use_params[0]]
    NOR_alpha = starfit_params[use_params[1]]
    NOR_max = starfit_params[use_params[2]]
    NOR_min = starfit_params[use_params[3]]
    NOR_beta = starfit_params[use_params[4]]

    c = math.pi * (timestep.dayofyear + WATER_YEAR_OFFSET) / 365
    NOR = NOR_mu + NOR_alpha * math.sin(2 * c) + NOR_beta * math.cos(2 * c)
    if (NOR <= NOR_max) and (NOR >= NOR_min):
        return NOR / 100
    elif NOR > NOR_max:
        return NOR_max / 100
    else:
        return NOR_min / 100


def plot_original_and_modified_starfit(reservoir, ax=None, percent_storage=False):
    t_start = "2017-01-01"
    t_end = "2017-12-31"
    daterange = pd.date_range(t_start, t_end, freq="d")
    NORHI_og = np.zeros(len(daterange))
    NORLO_og = np.zeros(len(daterange))

    # Get default STARFIT parameters
    res_starfit_params = get_starfit_params(reservoir=reservoir, modified=False)
    for i, t in enumerate(daterange):
        # NORhi
        NORHI_og[i] = get_NOR(
            starfit_params=res_starfit_params, which_NOR="NORhi", timestep=t
        )
        # NORlo
        NORLO_og[i] = get_NOR(
            starfit_params=res_starfit_params, which_NOR="NORlo", timestep=t
        )

    # Get modified STARFIT parameters and NOR
    NORHI_new = np.zeros(len(daterange))
    NORLO_new = np.zeros(len(daterange))

    res_starfit_params = get_starfit_params(reservoir=reservoir, modified=True)
    for i, t in enumerate(daterange):
        NORHI_new[i] = get_NOR(
            starfit_params=res_starfit_params, which_NOR="NORhi", timestep=t
        )
        NORLO_new[i] = get_NOR(
            starfit_params=res_starfit_params, which_NOR="NORlo", timestep=t
        )

    storage_capacity = get_reservoir_capacity(reservoir, modified=False)
    modified_storage_capacity = get_reservoir_capacity(reservoir, modified=True)
    curve = storage_curves[reservoir]
    window_size = 7

    ### Plotting
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    if percent_storage:
        ax.fill_between(
            daterange.dayofyear,
            NORHI_new * 100,
            NORLO_new * 100,
            color="darkorange",
            label="Modified NOR",
            alpha=0.5,
        )
        ax.fill_between(
            daterange.dayofyear,
            NORHI_og * 100,
            NORLO_og * 100,
            color="cornflowerblue",
            label="Default Extrapolated NOR",
            alpha=0.5,
        )
        ylab = "Percentage of Total Capacity"
    else:
        ax.fill_between(
            daterange.dayofyear,
            NORHI_new * modified_storage_capacity,
            NORLO_new * modified_storage_capacity,
            color="darkorange",
            label="Modified NOR",
            alpha=0.5,
        )
        ax.fill_between(
            daterange.dayofyear,
            NORHI_og * storage_capacity,
            NORLO_og * storage_capacity,
            color="cornflowerblue",
            label="Default Extrapolated NOR",
            alpha=0.5,
        )
        ylab = "Storage (MG)"

    # Plot obs storage
    for yr in range(2017, 2022):
        sub_daterange = pd.date_range(f"{yr}-01-01", f"{yr}-12-31", freq="D")
        lb_elev = lower_elevations.loc[sub_daterange, reservoir]
        lb_stor = np.interp(
            lb_elev.dropna().values, curve["Elevation (ft)"], curve["BG"]
        )
        lb_stor = lb_stor * 1000
        if percent_storage:
            lb_stor = lb_stor / modified_storage_capacity * 100

        d_nonrolled = pd.Series(lb_stor, index=lb_elev.dropna().index)
        d_rolling = d_nonrolled.rolling(window=window_size).mean()
        d_rolling[0:window_size] = lb_stor[0:window_size]
        d_rolling[-window_size:] = lb_stor[-window_size:]
        d = d_rolling

        if yr == 2017:
            ax.plot(
                lb_elev.dropna().index.dayofyear,
                d,
                color="black",
                alpha=1,
                zorder=3,
                label="Observed Storage",
                ls="dashed",
            )
        else:
            ax.plot(
                lb_elev.dropna().index.dayofyear,
                d,
                color="black",
                zorder=3,
                ls="dashed",
                alpha=0.75,
                lw=2,
            )

    ax.legend()
    ax.set_ylabel(ylab, fontsize=14)
    ax.set_xlabel("Day of Year (Jan-Dec)", fontsize=14)
    ax.set_title(f"{reservoir}", fontsize=14)
    return


if __name__ == "__main__":
    ### LOAD
    # Load observed elevation data
    lower_elevations = pd.read_csv(
        f"{input_dir}historic_reservoir_ops/lower_basin_reservoir_elevation.csv",
        index_col=0,
        parse_dates=True,
    )

    storage_curves = {}
    for res in ["blueMarsh", "beltzvilleCombined", "fewalter"]:
        storage_curves[res] = pd.read_csv(
            f"{input_dir}historic_reservoir_ops/{res}_elevation_storage_curve.csv",
            sep=",",
        )

    ### get reservoir storage capacities
    istarf = pd.read_csv(f"{model_data_dir}drb_model_istarf_conus.csv")

    capacities = {r: get_reservoir_capacity(r) for r in reservoir_list_nyc}
    capacities["combined"] = sum([capacities[r] for r in reservoir_list_nyc])

    nyc_hist_storage = pd.read_csv(
        f"{input_dir}/historic_NYC/NYC_storage_daily_2000-2021.csv",
        sep=",",
        index_col=0,
        parse_dates=True,
    )
    nyc_hist_storage.index = pd.to_datetime(nyc_hist_storage.index)

    ### PLOTTING
    # Plot both beltzvilleCombined and fewalter starfit curves
    fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=350)
    plot_original_and_modified_starfit(
        "beltzvilleCombined", ax=ax[0], percent_storage=False
    )
    plot_original_and_modified_starfit("fewalter", ax=ax[1], percent_storage=False)
    fig.tight_layout()
    plt.savefig(f"{fig_dir}starfit_modifications.png", dpi=350)
