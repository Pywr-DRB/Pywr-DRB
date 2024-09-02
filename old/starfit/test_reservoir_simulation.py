"""
Trevor Amestoy
Summer 2022

Testing reservoir simulator which uses STARFIT rule parameters.

Tests performance of Blue Marsh and Beltzville reservoirs for:
"""

# Core modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Custom simulation and harmonic functions using STARFIT params

from simulate_reservoir_daily import starfit_reservoir_simulation
from simulate_reservoir_daily import NOR_hi, NOR_lo, release_harmonic


def error_calculation(obs, sim, error_type="NSE"):
    if error_type == "RMSE":
        return np.sqrt(np.nanmean((sim - obs) ** 2))
    elif error_type == "NSE":
        return 1 - np.nansum((obs - sim) ** 2) / (
            np.nansum((obs - np.nanmean(obs)) ** 2)
        )
    else:
        print("Incorrect error type specification.")
        return


### Load STARFIT conus data for reservoirs in DRB
starfit = pd.read_csv("drb_model_istarf_conus.csv")
reservoirs = [res for res in starfit["reservoir"]]
reservoir_ids = [id for id in starfit["GRanD_ID"]]
reservoir_names = [name for name in starfit["GRanD_NAME"]]


# Constants
t = 365 * 2
n_sims = 1
n_time = t
time_step = "daily"
data_start = 12694
data_end = data_start + t


# Simulate Blue Marsh and Beltzville
for i in [4]:
    # Select one set of STARFIT data at a time
    reservoir_data = starfit.iloc[i]

    if i == 4:
        reservoir_lab = "Blue Marsh"
        resops_data = pd.read_csv(
            "./ResOpsUS_data/resops_BlueMarsh.csv", delimiter=",", header=0
        )
    elif i == 5:
        reservoir_lab = "Beltzville"
        resops_data = pd.read_csv(
            "./ResOpsUS_data/resops_Beltzville.csv", delimiter=",", header=0
        )

    I_bar = reservoir_data["GRanD_MEANFLOW_MGD"]
    x_lab = "Day"

    S_cap = reservoir_data["GRanD_CAP_MG"]
    R_max = (reservoir_data["Release_max"] + 1) * (I_bar)
    R_min = (reservoir_data["Release_min"] + 1) * (I_bar)

    # Test different initial storage
    initial_conditions = [
        (S_cap * reservoir_data["NORhi_mu"] / 100),
        (S_cap * reservoir_data["NORhi_mu"] * 1.2 / 100),
        (S_cap * reservoir_data["NORhi_mu"] * 0.6 / 100),
    ]

    # parse resops data
    resops_inflow = resops_data["inflow"][data_start:data_end].values * 22.824
    resops_inflow[resops_inflow < 0] = 0
    resops_outflow = resops_data["outflow"][data_start:data_end].values * 22.824
    resops_outflow[resops_outflow < 0] = 0

    # Blue Marsh storage is in MCM (convert to MG)
    if i == 4:
        resops_storage = resops_data["storage"][data_start:data_end].values * 264.172

    # Beltzville is CF and need to convert to MG (only at the end of period)
    elif i == 5:
        resops_storage = resops_data["storage"][data_start:data_end].values * 7.48052

    # Select one of the inflow timeseries
    sim_inflow = resops_inflow
    inflow_lab = "ResOpsUS Most Recent Year"

    # Initialize vectors
    R_avg = np.zeros(n_time)
    NOR_hi_harmonic = np.zeros(n_time)
    NOR_lo_harmonic = np.zeros(n_time)

    for d in range(n_time):
        R_avg[d] = release_harmonic(reservoir_data, d, timestep=time_step)
        NOR_hi_harmonic[d] = NOR_hi(reservoir_data, d, timestep=time_step)
        NOR_lo_harmonic[d] = NOR_lo(reservoir_data, d, timestep=time_step)

    # Initialize vectors for multiple sims
    S = np.zeros((n_sims, n_time))
    S_percent = np.zeros((n_sims, n_time))
    R = np.zeros((n_sims, n_time))

    S_initial = resops_storage[0]

    result = starfit_reservoir_simulation(starfit, reservoirs[i], sim_inflow, S_initial)

    S = result["storage"]
    R = result["outflow"]

    # Calculate error
    error = error_calculation(resops_outflow, R, error_type="RMSE")

    # Plot outputs
    x = np.arange(n_time)
    sim_labs = [
        "%$S_t$ for mean $S_i$",
        "%$S_t$ for high $S_i$",
        "%$S_t$ for low $S_i$",
    ]

    # Plot the inflow timeseries
    plt.plot(range(t), sim_inflow)
    plt.xlabel(x_lab)
    plt.ylabel("Inflow (MGD)")
    plt.title(str(f"{reservoir_lab}\n" + inflow_lab))
    plt.savefig(f"./figures/{reservoir_lab}_inflow.png")
    plt.show()

    # Seasonal release harmonic
    plt.plot(x, R_avg)
    plt.xlabel("Proportional Release")
    plt.title(f"{reservoir_lab} Reservoir\nAverage Weekly Release Harmonic")
    plt.xlabel(x_lab)
    # plt.savefig(f'./figures/{reservoir_lab}_release_harmonic.png')
    plt.show()

    # Harmonic NOR
    plt.plot(
        x,
        NOR_hi_harmonic,
        label="NOR Bounds",
        color="black",
        alpha=0.2,
        linestyle="dashed",
    )
    plt.plot(x, NOR_lo_harmonic, color="black", alpha=0.2, linestyle="dashed")
    plt.title(f"{reservoir_lab} Reservoir\nNormal Operating Range")
    plt.ylabel("Percent Storage Capacity (%)")
    plt.xlabel(x_lab)
    plt.legend()
    plt.savefig(f"./figures/{reservoir_lab}_NOR.png")
    plt.show()

    # Simulated storage with NOR
    plt.plot(x, S, label="Storage")
    plt.title(f"{reservoir_lab} Reservoir\nSimulated Storage:\n {inflow_lab}")
    plt.ylabel("Storage Volume (MG)")
    plt.xlabel(x_lab)
    plt.legend()
    plt.savefig(f"./figures/{reservoir_lab}_sim_storage.png")
    plt.show()

    # Simulated releases
    plt.plot(x, R, label="Release")
    plt.plot(
        x,
        (np.ones(n_time) * R_max),
        label="$R_{max}$",
        color="black",
        alpha=0.2,
        linestyle="dashed",
    )
    plt.plot(
        x,
        (np.ones(n_time) * R_min),
        label="$R_{min}$",
        color="black",
        alpha=0.5,
        linestyle="dashed",
    )
    plt.title(f"{reservoir_lab} Reservoir\nSimulated Release Actions:\n {inflow_lab}")
    plt.ylabel("Release Volume (MGD)")
    plt.xlabel(x_lab)
    plt.legend()
    plt.savefig(f"./figures/{reservoir_lab}_sim_releases.png")
    plt.show()

    # Compare simulated with actual outflow

    plt.plot(x, R, label="Sim. R")
    plt.plot(
        range(len(resops_outflow)),
        resops_outflow,
        color="red",
        label="Observed",
        linestyle="dashed",
    )
    text_x_coord = 0.75 * len(resops_outflow)
    text_y_coord = 0.75 * max(resops_outflow)
    plt.text(text_x_coord, text_y_coord, str(f"RMSE: {error:.2f}"))
    plt.title(
        f"{reservoir_lab} Reservoir\nComparison between observed outflow and simulated releases"
    )
    plt.xlabel(x_lab)
    plt.ylabel("Flow (MGD)")
    plt.legend()
    plt.savefig(f"./figures/{reservoir_lab}_compare_outflows.png")
    plt.show()

    # Compare simulated and actual storage

    plt.plot(x, S, label="Sim. S")
    plt.plot(x, resops_storage, label="Obs. S", color="red", linestyle="dashed")
    plt.plot(
        x,
        (NOR_hi_harmonic * S_cap),
        label="NOR Bounds",
        color="black",
        alpha=0.2,
        linestyle="dashed",
    )
    plt.plot(x, (NOR_lo_harmonic * S_cap), color="black", alpha=0.2, linestyle="dashed")
    plt.title(
        f"{reservoir_lab} Reservoir\nComparison between observed and simulated storage"
    )
    plt.ylabel("Storage Volume (MG)")
    plt.xlabel(x_lab)
    plt.legend()
    plt.savefig(f"./figures/{reservoir_lab}_compare_storages.png")
    plt.show()
