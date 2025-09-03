import os
import time
import logging
from itertools import product
import pywrdrb

# === SETUP LOGGING === #
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# === CONFIGURATION === #
cwd = os.getcwd()
results_dir = os.path.join(cwd, "model_runs")
fig_dir = os.path.join(results_dir, "diagnostics", "release_timeseries")
os.makedirs(results_dir, exist_ok=True)

inflow_type = 'pub_nhmv10_BC_withObsScaled'
start_date = "1983-10-01"
end_date = "2023-12-31"

reservoirs = ["beltzvilleCombined", "prompton", "fewalter"]
policy_types = {
    "beltzvilleCombined": ["STARFITReservoirRelease", "PWLReservoirRelease", "RBFReservoirRelease"],
    "prompton": ["STARFITReservoirRelease", "PWLReservoirRelease", "RBFReservoirRelease"],
    "fewalter": ["STARFITReservoirRelease", "PWLReservoirRelease", "RBFReservoirRelease"]
}

label_map = {
    "Release_NSE": "Best Release NSE",
    "Storage_NSE": "Best Storage NSE",
    "q20": "Best q20 Bias",
    "q80": "Best q80 Bias",
    "composite": "Best Overall"
}
metric_keys = list(label_map.keys())

drought_periods = {
    "Drought: 1995": ("1995-01-01", "1995-12-31"),
    "Drought: 1998–2002": ("1998-01-01", "2002-12-31"),
    "Drought: 2016": ("2016-01-01", "2016-12-31"),
    "Drought: 2022–2023": ("2022-01-01", "2023-12-31"),
    "Non-Drought: 1983–1994": ("1983-10-01", "1994-12-31"),
    "Non-Drought: 1996–1997": ("1996-01-01", "1997-12-31"),
    "Non-Drought: 2003–2015": ("2003-01-01", "2015-12-31"),
    "Non-Drought: 2017–2021": ("2017-01-01", "2021-12-31"),
    "Management: 2017–2023": ("2017-01-01", "2023-12-31"),
    "Full Period: 1983–2023": ("1983-10-01", "2023-12-31"),
    "Subset: 2017–2018": ("2017-01-01", "2018-12-31"),
    "Subset: 2019–2020": ("2019-01-01", "2020-12-31"),
    "Subset: 2021–2023": ("2021-01-01", "2023-12-31")
}

metadata_file = os.path.join(results_dir, "simulation_metadata.csv")
if not os.path.exists(metadata_file):
    with open(metadata_file, "w") as f:
        f.write("reservoir,policy_type,metric_key,metric_label,runtime_sec,output_size_MB\n")

output_files = {}

# === MAIN LOOP === #
for reservoir, policy_type_list in policy_types.items():
    for policy_type, metric_key in product(policy_type_list, metric_keys):
        metric_label = label_map[metric_key]
        run_id = f"{reservoir}_{policy_type}_{metric_key}"

        try:
            logger.info(f"Starting simulation: {run_id} ({metric_label})")

            # Build release policy options
            release_policy_dict = {
                reservoir: {
                    "type": policy_type,
                    "id": metric_label
                }
            }
            options = {"release_policy_dict": release_policy_dict}

            # === Build model === #
            mb = pywrdrb.ModelBuilder(
                start_date=start_date,
                end_date=end_date,
                inflow_type=inflow_type,
                options=options
            )
            mb.make_model()

            model_filename = os.path.join(results_dir, f"{run_id}_model.json")
            mb.write_model(model_filename)
            print(f"Model written to {model_filename}")

            # === Load and run model === #
            model = pywrdrb.Model.load(model_filename)
            output_filename = os.path.join(results_dir, f"{run_id}_output.hdf5")
            recorder = pywrdrb.OutputRecorder(model, output_filename)

            print("Running the model...")
            t_start = time.perf_counter()
            stats = model.run()
            t_end = time.perf_counter()
            runtime = t_end - t_start

            # Save the output path by policy
            base = os.path.splitext(os.path.basename(output_filename))[0]
            output_files[(reservoir, policy_type, metric_key)] = base

            assert os.path.exists(output_filename), f"Output not found for {run_id}"
            logger.info(f"Output saved: {output_filename}")
            logger.info(f"⏱Runtime: {runtime:.2f} sec")

            # === Log metadata === #
            with open(metadata_file, "a") as f:
                f.write(f"{reservoir},{policy_type},{metric_key},{metric_label},{runtime:.2f},{os.path.getsize(output_filename)/1e6:.2f}\n")

        except Exception as e:
            logger.error(f"Failed: {run_id}")
            logger.error(f"   Reason: {e}")
            continue

logger.info("All simulations complete.")

# Run default model 
mb = pywrdrb.ModelBuilder(
    start_date=start_date,
    end_date=end_date,
    inflow_type=inflow_type
)
mb.make_model()
model_filename = os.path.join(cwd, "default_model.json")

# Save the model to a JSON file
mb.write_model(model_filename)
# Load the model from the JSON file
model = pywrdrb.Model.load(model_filename)

# Attach an Output Recorder
output_filename = os.path.join(cwd,"pywrdrb_default_output.hdf5")
recorder = pywrdrb.OutputRecorder(model, output_filename)

# Run the model
stats = model.run()

# Check if the output file was created successfully
assert os.path.exists(output_filename), "Simulation output not found."
print(" Simulation completed and output file saved:", output_filename)


# === Load results === #
results_sets = ['major_flow', 'res_storage', 'reservoir_downstream_gage', 'lower_basin_mrf_contributions']

# === Initialize and load model output === #
data = pywrdrb.Data(
    print_status=True,
    results_sets=results_sets,
    output_filenames=[os.path.join(results_dir, f"{fname}.hdf5") for fname in output_files.values()]
)
print("Loading simulation output...")
# Load simulation output
data.load_output()

print("Simulation output loaded successfully.")
print("Available results sets:", data.results_sets)
# Create dictionaries to hold each policy's outputs
major_flow_df = {}
res_storage_df = {}
downstream_gage_df = {}
mrf_contrib_df = {}

for (reservoir, policy_type, metric_key), fname in output_files.items():
    key = f"{policy_type}_{metric_key}"  # this is just your label for later
    print(f"Loading data for {key} from {fname}...")
    major_flow_df[key] = data.major_flow[fname][0]
    res_storage_df[key] = data.res_storage[fname][0]
    downstream_gage_df[key] = data.reservoir_downstream_gage[fname][0]
    mrf_contrib_df[key] = data.lower_basin_mrf_contributions[fname][0]
    print(f"Data for {key} loaded successfully.")


print("Data loaded for all policies and metrics.")
print("Major flow DataFrame keys:", list(major_flow_df.keys()))

#Load default model output
# === Initialize and load model output === #
data = pywrdrb.Data(
    print_status=True,
    results_sets=results_sets,
    output_filenames=[output_filename]
)

# Load simulation output
data.load_output()

# === Extract dataframes for plotting === #
default_major_flow = data.major_flow["pywrdrb_default_output"][0]
default_res_storage = data.res_storage["pywrdrb_default_output"][0]
default_reservoir_downstream_gage = data.reservoir_downstream_gage["pywrdrb_default_output"][0]
default_lower_basin_mrf_contributions = data.lower_basin_mrf_contributions["pywrdrb_default_output"][0]

print("Default model output loaded successfully.")
# Load observed data if needed
data.load_observations()
print("Observations loaded successfully.")
# Access observed timeseries
df_obs_major_flow = data.major_flow["obs"][0]
df_obs_res_storage = data.res_storage["obs"][0]
df_obs_reservoir_downstream_gage = data.reservoir_downstream_gage["obs"][0]
print("going to plot reservoir release diagnostics...")

# Plotting functions 
def plot_reservoir_release_diagnostics_multi(
    reservoir,
    sim_dict,
    obs_df=None,
    start=None,
    end=None,
    ylabel="Flow (MGD)",
    save_path=None
):
    """
    Plot 6-panel diagnostics comparing observed vs. multiple simulated reservoir releases.
    Includes time series and flow duration curves (daily, monthly, annual).

    Parameters
    ----------
    reservoir : str
        Name of the reservoir to plot.
    sim_dict : dict[str, pd.DataFrame]
        Dictionary of simulated release DataFrames {policy_label: dataframe}.
    obs_df : pd.DataFrame, optional
        Observed release DataFrame (must include `reservoir` column).
    start : str or pd.Timestamp
        Start date for plotting.
    end : str or pd.Timestamp
        End date for plotting.
    save_path : str
        Path to save PNG. If None, just displays the plot.
    """

    import matplotlib.pyplot as plt
    import numpy as np

    # Check date range
    if start is None or end is None:
        raise ValueError("Please provide both start and end dates.")

    # Resampling and FDC setup
    def get_fdc(data):
        sorted_vals = np.sort(data.dropna())[::-1]
        prob = np.linspace(0, 100, len(sorted_vals))
        return prob, sorted_vals

    # --- Prep observed ---
    obs = None
    if obs_df is not None and reservoir in obs_df.columns:
        obs = obs_df[reservoir].loc[start:end]
        if obs.dropna().empty:
            obs = None

    # --- Prep simulated ---
    sim_ts = {k: df.loc[start:end] for k, df in sim_dict.items()}
    sim_monthly = {k: v.resample("ME").mean() for k, v in sim_ts.items()}
    sim_annual = {k: v.resample("YE").mean() for k, v in sim_ts.items()}
    sim_climatology = {k: v.groupby(v.index.month).mean() for k, v in sim_monthly.items()}
    sim_fdcs = {
        k: {
            "daily": get_fdc(v),
            "monthly": get_fdc(sim_monthly[k]),
            "annual": get_fdc(sim_annual[k])
        }
        for k, v in sim_ts.items()
    }

    if obs is not None:
        monthly_obs = obs.resample("ME").mean()
        annual_obs = obs.resample("YE").mean()
        obs_climatology = monthly_obs.groupby(monthly_obs.index.month).mean()
        obs_fdcs = {
            "daily": get_fdc(obs),
            "monthly": get_fdc(monthly_obs),
            "annual": get_fdc(annual_obs)
        }

    # === Plot === #
    fig, axs = plt.subplots(3, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Time Series
    if obs is not None:
        axs[0, 0].plot(obs, label="Observed", lw=1.5, color="black")
    for k, ts in sim_ts.items():
        axs[0, 0].plot(ts, label=k, lw=1)
    axs[0, 0].set_title(f"{reservoir}: Daily Time Series")
    axs[0, 0].set_ylabel(ylabel)
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # FDC - Daily
    if obs is not None:
        axs[0, 1].plot(*obs_fdcs["daily"], label="Observed", color="black")
    for k in sim_fdcs:
        axs[0, 1].plot(*sim_fdcs[k]["daily"], label=k)
    axs[0, 1].set_title(f"{reservoir}: Daily FDC")
    axs[0, 1].set_xlabel("Exceedance Probability (%)")
    axs[0, 1].set_yscale("log")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Monthly climatology
    if obs is not None:
        axs[1, 0].plot(month_names, obs_climatology.values, label="Observed", lw=2, marker="o", color="black")
    for k, v in sim_climatology.items():
        axs[1, 0].plot(month_names, v.values, label=k, lw=1.2, marker="o")
    axs[1, 0].set_title(f"{reservoir}: Monthly Climatology")
    axs[1, 0].set_ylabel(ylabel)
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # FDC - Monthly
    if obs is not None:
        axs[1, 1].plot(*obs_fdcs["monthly"], label="Observed", color="black")
    for k in sim_fdcs:
        axs[1, 1].plot(*sim_fdcs[k]["monthly"], label=k)
    axs[1, 1].set_title(f"{reservoir}: Monthly FDC")
    axs[1, 1].set_xlabel("Exceedance Probability (%)")
    axs[1, 1].set_yscale("log")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Annual time series
    if obs is not None:
        axs[2, 0].plot(annual_obs.index.year, annual_obs.values, label="Observed", lw=2, marker='o', color="black")
    for k, v in sim_annual.items():
        axs[2, 0].plot(v.index.year, v.values, label=k, lw=1.5, marker='o')
    axs[2, 0].set_title(f"{reservoir}: Annual Average")
    axs[2, 0].set_xlabel("Year")
    axs[2, 0].set_ylabel(ylabel)
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # FDC - Annual
    if obs is not None:
        axs[2, 1].plot(*obs_fdcs["annual"], label="Observed", color="black")
    for k in sim_fdcs:
        axs[2, 1].plot(*sim_fdcs[k]["annual"], label=k)
    axs[2, 1].set_title(f"{reservoir}: Annual FDC")
    axs[2, 1].set_xlabel("Exceedance Probability (%)")
    axs[2, 1].set_yscale("log")
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    fig.suptitle(f"{reservoir} – Release Diagnostics ({start} to {end})", fontsize=15, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    
    # Save or show the plot
    print(f"Saving plot to {save_path}")
    if save_path is None:
        save_path = os.path.join(fig_dir, f"{reservoir}_release_diagnostics.png")
    else:
        save_path = os.path.join(fig_dir, save_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_reservoir_release_storage_diagnostics_multi(
    reservoir,
    sim_release_dict,
    sim_storage_dict,
    obs_release_df=None,
    obs_storage_df=None,
    start=None,
    end=None,
    ylabel="Flow (MGD)",
    storage_ylabel="Storage (%)",
    save_path=None
):
    import matplotlib.pyplot as plt
    import numpy as np

    def get_fdc(data):
        sorted_vals = np.sort(data.dropna())[::-1]
        prob = np.linspace(0, 100, len(sorted_vals))
        return prob, sorted_vals

    if start is None or end is None:
        raise ValueError("Please provide start and end dates.")

    # --- Observed ---
    obs_release = None
    obs_storage = None
    if obs_release_df is not None and reservoir in obs_release_df.columns:
        obs_release = obs_release_df[reservoir].loc[start:end]
        if obs_release.dropna().empty:
            obs_release = None
    if obs_storage_df is not None and reservoir in obs_storage_df.columns:
        obs_storage = obs_storage_df[reservoir].loc[start:end]
        if obs_storage.dropna().empty:
            obs_storage = None

    # --- Simulated ---
    sim_release_ts = {k: df.loc[start:end] for k, df in sim_release_dict.items()}
    sim_storage_ts = {k: df.loc[start:end] for k, df in sim_storage_dict.items()}

    # --- Monthly and Annual Averages ---
    sim_release_monthly = {k: v.resample("ME").mean() for k, v in sim_release_ts.items()}
    sim_release_annual = {k: v.resample("YE").mean() for k, v in sim_release_ts.items()}
    sim_storage_monthly = {k: v.resample("ME").mean() for k, v in sim_storage_ts.items()}
    sim_storage_annual = {k: v.resample("YE").mean() for k, v in sim_storage_ts.items()}

    sim_release_clim = {k: v.groupby(v.index.month).mean() for k, v in sim_release_monthly.items()}
    sim_storage_clim = {k: v.groupby(v.index.month).mean() for k, v in sim_storage_monthly.items()}

    sim_release_fdc = {
        k: {
            "daily": get_fdc(ts),
            "monthly": get_fdc(sim_release_monthly[k]),
            "annual": get_fdc(sim_release_annual[k])
        }
        for k, ts in sim_release_ts.items()
    }

    if obs_release is not None:
        monthly_obs_release = obs_release.resample("ME").mean()
        annual_obs_release = obs_release.resample("YE").mean()
        obs_release_clim = monthly_obs_release.groupby(monthly_obs_release.index.month).mean()
        obs_release_fdc = {
            "daily": get_fdc(obs_release),
            "monthly": get_fdc(monthly_obs_release),
            "annual": get_fdc(annual_obs_release)
        }

    if obs_storage is not None:
        monthly_obs_storage = obs_storage.resample("ME").mean()
        annual_obs_storage = obs_storage.resample("YE").mean()
        obs_storage_clim = monthly_obs_storage.groupby(monthly_obs_storage.index.month).mean()

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # === Plot === #
    fig, axs = plt.subplots(3, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Row 1: Storage Time Series
    for k, ts in sim_storage_ts.items():
        axs[0, 0].plot(ts, label=k)
    if obs_storage is not None:
        axs[0, 0].plot(obs_storage, label="Observed", color="black", linewidth=2)
    axs[0, 0].set_title(f"{reservoir}: Daily Storage Time Series")
    axs[0, 0].set_ylabel(storage_ylabel)
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Row 1: Release Time Series
    for k, ts in sim_release_ts.items():
        axs[0, 1].plot(ts, label=k)
    if obs_release is not None:
        axs[0, 1].plot(obs_release, label="Observed", color="black", linewidth=2)
    axs[0, 1].set_title(f"{reservoir}: Daily Release Time Series")
    axs[0, 1].set_ylabel(ylabel)
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Row 1: Daily FDC
    for k in sim_release_fdc:
        axs[0, 2].plot(*sim_release_fdc[k]["daily"], label=k)
    if obs_release is not None:
        axs[0, 2].plot(*obs_release_fdc["daily"], label="Observed", color="black", linewidth=2)
    axs[0, 2].set_title(f"{reservoir}: Daily Release FDC")
    axs[0, 2].set_yscale("log")
    axs[0, 2].set_xlabel("Exceedance Probability (%)")
    axs[0, 2].legend()
    axs[0, 2].grid(True)

    # Row 2: Monthly Climatology Storage
    for k, v in sim_storage_clim.items():
        axs[1, 0].plot(month_names, v.values, label=k, marker="o")
    if obs_storage is not None:
        axs[1, 0].plot(month_names, obs_storage_clim.values, label="Observed", color="black", marker="o")
    axs[1, 0].set_title(f"{reservoir}: Monthly Avg Storage")
    axs[1, 0].set_ylabel(storage_ylabel)
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Row 2: Monthly Climatology Release
    for k, v in sim_release_clim.items():
        axs[1, 1].plot(month_names, v.values, label=k, marker="o")
    if obs_release is not None:
        axs[1, 1].plot(month_names, obs_release_clim.values, label="Observed", color="black", marker="o")
    axs[1, 1].set_title(f"{reservoir}: Monthly Avg Release")
    axs[1, 1].set_ylabel(ylabel)
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Row 2: Monthly FDC
    for k in sim_release_fdc:
        axs[1, 2].plot(*sim_release_fdc[k]["monthly"], label=k)
    if obs_release is not None:
        axs[1, 2].plot(*obs_release_fdc["monthly"], label="Observed", color="black", linewidth=2)
    axs[1, 2].set_title(f"{reservoir}: Monthly Release FDC")
    axs[1, 2].set_yscale("log")
    axs[1, 2].set_xlabel("Exceedance Probability (%)")
    axs[1, 2].legend()
    axs[1, 2].grid(True)

    # Row 3: Annual Avg Storage
    for k, v in sim_storage_annual.items():
        axs[2, 0].plot(v.index.year, v.values, label=k, marker='o')
    if obs_storage is not None:
        axs[2, 0].plot(annual_obs_storage.index.year, annual_obs_storage.values, label="Observed", color="black", marker='o')
    axs[2, 0].set_title(f"{reservoir}: Annual Avg Storage")
    axs[2, 0].set_ylabel(storage_ylabel)
    axs[2, 0].set_xlabel("Year")
    axs[2, 0].legend()
    axs[2, 0].grid(True)

    # Row 3: Annual Avg Release
    for k, v in sim_release_annual.items():
        axs[2, 1].plot(v.index.year, v.values, label=k, marker='o')
    if obs_release is not None:
        axs[2, 1].plot(annual_obs_release.index.year, annual_obs_release.values, label="Observed", color="black", marker='o')
    axs[2, 1].set_title(f"{reservoir}: Annual Avg Release")
    axs[2, 1].set_ylabel(ylabel)
    axs[2, 1].set_xlabel("Year")
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    # Row 3: Annual FDC
    for k in sim_release_fdc:
        axs[2, 2].plot(*sim_release_fdc[k]["annual"], label=k)
    if obs_release is not None:
        axs[2, 2].plot(*obs_release_fdc["annual"], label="Observed", color="black", linewidth=2)
    axs[2, 2].set_title(f"{reservoir}: Annual Release FDC")
    axs[2, 2].set_yscale("log")
    axs[2, 2].set_xlabel("Exceedance Probability (%)")
    axs[2, 2].legend()
    axs[2, 2].grid(True)

    fig.suptitle(f"{reservoir} – Release & Storage Diagnostics ({start} to {end})", fontsize=16, y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Save or show the plot
    print(f"Saving plot to {save_path}")
    if save_path is None:
        save_path = os.path.join(fig_dir, f"{reservoir}_release_diagnostics.png")
    else:
        save_path = os.path.join(fig_dir, save_path)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if save_path:
        fig.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()


for reservoir in reservoirs:
    for metric_key in metric_keys:
        metric_label = label_map[metric_key]
        print(f"Processing {reservoir} | {metric_label}")
        for period_label, (period_start, period_end) in drought_periods.items():
            logger.info(f" Plotting: {reservoir} | {metric_label} | {period_label}")
            print(f"Period: {period_start} to {period_end}")
            # === Initialize sim dicts ===
            sim_release_dict = {}
            sim_storage_dict = {}

            for policy_type in policy_types[reservoir]:
                short_policy = policy_type.replace("ReservoirRelease", "")
                line_label = f"{policy_type.replace('ReservoirRelease', '')} ({label_map[metric_key]})"

                
                
                try:
                    combined_key = f"{policy_type}_{metric_key}"
                    release_series = downstream_gage_df[combined_key][reservoir]
                    storage_series = res_storage_df[combined_key][reservoir]
                    line_label = f"{policy_type.replace('ReservoirRelease', '')} ({label_map[metric_key]})"

                    sim_release_dict[line_label] = release_series
                    sim_storage_dict[line_label] = storage_series
                except KeyError:
                    logger.warning(f"Missing data for {reservoir} | {policy_type} | {metric_key} — skipping.")
                    continue
  

            # === Add Default simulation ===
            if reservoir in default_reservoir_downstream_gage.columns:
                sim_release_dict["Default"] = default_reservoir_downstream_gage[reservoir]
            if reservoir in default_res_storage.columns:
                sim_storage_dict["Default"] = default_res_storage[reservoir]

            # === Save Paths ===
            file_prefix = f"{reservoir}_{metric_key}_{period_label.replace(':','').replace('–','_').replace(' ','_')}"
            release_save_path = os.path.join(fig_dir, f"{file_prefix}_release_timeseries.png")
            release_storage_save_path = os.path.join(fig_dir, f"{file_prefix}_release_storage_timeseries.png")
            print(f"Saving plots to {release_save_path} and {release_storage_save_path}")
            # === PLOT ===
            plot_reservoir_release_diagnostics_multi(
                reservoir=reservoir,
                sim_dict=sim_release_dict,
                obs_df=df_obs_reservoir_downstream_gage,
                start=period_start,
                end=period_end,
                save_path=release_save_path
            )

            plot_reservoir_release_storage_diagnostics_multi(
                reservoir=reservoir,
                sim_release_dict=sim_release_dict,
                sim_storage_dict=sim_storage_dict,
                obs_release_df=df_obs_reservoir_downstream_gage,
                obs_storage_df=df_obs_res_storage,
                start=period_start,
                end=period_end,
                save_path=release_storage_save_path
            )
print("All plots generated successfully.")