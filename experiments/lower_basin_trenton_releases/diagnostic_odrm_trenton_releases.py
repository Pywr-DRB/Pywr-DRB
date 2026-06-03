"""
Diagnostic: Compare simulated Beltzville/Blue Marsh Trenton MRF contributions
against reported ODRM releases from trenton_equiv_flow_obj_odrm_releases.csv.

Also compares observed vs simulated Trenton flow.
Uses pub_nhmv10_BC_withObsScaled dataset, restricted to the ODRM data period (2009+).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywrdrb
from pywrdrb.utils.constants import cfs_to_mgd
from pywrdrb.utils.dates import model_date_ranges

# ---- Configuration ----
inflow_type = "pub_nhmv10_BC_withObsScaled"
start_date, end_date = model_date_ranges[inflow_type]
output_dir = "output_data"
output_file = os.path.join(output_dir, f"{inflow_type}.hdf5")
fig_file = os.path.join(output_dir, "diagnostic_odrm_trenton_releases.svg")
os.makedirs(output_dir, exist_ok=True)

# ---- Step 1: Run simulation if output doesn't exist ----
if not os.path.exists(output_file):
    print(f"Running {inflow_type} simulation...")
    mb = pywrdrb.ModelBuilder(
        inflow_type=inflow_type,
        start_date=start_date,
        end_date=end_date,
        options={"flow_prediction_mode": 'perfect_foresight'},
    )
    mb.make_model()
    model_file = os.path.join(output_dir, f"{inflow_type}_model.json")
    mb.write_model(model_file)
    model = pywrdrb.Model.load(model_file)
    recorder = pywrdrb.OutputRecorder(
        model=model,
        output_filename=output_file,
    )
    model.run()
    os.remove(model_file)
    print("Simulation complete.")
else:
    print(f"Using existing output: {output_file}")

# ---- Step 2: Load simulation results ----
data = pywrdrb.Data()
data.load_output(
    output_filenames=[output_file],
    results_sets=["major_flow", "lower_basin_mrf_contributions"],
)
label = list(data.major_flow.keys())[0]
mrf_df = data.lower_basin_mrf_contributions[label][0]
flow_df = data.major_flow[label][0]

sim_beltz = mrf_df["mrf_trenton_beltzvilleCombined"].rename("beltzville_sim")
sim_bm = mrf_df["mrf_trenton_blueMarsh"].rename("blueMarsh_sim")
sim_trenton = flow_df["delTrenton"].rename("trenton_sim")

sim_combined = sim_beltz + sim_bm
sim_combined.name = "sim_combined"

# ---- Step 3: Load ODRM reported releases ----
odrm = pd.read_csv("trenton_equiv_flow_obj_odrm_releases.csv")
odrm["date"] = pd.to_datetime(odrm["Corrected date"])
odrm = odrm.set_index("date").sort_index()
odrm_mgd = odrm["Value"] * cfs_to_mgd
odrm_mgd.name = "odrm_reported"

# ---- Step 4: Load observed Trenton flow ----
data = pywrdrb.Data(results_sets=["major_flow"])
data.load_observations()
obs_trenton = data.major_flow["obs"][0]["delTrenton"]
obs_trenton.name = "trenton_obs"

# ---- Step 5: Align to overlapping period ----
overlap_start = max(sim_combined.index.min(), odrm_mgd.index.min())
overlap_end = min(odrm_mgd.index.max(), sim_combined.index.max())

df = pd.DataFrame({
    "sim_combined": sim_combined,
    "odrm_reported": odrm_mgd,
    "trenton_sim": sim_trenton,
    "trenton_obs": obs_trenton,
}).loc[overlap_start:overlap_end].dropna(how="all")

# ---- Step 6: Resample to weekly and monthly ----
df_weekly = df.resample("W").sum()
df_monthly = df.resample("MS").sum()

# ---- Step 7: Plot ----
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")

# Top row: ODRM releases comparison
for ax, df_agg, label in zip(axes[0], [df_weekly, df_monthly], ["Weekly", "Monthly"]):
    ax.plot(df_agg.index, df_agg["odrm_reported"], label="ODRM Reported", color="k", linewidth=1)
    ax.plot(df_agg.index, df_agg["sim_combined"], label="Simulated (Beltz + BM)", color="tab:blue", alpha=0.8, linewidth=1)
    ax.set_ylabel("Flow (MGD)")
    ax.set_title(f"{label} Total: Lower Basin Trenton MRF Contributions")
    ax.legend(fontsize=8)

# Bottom row: Trenton flow comparison
for ax, df_agg, label in zip(axes[1], [df_weekly, df_monthly], ["Weekly", "Monthly"]):
    ax.plot(df_agg.index, df_agg["trenton_obs"], label="Observed", color="k", linewidth=1)
    ax.plot(df_agg.index, df_agg["trenton_sim"], label="Simulated", color="tab:red", alpha=0.8, linewidth=1)
    ax.set_ylabel("Flow (MGD)")
    ax.set_title(f"{label} Total: Trenton Flow")
    ax.legend(fontsize=8)
    ax.set_xlabel("Date")

plt.suptitle(f"ODRM Release Diagnostic ({inflow_type})", fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig(fig_file, format="svg", bbox_inches="tight")
plt.show()
print(f"Figure saved: {fig_file}")
