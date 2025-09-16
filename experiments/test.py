import os
import pandas as pd
import matplotlib.pyplot as plt
import pywrdrb


# -------------------------------
# CHOICES: reservoir, policy, CSV
# -------------------------------

# Pick the DPS CSV you saved earlier (match policy + selection)
#TS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "sim_timeseries")
#DPS_CSV = os.path.join(TS_DIR, f"ts_{RES}_PiecewiseLinear_Best_Release_NSE.csv")
#assert os.path.exists(DPS_CSV), f"Missing DPS CSV: {DPS_CSV}"

# === BELTZVILLE (beltzvilleCombined) — PWL (Set 2, 15) ======================
RES = "beltzvilleCombined"
POLICY_CLASS = "ParametricReservoirRelease"
POLICY_TYPE  = "PWL"

PWL_PARAMS = (
    "0.458,0.7779,-1.0162,1.1401,1.5503,"
    "0.3208,0.9149,-0.0391,-0.3787,-1.2765,"
    "0.2863,0.8809,0.1839,0.1412,-1.1661"
)


inflow_type = "pub_nhmv10_BC_withObsScaled"
start_date  = "1983-10-01"
end_date    = "2023-12-31"

# -------------------------------
# Build model
# -------------------------------
options = {
    "release_policy_dict": {
        RES: {
            "class_type": POLICY_CLASS,   # ParametricReservoirRelease
            "policy_type": POLICY_TYPE,   # PWL
            "policy_id":   "inline",      # any label you want
            "params":      PWL_PARAMS,   # <-- inline vector!
        }
    }
}

cwd = os.getcwd()
print(f"cwd: {cwd}")

mb = pywrdrb.ModelBuilder(
    start_date=start_date,
    end_date=end_date,
    inflow_type=inflow_type,
    options=options,
)
mb.make_model()

model_filename = os.path.join(cwd, f"model_{POLICY_CLASS}_{POLICY_TYPE}_{RES}.json")
mb.write_model(model_filename)
print(f"Model written to {model_filename}")

# -------------------------------
# Run model
# -------------------------------
model = pywrdrb.Model.load(model_filename)
output_filename = os.path.join(cwd, f"output_{POLICY_CLASS}_{POLICY_TYPE}_{RES}.hdf5")
recorder = pywrdrb.OutputRecorder(model, output_filename)
print(f"Output will be saved to {output_filename}")

stats = model.run()
assert os.path.exists(output_filename), "Simulation output not found."
print(f"Simulation completed → {output_filename}")

# -------------------------------
# Load model output + observations
# -------------------------------
results_sets = [
    "major_flow",
    "res_storage",
    "reservoir_downstream_gage",
    "lower_basin_mrf_contributions",
]

data = pywrdrb.Data(
    print_status=True,
    results_sets=results_sets,
    output_filenames=[os.path.splitext(os.path.basename(output_filename))[0] + ".hdf5"],
)
data.load_output()
data.load_observations()

# --------------------------------
# Identify the output "base" once
# --------------------------------
base = os.path.splitext(os.path.basename(output_filename))[0]

# ===============================
# Ensure obs gage has 'prompton' by pulling NWIS DV if missing
# ===============================
# Use sim gage (already loaded) for the date window and alignment
df_sim_gage = data.reservoir_downstream_gage[base][0]   # simulated gage flows (per-reservoir columns)
df_obs_gage = data.reservoir_downstream_gage["obs"][0]  # observed gage flows (may be missing 'prompton')

def _make_tz_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx

if RES.lower() == "prompton" and ("prompton" not in df_obs_gage.columns or df_obs_gage["prompton"].isna().all()):
    try:
        from dataretrieval import nwis

        idx_sim = _make_tz_naive(df_sim_gage.index)
        s_date  = idx_sim.min().strftime("%Y-%m-%d")
        e_date  = idx_sim.max().strftime("%Y-%m-%d")

        NWIS_SITE = "01429000"   # Prompton Lake outlet gage
        PARAMETER = "00060"      # discharge (cfs), daily values
        print(f"Retrieving Prompton ({NWIS_SITE}) DV {PARAMETER} from {s_date} to {e_date} ...")

        df_raw = nwis.get_record(
            sites=NWIS_SITE,
            start=s_date,
            end=e_date,
            parameterCd=PARAMETER,
            service="dv",
        )

        if df_raw is None or len(df_raw) == 0:
            print("NWIS returned no DV data for Prompton; leaving obs unchanged.")
        else:
            df_raw.index = _make_tz_naive(pd.to_datetime(df_raw.index))
            df_raw.sort_index(inplace=True)

            # pick the "Mean" 00060 column if present, else the first non-qualifier col
            col = next((c for c in df_raw.columns if "00060" in c and ("Mean" in c or c.endswith("_Mean"))), None)
            if col is None:
                non_qual = [c for c in df_raw.columns if "qual" not in c.lower()]
                col = non_qual[0] if non_qual else df_raw.columns[0]

            df_prompton = df_raw[[col]].rename(columns={col: "prompton"}).astype(float)

            # convert cfs → MGD
            CFS_TO_MGD = 0.646317
            df_prompton["prompton"] = df_prompton["prompton"] * CFS_TO_MGD

            # align to obs index if it exists, otherwise align to sim index
            target_index = df_obs_gage.index if len(df_obs_gage.index) else idx_sim
            df_prompton = df_prompton.reindex(target_index)

            # write back into the Data container
            df_obs_gage = df_obs_gage.copy()
            if len(df_obs_gage.index) == 0:
                # create an empty frame with the target index if needed
                df_obs_gage = pd.DataFrame(index=target_index)
            df_obs_gage.loc[:, "prompton"] = df_prompton["prompton"].values
            data.reservoir_downstream_gage["obs"][0] = df_obs_gage

            print("Prompton appended to observations (units: MGD).")
    except Exception as e:
        print(f"NWIS Prompton append skipped: {e}")

# -------------------------------
# Build time series for plotting
# -------------------------------
# Releases (downstream gage)
df_sim_gage = data.reservoir_downstream_gage[base][0]
df_obs_gage = data.reservoir_downstream_gage["obs"][0]

sim_release = df_sim_gage[RES].astype(float)
obs_release = df_obs_gage[RES].astype(float)
idx_rel = sim_release.index.intersection(obs_release.index)
sim_release = sim_release.loc[idx_rel]
obs_release = obs_release.loc[idx_rel]

# Storage
df_sim_storage = data.res_storage[base][0]
df_obs_storage = data.res_storage["obs"][0]

sim_storage = df_sim_storage[RES].astype(float)
obs_storage = df_obs_storage[RES].astype(float)
idx_sto = sim_storage.index.intersection(obs_storage.index)
sim_storage = sim_storage.loc[idx_sto]
obs_storage = obs_storage.loc[idx_sto]

# ---- pick your window ----
START = "2014-01-01"
END   = "2019-12-31"

# window each series (works even if indices are DatetimeIndex with tz=None)
sim_storage_w = sim_storage.loc[START:END]
obs_storage_w = obs_storage.loc[START:END]
sim_release_w = sim_release.loc[START:END]
obs_release_w = obs_release.loc[START:END]

# (optional) quick sanity
if sim_storage_w.empty or sim_release_w.empty:
    print("Warning: selected window has no data. Check START/END or data coverage.")

# ===============================
# Plot: 2x1 (storage, release)
# ===============================
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

axes[0].plot(obs_storage_w.index, obs_storage_w.values, linestyle="--", linewidth=2, label="Observed storage")
axes[0].plot(sim_storage_w.index, sim_storage_w.values, linewidth=1.6, alpha=0.9, label="Simulated storage")
axes[0].set_title(f"{RES} — Storage ({START} to {END})")
axes[0].set_ylabel("Storage (MG)")
axes[0].grid(True, alpha=0.4)
axes[0].legend()

axes[1].plot(obs_release_w.index, obs_release_w.values, linestyle="--", linewidth=2, label="Observed release (gage)")
axes[1].plot(sim_release_w.index, sim_release_w.values, linewidth=1.6, alpha=0.9, label=f"Simulated release ({POLICY_TYPE})")
axes[1].set_title(f"{RES} — Release ({START} to {END})")
axes[1].set_ylabel("Release / Flow (MGD)")
axes[1].set_xlabel("Date")
axes[1].grid(True, alpha=0.4)
axes[1].legend()

plt.tight_layout()
plot_filename = os.path.join(cwd, f"{RES}_storage_release_{POLICY_TYPE}_{START}_{END}.png")
plt.savefig(plot_filename, dpi=300); plt.show()
print(f"Plot saved: {plot_filename}")

# =========================
# STARFIT (17 params)
# Order:
# [0]  NORhi_mu
# [1]  NORhi_min
# [2]  NORhi_max
# [3]  NORhi_alpha
# [4]  NORhi_beta
# [5]  NORlo_mu
# [6]  NORlo_min
# [7]  NORlo_max
# [8]  NORlo_alpha
# [9]  NORlo_beta
# [10] Release_alpha1
# [11] Release_alpha2
# [12] Release_beta1
# [13] Release_beta2
# [14] Release_c
# [15] Release_p1
# [16] Release_p2
# Example string:
# STARFIT_PARAMS = "15.08,5.0,20.0,0.0,-15.0,9.0,1.6,14.2,-1.0,-30.0,0.2118,-0.0357,0.1302,-0.0248,-0.123,0.183,0.732"

# =========================
# RBF (n_rbfs * (2*n_inputs + 1) params)  with n_inputs = 3 → vars = [storage, inflow, doy]
# Order pattern:
#   Weights:          [w1, ..., wN]
#   Centers (by RBF): [c1_storage, c1_inflow, c1_doy, ..., cN_storage, cN_inflow, cN_doy]
#   Scales  (by RBF): [r1_storage, r1_inflow, r1_doy, ..., rN_storage, rN_inflow, rN_doy]
#
# Example with n_rbfs = 2 (total = 2*(2*3+1)=14):
# [0]  w1
# [1]  w2
# [2]  c1_storage
# [3]  c1_inflow
# [4]  c1_doy
# [5]  c2_storage
# [6]  c2_inflow
# [7]  c2_doy
# [8]  r1_storage
# [9]  r1_inflow
# [10] r1_doy
# [11] r2_storage
# [12] r2_inflow
# [13] r2_doy
# Example string:
# RBF_PARAMS = "0.4,0.6, 0.3,0.5,0.5, 0.7,0.2,0.6,  0.1,0.2,0.3, 0.15,0.25,0.35"

# =========================
# PWL (Piecewise Linear) — M=3 segments, n_inputs=3 → (2*M-1)*3 = 15 params
# Vector is 3 blocks in this order: storage | inflow | day
# Within each block: [x1, x2, theta1, theta2, theta3]
# Indices:
# [0]  storage_x1
# [1]  storage_x2
# [2]  storage_theta1
# [3]  storage_theta2
# [4]  storage_theta3
# [5]  inflow_x1
# [6]  inflow_x2
# [7]  inflow_theta1
# [8]  inflow_theta2
# [9]  inflow_theta3
# [10] day_x1
# [11] day_x2
# [12] day_theta1
# [13] day_theta2
# [14] day_theta3
# Example string:
# PWL_PARAMS = "0.1484,0.7055,0.3625,1.4889,-0.1242, 0.0809,0.7621,1.3527,-0.4634,0.1553, 0.0995,0.6333,0.8016,-0.6864,1.3368"

# RES = "beltzvilleCombined"
# POLICY_CLASS = "ParametricReservoirRelease"   # wrapper that accepts inline params
# POLICY_TYPE  = "RBF"                          # inner policy: STARFIT | RBF | PWL

# # Your 14-length RBF vector (must match pywrdrb's n_rbfs / n_rbf_inputs config)
# RBF_PARAMS = "0.8326,0.3053,0.8705,0.1702,0.39,0.6752,0.6042,0.7022,0.0548,0.7963,0.9912,0.1115,0.4742,0.1343"

# # === SWITCH: STARFIT on fewalter ============================================
# RES = "fewalter"
# POLICY_CLASS = "ParametricReservoirRelease"   # wrapper that accepts inline params
# POLICY_TYPE  = "STARFIT"                      # inner policy: STARFIT | RBF | PWL

# STARFIT_PARAMS = (
#     "69.392,33.682,57.512,2.0718,-4.5984,"
#     "12.529,0.0,3.3981,-5.8849,-41.113,"
#     "-0.6544,2.4273,15.897,-0.47023,-1.3845,92.989,0.014106"
# )

# # === SWITCH: PWL on fewalter =================================================
# RES = "fewalter"
# POLICY_CLASS = "ParametricReservoirRelease"   # wrapper that accepts inline params
# POLICY_TYPE  = "PWL"                          # inner policy: STARFIT | RBF | PWL

# PWL_PARAMS = (
#     "0.1484,0.7055,0.3625,1.4889,-0.1242,"
#     "0.0809,0.7621,1.3527,-0.4634,0.1553,"
#     "0.0995,0.6333,0.8016,-0.6864,1.3368"
# )

# RES = "fewalter"
# POLICY_CLASS = "ParametricReservoirRelease"   # use the wrapper that accepts inline params
# POLICY_TYPE  = "PWL"                          # inner policy family: STARFIT | RBF | PWL
# # Pick the DPS CSV you saved earlier (match policy + selection)
# #TS_DIR = os.path.join(os.path.dirname(__file__), "outputs", "sim_timeseries")
# #DPS_CSV = os.path.join(TS_DIR, f"ts_{RES}_PiecewiseLinear_Best_Release_NSE.csv")
# #assert os.path.exists(DPS_CSV), f"Missing DPS CSV: {DPS_CSV}"

# # Your 15-length PWL vector (M=3 segments; (2M−1)*3 = 15 params)
# PWL_PARAMS = "0.1484,0.7055,0.3625,1.4889,-0.1242,0.0809,0.7621,1.3527,-0.4634,0.1553,0.0995,0.6333,0.8016,-0.6864,1.3368"

# # Make scaling identical to the simple runner (fewalter context)
# CTX_OVERRIDES = {
#     "R_min": 32.30,        # MGD (≈ 50 cfs DRBC conservation release)
#     "R_max": 7690.0,       # MGD (OBS-based max you’ve been using)
#     "S_cap": 35800.0,      # MG
#     "I_min": 0.0,          # MGD
#     "I_max": 19099.99,     # MGD (your CTX print)
# }