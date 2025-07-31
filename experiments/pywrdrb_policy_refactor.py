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

metadata_file = os.path.join(results_dir, "simulation_metadata.csv")
if not os.path.exists(metadata_file):
    with open(metadata_file, "w") as f:
        f.write("reservoir,policy_type,metric_key,metric_label,runtime_sec,output_size_MB\n")

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

            # === Load and run model === #
            model = pywrdrb.Model.load(model_filename)
            output_filename = os.path.join(results_dir, f"{run_id}_output.hdf5")
            recorder = pywrdrb.OutputRecorder(model, output_filename)

            t_start = time.perf_counter()
            stats = model.run()
            t_end = time.perf_counter()
            runtime = t_end - t_start

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
