import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.plotting.bar import plot as barplot
from reservoir_models_for_SA import sim_reservoir_S

################################################################################
# Find parameter bounds: See find_starfit_param_bounds.py for
#                       visual value distribution
################################################################################

starfit_conus = pd.read_csv("ISTARF-CONUS.csv")
param_bounds = pd.DataFrame()
param_bounds_mat = []

consider_params = [
    "NORhi_alpha",
    "NORhi_beta",
    "NORhi_max",
    "NORhi_min",
    "NORhi_mu",
    "NORlo_alpha",
    "NORlo_beta",
    "NORlo_max",
    "NORlo_min",
    "NORlo_mu",
    "Release_alpha1",
    "Release_alpha2",
    "Release_beta1",
    "Release_beta2",
    "Release_max",
    "Release_min",
    "Release_c",
    "Release_p1",
    "Release_p2",
]

p = np.linspace(1, 100, 100)
hi = 95
lo = 5

for param in consider_params:
    param_data = starfit_conus[param].replace(-np.inf, 0)
    param_data = param_data.replace(np.inf, 100)

    percentiles = np.nanpercentile(param_data, p)
    param_bounds[param] = percentiles
    param_bounds_mat.append([percentiles[lo], percentiles[hi]])


################################################################################
# SALib
################################################################################

# Test a single reservoir
test_reservoir = "beltzville"

# Select the bounds according to above percentile ranges
problem = {
    "num_vars": len(consider_params),
    "names": consider_params,
    "bounds": param_bounds_mat,
}

# Generate samples
param_samples = saltelli.sample(problem, 1024)

# Run the model for all samples
Y = np.zeros([param_samples.shape[0]])
for i, X in enumerate(param_samples):
    Y[i] = sim_reservoir_S(X, reservoir_name=test_reservoir)
    if i % 100 == 0:
        print(i)


# Calculate metrics
Si = sobol.analyze(problem, Y)
total_Si, first_Si, second_Si = Si.to_df()
higher_S = total_Si - first_Si

# Export
Si.to_csv("./S_index.csv", sep=",")
sensitivity = Si.to_df()
sensitivity.to_csv("./senesitivities.csv", sep=",")

total_Si, first_Si, second_Si = Si.to_df()

total_Si.columns = ["S", "S_conf"]
first_Si.columns = ["S", "S_conf"]
higher_S = total_Si.subtract(first_Si)

bar_labs = first_Si.index.to_list()
plt.bar(x=range(len(bar_labs)), height=first_Si["S1"])
plt.xticks(range(len(bar_labs)), bar_labs, rotation=90)
plt.title("First Order Sensitivity Index")
plt.savefig(str("./figures/S1_" + test_reservoir + ".png"))
plt.show()

S2bar_labs = second_Si.index.to_list()
plt.bar(x=range(len(S2bar_labs)), height=second_Si["S2"])
plt.xticks(range(len(S2bar_labs)), S2bar_labs, rotation=90)
plt.title("Second Order Sensitivity Index")
plt.savefig(str("./figures/S2_" + test_reservoir + ".png"))
plt.show()

fig, ax = plt.subplots()
bar_labs = total_Si.index.to_list()
ax.bar(bar_labs, first_Si["S"], label="First Order", color="black")
ax.bar(
    bar_labs, higher_S["S"], bottom=first_Si["S"], label="Higher Order", color="grey"
)
ax.set_xticks(range(len(bar_labs)), bar_labs, rotation=90)
ax.set_xlabel("Parameter")
plt.title("Sensitivity Index")
plt.legend()
plt.savefig(str("./figures/S_total_" + test_reservoir + ".png"))
plt.show()
