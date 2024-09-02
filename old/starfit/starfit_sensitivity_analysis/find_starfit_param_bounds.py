import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# Load full ISTARF-CONUS dataset
starfit_conus = pd.read_csv("ISTARF-CONUS.csv")

# Load STARFIT conus data for reservoirs in DRB
drb_reservoirs = ["Blue Marsh Dam", "Beltzville Dam"]


# Initialize
param_bounds = pd.DataFrame()
param_bounds_mat = []

# List of uncertain parameter keys
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

# Initialize percentile constants
p = np.linspace(1, 100, 100)
hi = 90
lo = 10

fig = plt.figure(figsize=(10, 30), dpi=300)
plot_location = 1

for param in consider_params:
    # Pull specific parameter data
    data = starfit_conus[param].replace(-np.inf, 0)
    data = data.replace(np.inf, 100)

    percentiles = np.nanpercentile(data, p)
    param_bounds[param] = percentiles

    param_bounds_mat.append([percentiles[hi], percentiles[lo]])

    # Plot param specific subplot
    ax = fig.add_subplot(7, 3, plot_location)
    ax.plot(p, percentiles, label="Dist. of all STARFIT-CONUS values")
    ymin, ymax = ax.get_ylim()
    ax.vlines(
        [lo, hi],
        ymin=ymin,
        ymax=ymax,
        colors=["black", "black"],
        ls="--",
        lw=2,
        alpha=0.5,
        label="Selected value limits",
    )

    # Find the location of the DRB reservoir param value along curve
    for r, c, l in zip(drb_reservoirs, ["red", "green"], drb_reservoirs):
        # Rank all param values
        param_ranks = rankdata(data)
        res_index = starfit_conus.index[starfit_conus["GRanD_NAME"] == r].tolist()
        res_rank = param_ranks[res_index]

        res_perc = 100 * res_rank / (len(param_ranks) + 1)

        ax.scatter(res_perc, data[res_index].values, color=c, label=l)

    ax.set_title(param)

    plot_location += 1

plt.legend(bbox_to_anchor=(1.05, 1.0), title="Legend:")
plt.suptitle(
    "STARFIT Parameter Value Percentiles\nFull ISTARF-CONUS dataset considered",
    fontsize=16,
    y=0.91,
)
plt.savefig("starfit_parameter_ranges.png")
plt.show()
