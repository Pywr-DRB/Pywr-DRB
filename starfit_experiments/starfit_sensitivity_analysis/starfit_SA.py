import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.plotting.bar import plot as barplot
from reservoir_models_for_SA import sim_reservoir_S
import seaborn as sns

################################################################################
# Find parameter bounds
################################################################################

starfit_conus = pd.read_csv('ISTARF-CONUS.csv')
param_bounds = pd.DataFrame()
param_bounds_mat = []

consider_params = ['NORhi_alpha', 'NORhi_beta', 'NORhi_max', 'NORhi_min',
    'NORhi_mu', 'NORlo_alpha', 'NORlo_beta', 'NORlo_max', 'NORlo_min',
    'NORlo_mu', 'Release_alpha1', 'Release_alpha2', 'Release_beta1',
    'Release_beta2', 'Release_max', 'Release_min', 'Release_c', 'Release_p1',
    'Release_p2']

p = np.linspace(1,100,100)
hi = 90
lo = 10
fig = plt.figure(figsize = (10,30), dpi = 300)
plot_location = 1

for param in consider_params:

    param_data= starfit_conus[param].replace(-np.inf, 0)
    param_data= param_data.replace(np.inf, 100)

    percentiles = np.nanpercentile(param_data, p)
    param_bounds[param] = percentiles
    param_bounds_mat.append([percentiles[lo], percentiles[hi]])

    ax = fig.add_subplot(7,3,plot_location)
    ax.plot(p, percentiles)
    ymin, ymax = ax.get_ylim()
    ax.vlines([lo,hi], ymin = ymin, ymax = ymax, colors = ['black', 'black'], ls='--', lw=2, alpha=0.5)
    ax.set_title(param)

    plot_location += 1

plt.suptitle('STARFIT Parameter Value Percentiles\nFull ISTARF-CONUS dataset considered', fontsize=16, y = 0.91)
#plt.savefig('starfit_parameter_ranges.png')
plt.show()

################################################################################
# SALib
################################################################################

problem = {
    'num_vars': len(consider_params),
    'names' : consider_params,
    'bounds': param_bounds_mat
}
"""
# Generate samples
param_samples = saltelli.sample(problem, 1024)

# Run the model for all samples
Y = np.zeros([param_samples.shape[0]])
for i, X in enumerate(param_samples):
    Y[i] = sim_reservoir_S(X)
    if i%100 == 0:
        print(i)
"""

Si = sobol.analyze(problem, Y)
total_Si, first_Si, second_Si = Si.to_df()

barplot(total_Si)


bar_labs = first_Si.index.to_list()
plt.bar(x = range(len(bar_labs)) , height = first_Si['S1'])
plt.xticks(range(len(bar_labs)), bar_labs, rotation = 90)
plt.show()

bar_labs = total_Si.index.to_list()
plt.bar(x = range(len(bar_labs)) , height = total_Si['ST'])
plt.xticks(range(len(bar_labs)), bar_labs, rotation = 90)
plt.show()


sns.heatmap(second_Si['S2'])

S2bar_labs = second_Si.index.to_list()
plt.bar(x = range(len(S2bar_labs)) , height = second_Si['S2'])
plt.xticks(range(len(S2bar_labs)), S2bar_labs, rotation = 90)
plt.show()