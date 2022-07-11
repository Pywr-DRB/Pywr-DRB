import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
    data = starfit_conus[param].replace(-np.inf, 0)
    data = data.replace(np.inf, 100)
    
    percentiles = np.nanpercentile(data, p)
    print(param)
    param_bounds[param] = percentiles
    
    param_bounds_mat.append([percentiles[hi], percentiles[lo]])
    
    ax = fig.add_subplot(7,3,plot_location)
    ax.plot(p, percentiles)
    ymin, ymax = ax.get_ylim()  
    ax.vlines([lo,hi], ymin = ymin, ymax = ymax, colors = ['black', 'black'], ls='--', lw=2, alpha=0.5)
    ax.set_title(param)
    
    plot_location += 1

plt.suptitle('STARFIT Parameter Value Percentiles\nFull ISTARF-CONUS dataset considered', fontsize=16, y = 0.91)
plt.savefig('starfit_parameter_ranges.png')
plt.show()
