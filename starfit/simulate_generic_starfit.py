"""
Trevor Amestoy
Cornell University

"""

from STARFIT import Reservoir
import numpy as np
import pandas as pd


# Constants
Rmax = 1
Rmin = 0
mean_flow = 100     #MGD
cap = 500           #MG
timestep = 'daily'

# Periodic release
Ra1 = 0.5
Ra2 = 0.5
Rb1 = 0.5
Rb2 = 0.5
Rc = 0.5
Rp1 = 1
Rp2 = 1

# NOR
NORhi_mu = 1
NORhi_beta = 1
NORhi_min = 0
NORhi_max = 1
NORlo_mu = 1
NORlo_beta = 1
NORlo_min = 0
NORlo_max = 1

# Combine
starfit_params = {'Release_max': Rmax,
                    'Release_min': Rmin,
                    'Release_alpha1': Ra1,
                    'Release_alpha2': Ra2,
                    'Release_beta1': Rb1,
                    'Release_beta2': Rb2,
                    'Release_p1': Rp1,
                    'Release_p2': Rp2,
                    'Release_c': Rc,
                    'NORhu_mu': NORhi_mu,
                    'NORhi_max': NORhi_max,
                    'NORhi_min': NORhi_min,
                    'NORlo_mu': NORlo_mu,
                    'NORlo_beta': NORlo_beta,
                    'NORlo_max': NORlo_max,
                    'NORlo_min': NORlo_min,
                    'GRanD_MEANFLOW_MGD': mean_flow,
                    'GRanD_CAP_MG': cap}
