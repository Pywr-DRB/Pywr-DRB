import numpy as np
import pandas as pd
from math import pi, sin, cos

starfit = pd.read_csv('model_data/drb_model_istarf_conus.csv')
sw = {k: starfit[k].iloc[6] for k in starfit.columns}

def update(storage, inflow, week):
    shat = storage / sw['GRanD_CAP_MG']
    ihat = (inflow - sw['GRanD_MEANFLOW_MGD']) / sw['GRanD_MEANFLOW_MGD']

    t = week/52
    sint = sin(2 * pi * t)
    cost = cos(2 * pi * t)
    sin2t = sin(4 * pi * t)
    cos2t = cos(4 * pi * t)

    NORhi = sw['NORhi_mu'] + sw['NORhi_alpha'] * sint + sw['NORhi_beta'] * cost
    NORhi = min(max(NORhi, sw['NORhi_min']), sw['NORhi_max']) / 100

    NORlo = sw['NORlo_mu'] + sw['NORlo_alpha'] * sint + sw['NORlo_beta'] * cost
    NORlo = min(max(NORlo, sw['NORlo_min']), sw['NORlo_max']) / 100

    Rt = sw['Release_alpha1'] * sint + sw['Release_alpha2'] * sin2t + sw['Release_beta1'] * cost + sw['Release_beta2'] * cos2t
    At = (shat - NORlo) / (NORhi - NORlo)
    eps = sw['Release_c'] + sw['Release_p1'] * At + sw['Release_p2'] * ihat
    inNOR_target = sw['GRanD_MEANFLOW_MGD'] * (Rt + eps + 1)  
    print(Rt, At, ihat, eps)

    release_max = (sw['Release_max'] + 1) * sw['GRanD_MEANFLOW_MGD'] 
    release_min = (sw['Release_min'] + 1) * sw['GRanD_MEANFLOW_MGD']
    aboveNOR_target = (sw['GRanD_CAP_MG'] * (shat - NORhi) + inflow * 7) / 7
    belowNOR_target = release_min 

    if shat < NORlo:
        target = belowNOR_target
    elif shat <= NORhi:
        target = inNOR_target
    else:
        target = aboveNOR_target

    release = max(min(target, release_max), release_min) 

    return NORhi, NORlo, belowNOR_target, inNOR_target, aboveNOR_target, target, release

print(update(36326.89897021, 73.79377884, 5))
