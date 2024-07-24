"""
This file contains various lists relavent to the Pywr-DRB model.
"""


reservoir_list = ['cannonsville', 'pepacton', 'neversink', 'wallenpaupack', 'prompton', 'shoholaMarsh', \
                   'mongaupeCombined', 'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon', \
                   'assunpink', 'ontelaunee', 'stillCreek', 'blueMarsh', 'greenLane']

reservoir_list_nyc = reservoir_list[:3]

majorflow_list = ['delLordville', 'delMontague', 'delDRCanal', 'delTrenton', 'outletAssunpink', 'outletSchuylkill',
                  '01425000', '01417000', '01436000', '01433500', '01449800', '01447800', '01463620', '01470960']

majorflow_list_figs = ['delMontague', 'delTrenton', 'outletSchuylkill']

# The USGS gage data available downstream of reservoirs
reservoir_link_pairs = {'cannonsville': '01425000',
                           'pepacton': '01417000',
                           'neversink': '01436000',
                           'mongaupeCombined': '01433500',
                           'beltzvilleCombined': '01449800',
                           'fewalter': '01447800',
                           'assunpink': '01463620',
                           'blueMarsh': '01470960'}

starfit_reservoir_list = ['wallenpaupack', 'prompton', 'shoholaMarsh', 
                          'mongaupeCombined', 'beltzvilleCombined', 'fewalter', 
                          'merrillCreek', 'hopatcong', 'nockamixon', 
                          'assunpink', 'ontelaunee', 'stillCreek', 'blueMarsh', 'greenLane']


modified_starfit_reservoir_list = ['blueMarsh', 'beltzvilleCombined', 'fewalter']

seasons_dict = {m: 'DJF' if m in (12,1,2) else 'MAM' if m in (3,4,5) else 'JJA' if m in (6,7,8) else 'SON' for m in range(1,13)}


drbc_lower_basin_reservoirs = ['beltzvilleCombined', 'blueMarsh', 
                               'nockamixon']  # 'wallenpaupack' at comission request; not implemented
