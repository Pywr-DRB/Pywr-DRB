"""
This file contains various lists relavent to the Pywr-DRB model.
"""


reservoir_list = ['cannonsville', 'pepacton', 'neversink', 'wallenpaupack', 'prompton', 'shoholaMarsh', \
                   'mongaupeCombined', 'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon', \
                   'assunpink', 'ontelaunee', 'stillCreek', 'blueMarsh', 'greenLane', 'marshCreek']

majorflow_list = ['delLordville', 'delMontague', 'delTrenton', 'outletAssunpink', 'outletSchuylkill', 'outletChristina',
                  '01425000', '01417000', '01436000', '01433500', '01449800',
                  '01447800', '01463620', '01470960']


# The USGS gage data available downstream of reservoirs
reservoir_link_pairs = {'cannonsville': '01425000',
                           'pepacton': '01417000',
                           'neversink': '01436000',
                           'mongaupeCombined': '01433500',
                           'beltzvilleCombined': '01449800',
                           'fewalter': '01447800',
                           'assunpink': '01463620',
                           'blueMarsh': '01470960'}
