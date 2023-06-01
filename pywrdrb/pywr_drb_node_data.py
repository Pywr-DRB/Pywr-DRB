"""
This script stores dictionaries which specify the connections between 
Pywr-DRB nodes and data sources for different inflow datasets.

There are unique node source matches depending on whether the dataset is:
- Observed data only
- Reconstructed historic data with PUB
- NHMv10
- NWMv2.1
- WEAP
"""

## Set of all upstream nodes for every downstream node
upstream_nodes_dict = {'01425000': ['cannonsville'],
                       '01417000': ['pepacton'],
                       'delLordville': ['cannonsville', 'pepacton', '01425000', '01417000'],
                       '01436000': ['neversink'],
                       '01433500': ['mongaupeCombined'],
                       'delMontague': ['cannonsville', 'pepacton', '01425000', '01417000', 'delLordville',
                                       'prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined',
                                       'neversink', '01436000', '01433500'],
                       '01449800': ['beltzvilleCombined'],
                       '01447800': ['fewalter'],
                       'delDRCanal': ['cannonsville', 'pepacton', '01425000', '01417000', 'delLordville',
                                      'prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined', 'neversink', '01436000', '01433500', 'delMontague',
                                      'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon', '01449800', '01447800'],
                       'delTrenton': ['cannonsville', 'pepacton', '01425000', '01417000', 'delLordville',
                                      'prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined', 'neversink', '01436000', '01433500', 'delMontague',
                                      'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon', '01449800', '01447800',
                                      'delDRCanal'],
                       '01463620': ['assunpink'],
                       'outletAssunpink': ['assunpink', '01463620'],
                       '01470960': ['blueMarsh'],
                       'outletSchuylkill': ['ontelaunee', 'stillCreek', 'blueMarsh', 'greenLane', '01470960'],
                       'outletChristina': ['marshCreek']
                       }

obs_pub_site_matches = {'cannonsville': ['01423000', '0142400103'], # 0142400103 doesnt start until '96 ##TODO: PUB until 96
                    'pepacton': ['01415000', '01414500', '01414000', '01413500'], # '01414000' doesnt start until 1996; likely underestimate before
                    'neversink': ['01435000'],
                    'wallenpaupack': None, 
                    'prompton': ['01428750'],       ## PUB till 1986-09-30 then '01428750' observation start
                    'shoholaMarsh': None, 
                    'mongaupeCombined': None, 
                    'beltzvilleCombined': ['01449360'],  ## Not complete inflow                              
                    'fewalter': ['01447720', '01447500'],
                    'merrillCreek': None, 
                    'hopatcong': None,   
                    'nockamixon': None, 
                    'assunpink': None,  
                    'ontelaunee': None, 
                    'stillCreek': None, 
                    'blueMarsh': None,  
                    'greenLane': ['01472199', '01472198'],   # PUB until 1981
                    'marshCreek': None,  
                    '01425000': None,
                    '01417000': None,
                    'delLordville': ['01427207'], # PUB until 2006 then '01427207'
                    '01436000': None,
                    '01433500': None, 
                    'delMontague': ['01438500'],
                    '01449800': None,
                    '01447800': None,
                    'delDRCanal': ['01463500'], ### note DRCanal and Trenton are treated as being coincident, with DRCanal having the physical catchment inflows and withdrawals. DRCanal is where NJ deliveries leave from, and delTrenton is where min flow is enforced, so that this is downstream of deliveries.
                    'delTrenton': ['01463500'],
                    '01463620': None, 
                    'outletAssunpink': ['01464000'],
                    '01470960': None, 
                    'outletSchuylkill': ['01474500'],
                    'outletChristina': ['01481500', '01478650'],
                    }

obs_site_matches = {'cannonsville': ['01423000'], 
                    'pepacton': ['01415000', '01414500', '01414000', '01413500'],  # '01414000' doesnt start until 1996; likely underestimate before
                    'neversink': ['01435000'],
                    'wallenpaupack': [], ## PUB 
                    'prompton': [],  ## PUB :None till 1986-09-30 then '01428750' observation start
                    'shoholaMarsh': [], ## PUB
                    'mongaupeCombined': [], ## PUB
                    'beltzvilleCombined': ['01449360'],  ## Not complete inflow                              
                    'fewalter': ['01447720', '01447500'],
                    'merrillCreek': [], ## PUB
                    'hopatcong': [],  ## PUB 
                    'nockamixon': [], ## PUB
                    'assunpink': [],  ## PUB
                    'ontelaunee': [], ## PUB
                    'stillCreek': [],  ## PUB
                    'blueMarsh': [],  ## PUB
                    'greenLane': ['01472199', '01472198'],  
                    'marshCreek':[], ## PUB
                    '01425000': ['01425000'],
                    '01417000': ['01417000'],
                    'delLordville': ['01427207'],
                    '01436000': ['01436000'],
                    '01433500': ['01433500'],
                    'delMontague': ['01438500'],
                    '01449800': ['01449800'],
                    '01447800': ['01447800'],
                    'delDRCanal': ['01463500'],
                    'delTrenton': ['01463500'],
                    '01463620': ['01463620'],        # This has periods of missing data
                    'outletAssunpink': ['01464000'],
                    '01470960': ['01470960'],
                    'outletSchuylkill': ['01474500'],
                    'outletChristina': ['01480685'] # ['01481500', '01478650'],
                    }


## NHM segment IDs
nhm_site_matches = {'cannonsville': ['1562'],
                    'pepacton': ['1449'],
                    'neversink': ['1645'],
                    'wallenpaupack': ['1602'], 
                    'prompton': ['1586'],
                    'shoholaMarsh': ['1598'], ## Note, Shohola has no inlet gauge or NHM segment - use wallenpaupack upstream
                    'mongaupeCombined': ['1640'], 
                    'beltzvilleCombined': ['1710'], 
                    'fewalter': ['1694'],
                    'merrillCreek': ['1467'], ## Merrill Creek doesnt have gage or HRU - using nearby small stream flow
                    'hopatcong': ['1470'],  ## NOTE, this is downstream but should be good
                    'nockamixon': ['1470'],
                    'assunpink': ['1496'],  ## NOTE, this is downstream of reservoir but above the link, should be good
                    'ontelaunee': ['2279'], 
                    'stillCreek': ['2277'],  ## Note, this is downstream of reservoir and lakes
                    'blueMarsh': ['2335'],
                    'greenLane': ['2310'],
                    'marshCreek': ['2009'],
                    '01425000': ['1566'],
                    '01417000': ['1444'],
                    'delLordville': ['1573'],
                    '01436000': ['1638'],
                    '01433500': ['1647'],
                    'delMontague': ['1659'],
                    '01449800': ['1697'],
                    '01447800': ['1695'],
                    'delDRCanal': ['1498'],
                    'delTrenton': ['1498'],
                    '01463620': ['1492'],
                    'outletAssunpink': ['1493'],
                    '01470960': ['2333'],
                    'outletSchuylkill': ['2338'],
                    'outletChristina': ['2005'] # This can be improved but low priority
                    }

## NWM data IDs are either COMIDs or USGS-IDs (USGS-IDs start with 0)
nwm_site_matches = {'cannonsville': ['2613174'],    # Lake inflow
                    'pepacton': ['1748473'],        # Lake inflow
                    'neversink': ['4146742'],       # Lake inflow
                    'wallenpaupack': ['2741600'],   # Lake inflow
                    'prompton': ['2739068'],        # Lake inflow
                    'shoholaMarsh': ['120052035'],  # Lake inflow
                    'mongaupeCombined': ['4148582'],    # Lake inflow
                    'beltzvilleCombined': ['4186689'],  # Lake inflow
                    'fewalter': ['4185065'],        # Lake inflow
                    'merrillCreek': ['2588031'],    # No NWM lake; using available segment flow
                    'hopatcong': ['2585287'],       # Lake inflow
                    'nockamixon': ['2591099'],      # No NWM lake; using available segment flow
                    'assunpink': ['2589015'],       # Lake inflow
                    'ontelaunee': ['4779981'],      # Lake inflow
                    'stillCreek': ['4778721'],      # Lake inflow
                    'blueMarsh': ['4782813'],       # Lake inflow
                    'greenLane': ['4780087'],       # Lake inflow 
                    'marshCreek': ['4648728'],      # No NWM lake; using available segment flow
                    '01425000': ['01425000'],
                    '01417000': ['01417000'],
                    'delLordville': ['2617364'],
                    '01436000': ['01436000'],
                    '01433500': ['01433500'],
                    'delMontague': ['4151628'],
                    '01449800': ['01449800'],
                    '01447800': ['01447800'],
                    'delDRCanal': ['2590277'],
                    'delTrenton': ['2590277'],
                    '01463620': ['01463620'],
                    'outletAssunpink': ['2590137'],
                    '01470960': ['01470960'],
                    'outletSchuylkill': ['4784841'],
                    'outletChristina': ['4652144'],
                    }


### NWM inflows corresponding to lake objects
nwm_lake_site_matches = {'cannonsville': '2613174',
                            '01425000': 'none',
                            'pepacton': '1748473',
                            '01417000': 'none',
                            'delLordville': 'none',
                            'neversink': '4146742',
                            '01436000': 'none',
                            'wallenpaupack': '2741600',
                            'prompton': '2739068',
                            'shoholaMarsh': '120052035',
                            'mongaupeCombined': '4148582',
                            '01433500': 'none',
                            'delMontague': 'none',
                            'beltzvilleCombined': '4186689',
                            '01449800': 'none',
                            'fewalter': '4185065',
                            '01447800': 'none',
                            'merrillCreek': 'none',
                            'hopatcong': '2585287',
                            'nockamixon': 'none',
                            'delDRCanal': 'none',
                            'delTrenton': 'none',
                            'assunpink': '2589015',
                            '01463620': 'none',
                            'outletAssunpink': 'none',
                            'ontelaunee': '4779981',
                            'stillCreek': '4778721',
                            'blueMarsh': '4782813',
                            '01470960': 'none',
                            'greenLane': '4780087',
                            'outletSchuylkill': 'none',
                            'marshCreek': 'none',
                            'outletChristina': 'none'
                            }