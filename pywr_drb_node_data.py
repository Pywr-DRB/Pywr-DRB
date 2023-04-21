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

## A dictionary describing Pywr-DRB nodes and corresponding upstream nodes
site_matches_link = {'01425000': ['cannonsville'],
                     '01417000': ['pepacton'],
                     'delLordville': ['cannonsville', 'pepacton', '01425000', '01417000'],
                     '01436000': ['neversink'],
                     '01433500': ['mongaupeCombined'],
                     'delMontague': ['cannonsville', 'pepacton', '01425000', '01417000', 'delLordville',
                                     'prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined', 
                                     'neversink', '01436000', '01433500'],
                     '01449800': ['beltzvilleCombined'],
                     '01447800': ['fewalter'],
                     'delTrenton': ['cannonsville', 'pepacton', '01425000', '01417000', 'delLordville',
                                    'prompton', 'wallenpaupack', 'shoholaMarsh', 'mongaupeCombined', 'neversink', '01436000', '01433500', 'delMontague',
                                    'beltzvilleCombined', 'fewalter', 'merrillCreek', 'hopatcong', 'nockamixon', '01449800', '01447800'],
                     '01463620': ['assunpink'],
                     'outletAssunpink': ['assunpink', '01463620'],
                     '01470960': ['blueMarsh'],
                     'outletSchuylkill': ['ontelaunee', 'stillCreek', 'blueMarsh', 'greenLane', '01470960'],
                     'outletChristina': ['marshCreek']
                     }


obs_pub_site_matches = {'cannonsville': None,  # ['01423000'] is on the mainstem but biased low
                    'pepacton': ['01415000', '01414500', '01414000', '01413500'], # '01414000' doesnt start until 1996; likely underestimate before
                    'neversink': ['01435000'],
                    'wallenpaupack': None, 
                    'prompton': None,       ## PUB #TODO: PUB till 1986-09-30 then '01428750' observation start
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
                    'greenLane': ['01472199', '01472198'],   # These two start 1981
                    'marshCreek': None,  
                    '01425000': None,
                    '01417000': None,
                    'delLordville': None, # '01427207' starts in 2006
                    '01436000': None,
                    '01433500': None, 
                    'delMontague': ['01438500'],
                    '01449800': None,
                    '01447800': None,
                    'delTrenton': ['01463500'],
                    '01463620': None, 
                    'outletAssunpink': ['01464000'],
                    '01470960': None, 
                    'outletSchuylkill': ['01474500'],
                    'outletChristina': ['01480685'] # ['01481500', '01478650'],
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
                    'delTrenton': ['1498'],
                    '01463620': ['1492'],
                    'outletAssunpink': ['1493'],
                    '01470960': ['2333'],
                    'outletSchuylkill': ['2338'],
                    'outletChristina': ['2005'] # This can be improved but low priority
                    }

## NWM data IDs are either COMIDs or USGS-IDs (USGS-IDs start with 0)
nwm_site_matches = {'cannonsville': ['01425000'],
                    'pepacton': ['01417000'],
                    'neversink': ['01436000'],
                    'wallenpaupack': ['01429000'], ## Note, wanted to do 01431500 minus 01432110, but don't have latter from Aubrey, so use Prompton for now
                    'prompton': ['01429000'],
                    'shoholaMarsh': ['01429000'], ## Note, Shohola has no inlet gauge
                    'mongaupeCombined': ['01433500'], # NOTE, this is an outlet.  TODO See:  01432900
                    'beltzvilleCombined': ['01449800'],  ## NOTE, This is the outlet. TODO See: 01449360
                    'fewalter': ['01447800'],
                    'merrillCreek': ['01459500'], ## Merrill Creek doesnt have gage - use Nockamixon nearby to get flow shape
                    'hopatcong': ['01455500'],  ## NOTE, this is an outlet. There are inlet gauges with poor records
                    'nockamixon': ['01459500'], ## NOTE, this is far downstream of outlet.
                    'assunpink': ['01463620'],  ## Note, this is outlet of system.
                    'ontelaunee': ['01470960'], ## Note, should have 01470761, but didnt get from Aubrey, so use Blue Marsh for now
                    'stillCreek': ['01469500'],  ## Note, this is downstream of reservoir and lakes
                    'blueMarsh': ['01470960'],
                    'greenLane': ['01473000'],  ## Note, this is far downstream of outlet.
                    'marshCreek': ['01480685'], ## Note, this is an outlet. TODO See: 01480675
                    '01425000': ['01425000'],
                    '01417000': ['01417000'],
                    'delLordville': ['01427207'],
                    '01436000': ['01436000'],
                    '01433500': ['01433500'],
                    'delMontague': ['01438500'],
                    '01449800': ['01449800'],
                    '01447800': ['01447800'],
                    'delTrenton': ['01463500'],
                    '01463620': ['01463620'],
                    'outletAssunpink': ['01463620'],  ## Should be '01464000' but don't have NWM data currently for this location
                    '01470960': ['01470960'],
                    'outletSchuylkill': ['01474500'],
                    'outletChristina': ['01480685'],
                    }

