from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom

### set up icons
reservoir_icon = 'icons/reservoir.png'
gage_icon = 'icons/stream.png'
diversion_icon = 'icons/demand.png'

# ### customize graphviz attributes
# graph_attr = {
#     'fontsize': '30',
# }
# node_attr = {
#     'fontsize': '30',
# }

# ### create a diagram to depict the node network
# with Diagram("Pywr-DRB Node Network Style 1", show=False, graph_attr=graph_attr, node_attr=node_attr, direction='TB'):
#     ### set up standalond nodes
#     nyc_diversion = Custom('NYC Diversion', diversion_icon)
#     nj_diversion = Custom('NJ Diversion', diversion_icon)
#     montague = Custom('Montague', gage_icon)
#     trenton1 = Custom('Trenton 1', gage_icon)
#     trenton2 = Custom('Trenton 2', gage_icon)
#     outlet = Custom('Delaware Bay', gage_icon)
#
#     ### set up clusters of nodes
#     with Cluster('NYC Reservoirs', graph_attr=graph_attr):
#         nyc_reservoirs = [Custom('Cannonsville', reservoir_icon),
#                           Custom('Pepacton', reservoir_icon),
#                           Custom('Neversink', reservoir_icon)]
#     with Cluster('Non-NYC Inputs to Montague', graph_attr=graph_attr):
#         inputs_montague = [Custom('Prompton', reservoir_icon),
#                            Custom('Wallenpaupack', reservoir_icon),
#                            Custom('Mongaup', reservoir_icon)]
#     with Cluster('Downstream of Montague Inputs to Trenton', graph_attr=graph_attr):
#         inputs_trenton = [Custom('Beltzville', reservoir_icon),
#                            Custom('F.E. Walter', reservoir_icon),
#                            Custom('Nockamixon', reservoir_icon),
#                            Custom('Hopatcong', reservoir_icon)]
#     with Cluster('Downstream of Trenton', graph_attr=graph_attr):
#         inputs_outlet = [Custom('Blue Marsh', reservoir_icon),
#                            Custom('Green Lane', reservoir_icon),
#                            Custom('Still Creek', reservoir_icon),
#                            Custom('Assunpink', reservoir_icon)]
#
#     ### tie them all together
#     nyc_reservoirs >> nyc_diversion
#     nyc_reservoirs >> montague >> trenton1 >> trenton2 >> outlet
#     inputs_montague >> montague
#     inputs_trenton >> trenton1
#     trenton1 >> nj_diversion
#     inputs_outlet >> outlet
#
#
#
#
#
#
# ### create a diagram to depict the node network
# with Diagram("Pywr-DRB Node Network Style 2", show=False, graph_attr=graph_attr, node_attr=node_attr, direction='TB'):
#     ### diversion nodes
#     NYCDiversion = Custom('NYC Diversion', diversion_icon)
#     NJDiversion = Custom('NJ Diversion', diversion_icon)
#
#     ### river nodes
#
#     Lordville = Custom('Lordville', gage_icon)
#     Montague = Custom('Montague', gage_icon)
#     Trenton1 = Custom('Trenton 1', gage_icon)
#     Trenton2 = Custom('Trenton 2', gage_icon)
#     DelawareBay = Custom('Delaware Bay', gage_icon)
#     USGS01425000 = Custom('USGS01425000', gage_icon)
#     USGS01417000 = Custom('USGS01417000', gage_icon)
#     USGS01436000 = Custom('USGS01436000', gage_icon)
#     USGS01433500 = Custom('USGS01433500', gage_icon)
#     USGS01449800 = Custom('USGS01449800', gage_icon)
#     USGS01447800 = Custom('USGS01447800', gage_icon)
#     USGS01463620 = Custom('USGS01463620', gage_icon)
#     USGS01470960 = Custom('USGS01470960', gage_icon)
#
#     ### reservoir nodes
#     Cannonsville = Custom('Cannonsville', reservoir_icon)
#     Pepacton = Custom('Pepacton', reservoir_icon)
#     Neversink = Custom('Neversink', reservoir_icon)
#     Prompton = Custom('Prompton', reservoir_icon)
#     Wallenpaupack = Custom('Wallenpaupack', reservoir_icon)
#     Mongaup = Custom('Mongaup', reservoir_icon)
#     Beltzville = Custom('Beltzville', reservoir_icon)
#     FEWalter = Custom('F.E. Walter', reservoir_icon)
#     Nockamixon = Custom('Nockamixon', reservoir_icon)
#     Hopatcong = Custom('Hopatcong', reservoir_icon)
#     BlueMarsh = Custom('Blue Marsh', reservoir_icon)
#     GreenLane = Custom('Green Lane', reservoir_icon)
#     StillCreek = Custom('Still Creek', reservoir_icon)
#     Assunpink = Custom('Assunpink', reservoir_icon)
#
#
#     ### tie them all together
#     Cannonsville >> NYCDiversion
#     Pepacton >> NYCDiversion
#     Neversink >> NYCDiversion
#     Cannonsville >> Lordville >> Montague >> Trenton1 >> Trenton2 >> DelawareBay
#     Pepacton >> Lordville
#     Lordville >> Montague
#     Prompton >> Montague
#     Wallenpaupack >> Montague
#     Mongaup >> Montague
#     Beltzville >> Trenton1
#     FEWalter >> Trenton1
#     Nockamixon >> Trenton1
#     Hopatcong >> Trenton1
#     Trenton1 >> NJDiversion
#     Trenton2 >> DelawareBay
#     Assunpink >> DelawareBay
#     BlueMarsh >> DelawareBay
#     GreenLane >> DelawareBay
#     StillCreek >> DelawareBay
#






### customize graphviz attributes
graph_attr = {
    'fontsize': '40',
    'splines': 'spline',
    # 'layout': 'fdp'
}

reservoir_icon = 'icons/reservoir.png'
river_icon = 'icons/river.png'
gage_icon = 'icons/measurement.png'
diversion_icon = 'icons/demand.png'
### create a diagram to depict the node network
# with Diagram("Pywr-DRB Node Network", show=False, graph_attr=graph_attr, direction='TB'):
with Diagram("Pywr-DRB Node Network", show=False, graph_attr=graph_attr, direction='LR'):


    ### diversion nodes
    graph_attr['bgcolor'] = 'mediumseagreen'

    with Cluster('NYC Diversion', graph_attr=graph_attr):
        NYCDiversion = Custom('', diversion_icon)

    with Cluster('NJ Diversion', graph_attr=graph_attr):
        NJDiversion = Custom('', diversion_icon)



    ### function for creating edge with color based on delay (days)
    def create_edge(lag_days):
        penwidth = '4'
        if lag_days == 0:
            return Edge(color='black', style='solid', penwidth=penwidth)
        elif lag_days == 1:
            return Edge(color='black', style='dashed', penwidth=penwidth)
        elif lag_days == 2:
            return Edge(color='black', style='dotted', penwidth=penwidth)


    ### cluster of minor nodes within major node
    def create_node_cluster(label, has_reservoir, has_gage):
        if has_reservoir and label in ['Cannonsville', 'Pepacton', 'Neversink']:
            bgcolor='firebrick'
        elif has_reservoir:
            bgcolor='lightcoral'
        else:
            bgcolor='cornflowerblue'
        graph_attr['bgcolor'] = bgcolor

        with Cluster(label, graph_attr=graph_attr):
            cluster_river = Custom('', river_icon)

            if has_reservoir:
                cluster_reservoir = Custom('', reservoir_icon)
                cluster_river >> create_edge(0) >> cluster_reservoir
                if has_gage:
                    cluster_gage = Custom('', gage_icon)
                    cluster_reservoir >> create_edge(0) >> cluster_gage
                    return {'river': cluster_river, 'reservoir': cluster_reservoir, 'out': cluster_gage}
                else:
                    return {'river': cluster_river, 'reservoir': cluster_reservoir, 'out': cluster_reservoir}
            else:
                if has_gage:
                    cluster_gage = Custom('', gage_icon)
                    cluster_river >> create_edge(0) >> cluster_gage
                    return {'river': cluster_river, 'reservoir': None, 'out': cluster_gage}
                else:
                    return {'river': cluster_river, 'reservoir': None, 'out': cluster_river}



    ### river nodes
    Lordville = create_node_cluster('Lordville', has_reservoir=False, has_gage=True)
    Montague = create_node_cluster('Montague', has_reservoir=False, has_gage=True)
    Trenton1 = create_node_cluster('Trenton 1', has_reservoir=False, has_gage=False)
    Trenton2  = create_node_cluster('Trenton 2', has_reservoir=False, has_gage=True)
    DelawareBay  = create_node_cluster('Delaware Bay', has_reservoir=False, has_gage=True)


    ### reservoir nodes
    Cannonsville = create_node_cluster('Cannonsville', has_reservoir=True, has_gage=True)
    Pepacton = create_node_cluster('Pepacton', has_reservoir=True, has_gage=True)
    Neversink = create_node_cluster('Neversink', has_reservoir=True, has_gage=True)
    Prompton = create_node_cluster('Prompton', has_reservoir=True, has_gage=False)
    Wallenpaupack = create_node_cluster('Wallenpaupack', has_reservoir=True, has_gage=False)
    ShoholaMarsh = create_node_cluster('Shohola Marsh', has_reservoir=True, has_gage=True)
    Mongaup = create_node_cluster('Mongaup', has_reservoir=True, has_gage=True)
    Beltzville = create_node_cluster('Beltzville', has_reservoir=True, has_gage=True)
    FEWalter = create_node_cluster('F.E. Walter', has_reservoir=True, has_gage=True)
    MerrillCreek = create_node_cluster('Merrill Creek', has_reservoir=True, has_gage=False)
    Hopatcong = create_node_cluster('Hopatcong', has_reservoir=True, has_gage=False)
    Nockamixon = create_node_cluster('Nockamixon', has_reservoir=True, has_gage=False)
    Assunpink = create_node_cluster('Assunpink', has_reservoir=True, has_gage=True)
    StillCreek = create_node_cluster('Still Creek', has_reservoir=True, has_gage=False)
    Ontelaunee = create_node_cluster('Ontelaunee', has_reservoir=True, has_gage=False)
    BlueMarsh = create_node_cluster('Blue Marsh', has_reservoir=True, has_gage=True)
    GreenLane = create_node_cluster('Green Lane', has_reservoir=True, has_gage=False)


    ### tie them all together
    Cannonsville['reservoir'] >> create_edge(0) >> NYCDiversion
    Pepacton['reservoir'] >> create_edge(0) >> NYCDiversion
    Neversink['reservoir'] >> create_edge(0) >> NYCDiversion
    Cannonsville['out'] >> create_edge(0) >> Lordville['river']
    Pepacton['out'] >> create_edge(0) >> Lordville['river']
    Lordville['out'] >> create_edge(2) >> Montague['river']
    Neversink['out'] >> create_edge(1) >> Montague['river']
    Prompton['out'] >> create_edge(1) >> Montague['river']
    Wallenpaupack['out'] >> create_edge(1) >> Montague['river']
    ShoholaMarsh['out'] >> create_edge(1) >> Montague['river']
    Mongaup['out'] >> create_edge(0) >> Montague['river']
    Montague['out'] >> create_edge(2) >> Trenton1['river']
    Beltzville['out'] >> create_edge(2) >> Trenton1['river']
    FEWalter['out'] >> create_edge(2) >> Trenton1['river']
    MerrillCreek['out'] >> create_edge(1) >> Trenton1['river']
    Hopatcong['out'] >> create_edge(1) >> Trenton1['river']
    Nockamixon['out'] >> create_edge(0) >> Trenton1['river']
    Trenton1['out'] >> create_edge(0) >> Trenton2['river']
    Trenton1['out'] >> create_edge(0) >> NJDiversion
    Trenton2['out'] >> create_edge(0) >> DelawareBay['river']
    Assunpink['out'] >> create_edge(0) >> DelawareBay['river']
    Ontelaunee['out'] >> create_edge(2) >> DelawareBay['river']
    StillCreek['out'] >> create_edge(2) >> DelawareBay['river']
    BlueMarsh['out'] >> create_edge(2) >> DelawareBay['river']
    GreenLane['out'] >> create_edge(1) >> DelawareBay['river']





#
# ### customize graphviz attributes
# graph_attr = {
#     'fontsize': '15',
#     'splines': 'spline',
#     'layout': 'neato'
# }
#
# reservoir_icon = 'icons/reservoir.png'
# river_icon = 'icons/river.png'
# gage_icon = 'icons/measurement.png'
# diversion_icon = 'icons/demand.png'
# ### create a diagram to depict the node network
# with Diagram("Pywr-DRB Node Network", show=False, graph_attr=graph_attr):
#
#     ### diversion nodes
#     graph_attr['bgcolor'] = 'mediumseagreen'
#
#     with Cluster('NYC Diversion', graph_attr=graph_attr):
#         NYCDiversion = Custom('', diversion_icon, pin='true', pos='10,21')
#
#     with Cluster('NJ Diversion', graph_attr=graph_attr):
#         NJDiversion = Custom('', diversion_icon, pin='true', pos='10,10')
#
#
#
#     ### function for creating edge with color based on delay (days)
#     def create_edge(lag_days):
#         penwidth = '4'
#         if lag_days == 0:
#             return Edge(color='black', style='solid', penwidth=penwidth)
#         elif lag_days == 1:
#             return Edge(color='black', style='dashed', penwidth=penwidth)
#         elif lag_days == 2:
#             return Edge(color='black', style='dotted', penwidth=penwidth)
#
#
#     ### cluster of minor nodes within major node
#     def create_node_cluster(label, has_reservoir, has_gage, base_position):
#         if has_reservoir and label in ['Cannonsville', 'Pepacton', 'Neversink']:
#             bgcolor='firebrick'
#         elif has_reservoir:
#             bgcolor='lightcoral'
#         else:
#             bgcolor='cornflowerblue'
#         graph_attr['bgcolor'] = bgcolor
#
#         with Cluster(label, graph_attr=graph_attr):
#             cluster_river = Custom('', river_icon, pin='true', pos=f'{base_position[0]},{base_position[1]}')
#
#             if has_reservoir:
#                 cluster_reservoir = Custom('', reservoir_icon, pin='true', pos=f'{base_position[0]},{base_position[1]-1}')
#                 cluster_river >> create_edge(0) >> cluster_reservoir
#                 if has_gage:
#                     cluster_gage = Custom('', gage_icon, pin='true', pos=f'{base_position[0]},{base_position[1]-2}')
#                     cluster_reservoir >> create_edge(0) >> cluster_gage
#                     return {'river': cluster_river, 'reservoir': cluster_reservoir, 'out': cluster_gage}
#                 else:
#                     return {'river': cluster_river, 'reservoir': cluster_reservoir, 'out': cluster_reservoir}
#             else:
#                 if has_gage:
#                     cluster_gage = Custom('', gage_icon, pin='true', pos=f'{base_position[0]},{base_position[1]-1}')
#                     cluster_river >> create_edge(0) >> cluster_gage
#                     return {'river': cluster_river, 'reservoir': None, 'out': cluster_gage}
#                 else:
#                     return {'river': cluster_river, 'reservoir': None, 'out': cluster_river}
#
#
#
#     ### river nodes
#     Lordville = create_node_cluster('Lordville', has_reservoir=False, has_gage=True, base_position=(1,18))
#     Montague = create_node_cluster('Montague', has_reservoir=False, has_gage=True, base_position=(5,14))
#     Trenton1 = create_node_cluster('Trenton 1', has_reservoir=False, has_gage=False, base_position=(5,10))
#     Trenton2  = create_node_cluster('Trenton 2', has_reservoir=False, has_gage=True, base_position=(7,10))
#     DelawareBay  = create_node_cluster('Delaware Bay', has_reservoir=False, has_gage=True, base_position=(9,6))
#
#
#     ### reservoir nodes
#     Cannonsville = create_node_cluster('Cannonsville', has_reservoir=True, has_gage=True, base_position=(0,20))
#     Pepacton = create_node_cluster('Pepacton', has_reservoir=True, has_gage=True, base_position=(2,20))
#     Neversink = create_node_cluster('Neversink', has_reservoir=True, has_gage=True, base_position=(4,20))
#     Prompton = create_node_cluster('Prompton', has_reservoir=True, has_gage=False, base_position=(0,16))
#     Wallenpaupack = create_node_cluster('Wallenpaupack', has_reservoir=True, has_gage=False, base_position=(2,16))
#     ShoholaMarsh = create_node_cluster('Shohola Marsh', has_reservoir=True, has_gage=True, base_position=(4,16))
#     Mongaup = create_node_cluster('Mongaup', has_reservoir=True, has_gage=True, base_position=(6,16))
#     Beltzville = create_node_cluster('Beltzville', has_reservoir=True, has_gage=True, base_position=(2, 12))
#     FEWalter = create_node_cluster('F.E. Walter', has_reservoir=True, has_gage=True, base_position=(0, 12))
#     MerrillCreek = create_node_cluster('Merrill Creek', has_reservoir=True, has_gage=False, base_position=(4,12))
#     Hopatcong = create_node_cluster('Hopatcong', has_reservoir=True, has_gage=False, base_position=(6,12))
#     Nockamixon = create_node_cluster('Nockamixon', has_reservoir=True, has_gage=False, base_position=(8,12))
#     Assunpink = create_node_cluster('Assunpink', has_reservoir=True, has_gage=True, base_position=(10,8))
#     StillCreek = create_node_cluster('Still Creek', has_reservoir=True, has_gage=False, base_position=(0,8))
#     Ontelaunee = create_node_cluster('Ontelaunee', has_reservoir=True, has_gage=False, base_position=(2,8))
#     BlueMarsh = create_node_cluster('Blue Marsh', has_reservoir=True, has_gage=True, base_position=(4,8))
#     GreenLane = create_node_cluster('Green Lane', has_reservoir=True, has_gage=False, base_position=(6,8))
#
#
#     ### tie them all together
#     Cannonsville['reservoir'] >> create_edge(0) >> NYCDiversion
#     Pepacton['reservoir'] >> create_edge(0) >> NYCDiversion
#     Neversink['reservoir'] >> create_edge(0) >> NYCDiversion
#     Cannonsville['out'] >> create_edge(0) >> Lordville['river']
#     Pepacton['out'] >> create_edge(0) >> Lordville['river']
#     Lordville['out'] >> create_edge(2) >> Montague['river']
#     Neversink['out'] >> create_edge(1) >> Montague['river']
#     Prompton['out'] >> create_edge(1) >> Montague['river']
#     Wallenpaupack['out'] >> create_edge(1) >> Montague['river']
#     ShoholaMarsh['out'] >> create_edge(1) >> Montague['river']
#     Mongaup['out'] >> create_edge(0) >> Montague['river']
#     Montague['out'] >> create_edge(2) >> Trenton1['river']
#     Beltzville['out'] >> create_edge(2) >> Trenton1['river']
#     FEWalter['out'] >> create_edge(2) >> Trenton1['river']
#     MerrillCreek['out'] >> create_edge(1) >> Trenton1['river']
#     Hopatcong['out'] >> create_edge(1) >> Trenton1['river']
#     Nockamixon['out'] >> create_edge(0) >> Trenton1['river']
#     Trenton1['out'] >> create_edge(0) >> Trenton2['river']
#     Trenton1['out'] >> create_edge(0) >> NJDiversion
#     Trenton2['out'] >> create_edge(0) >> DelawareBay['river']
#     Assunpink['out'] >> create_edge(0) >> DelawareBay['river']
#     Ontelaunee['out'] >> create_edge(2) >> DelawareBay['river']
#     StillCreek['out'] >> create_edge(2) >> DelawareBay['river']
#     BlueMarsh['out'] >> create_edge(2) >> DelawareBay['river']
#     GreenLane['out'] >> create_edge(1) >> DelawareBay['river']
#



