"""
This file contains style specifications for figures.
"""
import matplotlib.cm as cm

full_node_label_dict = {}


# pywr_{model} and {model} colors are the same
base_model_colors = {'obs': '#191919', # Dark grey
                'obs_pub_nhmv10':'#0f6da9', # grey-blue
                'obs_pub_nwmv21':'#0f6da9', # grey-blue
                'obs_pub_nhmv10_ObsScaled':'#0f6da9', # grey-blue
                'obs_pub_nwmv21_ObsScaled':'#0f6da9', # grey-blue
                'nhmv10': '#B58800', # yellow
                'nwmv21': '#9E1145', # green
                'nwmv21_withLakes': '#9E1145', # green 
                'WEAP_29June2023_gridmet': '#1F6727', # teal cyan
                'pywr_obs_pub_nhmv10':'darkorange', 
                'pywr_obs_pub_nwmv21':'forestgreen', 
                'pywr_obs_pub_nhmv10_ObsScaled':'#0f6da9', # darker grey-blue
                'pywr_obs_pub_nwmv21_ObsScaled':'#0f6da9', # darker grey-blue
                'pywr_nhmv10': '#B58800', # darker yellow
                'pywr_nwmv21': '#9E1145', # darker pink
                'pywr_nwmv21_withLakes': '#9E1145', # darker pink
                'pywr_WEAP_29June2023_gridmet': '#1F6727'} # darker teal


# pywr_{model} colors are darker than {model} colors
paired_model_colors = {'obs': '#2B2B2B',
                'obs_pub_nhmv10_ObsScaled': '#7db9e5', # grey-blue
                'obs_pub_nwmv21_ObsScaled': '#7db9e5', # grey-blue
                'obs_pub_nhmv10': '#7db9e5', # grey-blue
                'obs_pub_nwmv21': '#7db9e5', # grey-blue
                'nhmv10': '#FDE088', # yellow
                'nwmv21': '#FF91B9', # pink
                'nwmv21_withLakes': '#FF91B9', # green 
                'WEAP_29June2023_gridmet': '#95F19F', # lt grn
                'pywr_obs_pub_nhmv10':'#0f6da9', # darker grey-blue
                'pywr_obs_pub_nwmv21':'#0f6da9', # darker grey-blue
                'pywr_obs_pub_nhmv10_ObsScaled':'#0f6da9', # darker grey-blue
                'pywr_obs_pub_nwmv21_ObsScaled':'#0f6da9', # darker grey-blue
                'pywr_nhmv10': '#B58800', # darker yellow
                'pywr_nwmv21': '#9E1145', # darker green
                'pywr_nwmv21_withLakes': '#9E1145', # darker green
                'pywr_WEAP_29June2023_gridmet': '#1F6727'} # drk grn

model_colors_diagnostics_paper = {'obs': '0.5',
                                  'nhmv10': cm.get_cmap('Reds')(0.4),
                                  'nwmv21': cm.get_cmap('Oranges')(0.4),
                                  'nhmv10_withObsScaled': cm.get_cmap('Purples')(0.4),
                                  'nwmv21_withObsScaled': cm.get_cmap('Blues')(0.4),
                                  'pywr_nhmv10': cm.get_cmap('Reds')(0.8),
                                  'pywr_nwmv21': cm.get_cmap('Oranges')(0.8),
                                  'pywr_nhmv10_withObsScaled': cm.get_cmap('Purples')(0.8),
                                  'pywr_nwmv21_withObsScaled': cm.get_cmap('Blues')(0.9)
                                  }
# model_colors_diagnostics_paper2 = {'obs': '0.5',
#                                   'nhmv10': '#D8B70A',
#                                   'nwmv21': '#D8B70A',
#                                   'nhmv10_withObsScaled': '#D8B70A',
#                                   'nwmv21_withObsScaled': '#D8B70A',
#                                   'pywr_nhmv10': '#02401B',
#                                   'pywr_nwmv21': '#02401B',
#                                   'pywr_nhmv10_withObsScaled': '#02401B',
#                                   'pywr_nwmv21_withObsScaled': '#02401B'
#                                   }
model_colors_diagnostics_paper2 = {'obs': '0.4',
                                  'nhmv10': '#D8B70A',
                                  'nwmv21': '#D8B70A',
                                  'nhmv10_withObsScaled': '#D8B70A',
                                  'nwmv21_withObsScaled': '#D8B70A',
                                  'pywr_nhmv10': cm.get_cmap('Purples')(0.8),
                                  'pywr_nwmv21': cm.get_cmap('Purples')(0.8),
                                  'pywr_nhmv10_withObsScaled': cm.get_cmap('Purples')(0.8),
                                  'pywr_nwmv21_withObsScaled': cm.get_cmap('Purples')(0.8)
                                  }
model_colors_diagnostics_paper3 = {'obs': '0.4',
                                  'nhmv10': '#D8B70A',
                                  'nwmv21': '#D8B70A',
                                  'nhmv10_withObsScaled': '#D8B70A',
                                  'nwmv21_withObsScaled': '#D8B70A',
                                  'pywr_nhmv10': '#D8B70A',
                                  'pywr_nwmv21': '#A2A475',
                                  'pywr_nhmv10_withObsScaled': '#02401B',
                                  'pywr_nwmv21_withObsScaled': '#81A88D'
                                  }

model_label_dict = {'obs': 'Observed',
                      'nhmv10': 'NHM',
                      'nwmv21': 'NWM',
                      'nhmv10_withObsScaled': 'hNHM',
                      'nwmv21_withObsScaled': 'hNWM',
                      'pywr_nhmv10': 'Pywr-NHM',
                      'pywr_nwmv21': 'Pywr-NWM',
                      'pywr_nhmv10_withObsScaled': 'Pywr-hNHM',
                      'pywr_nwmv21_withObsScaled': 'Pywr-hNWM'
                      }

model_linestyle_dict = {'obs': '-',
                        'pywr_nhmv10': (0,(1,3)),
                        'pywr_nwmv21': (0,(3,6)),
                        'pywr_nhmv10_withObsScaled': (0,(1,1)),
                        'pywr_nwmv21_withObsScaled': (0,(3,1))
                        }

node_label_dict = {'pepacton': 'Pep', 'cannonsville': 'Can', 'neversink': 'Nev', 'prompton': 'Pro', 'assunpink': 'AspRes', \
                   'beltzvilleCombined': 'Bel', 'blueMarsh': 'Blu', 'mongaupeCombined': 'Mgp', 'fewalter': 'FEW', \
                   'delLordville': 'Lor', 'delMontague': 'Mtg', 'delTrenton': 'Tre', 'outletAssunpink': 'Asp', \
                   'outletSchuylkill': 'Sch'}

node_label_full_dict = {'pepacton': 'Pepacton', 'cannonsville': 'Cannonsville', 'neversink': 'Neversink', 'prompton': 'Pro', 'assunpink': 'AspRes', \
                   'beltzvilleCombined': 'Bel', 'blueMarsh': 'Blu', 'mongaupeCombined': 'Mgp', 'fewalter': 'FEW', \
                   'delLordville': 'Lor', 'delMontague': 'Montague', 'delTrenton': 'Trenton', 'outletAssunpink': 'Asp', \
                   'outletSchuylkill': 'Schuylkill', 'NYCAgg': 'NYC Total'}

month_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

base_marker = "o"
pywr_marker = "x"
scatter_model_markers = {'obs': base_marker, #
                         'obs_pub_nhmv10_ObsScaled': base_marker, #
                         'obs_pub_nwmv21_ObsScaled': base_marker, #
                         'nhmv10': base_marker,
                         'nwmv21': base_marker,
                         'nwmv21_withLakes': base_marker,
                         'nhmv10_withObsScaled': base_marker,
                         'nwmv21_withObsScaled': base_marker,
                         'WEAP_29June2023_gridmet': base_marker,
                         'pywr_obs_pub_nhmv10_ObsScaled': pywr_marker, #
                         'pywr_obs_pub_nwmv21_ObsScaled': pywr_marker, #
                         'pywr_nhmv10': pywr_marker, #
                         'pywr_nwmv21': pywr_marker, #
                         'pywr_nwmv21_withLakes': pywr_marker, #
                         'pywr_WEAP_29June2023_gridmet': pywr_marker,
                         'pywr_nhmv10_withObsScaled': pywr_marker,
                         'pywr_nwmv21_withObsScaled': pywr_marker
                         }

node_colors = {}


model_hatch_styles = {'obs': '',
                        'obs_pub_nhmv10_ObsScaled': '',
                        'obs_pub_nwmv21_ObsScaled': '',
                        'obs_pub_nhmv10': '',
                        'obs_pub_nwmv21': '',
                        'nhmv10': '', 
                        'nwmv21': '', 
                        'nwmv21_withLakes': '',
                        'nhmv10_withObsScaled': '',
                        'nwmv21_withObsScaled': '',
                        'WEAP_29June2023_gridmet': '',
                        'pywr_nhmv10': '///',
                        'pywr_obs_pub_nhmv10_ObsScaled': '///',
                        'pywr_obs_pub_nwmv21_ObsScaled': '///',
                        'pywr_obs_pub_nhmv10': '///',
                        'pywr_obs_pub_nwmv21': '///',
                        'pywr_nwmv21': '///',
                        'pywr_nwmv21_withLakes': '///',
                        'pywr_WEAP_29June2023_gridmet': '///',
                        'pywr_nhmv10_withObsScaled': '///',
                        'pywr_nwmv21_withObsScaled': '///'
}

