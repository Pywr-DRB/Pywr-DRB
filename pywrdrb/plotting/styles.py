"""
This file contains style specifications for figures.

Colorblind-friendly colors taken from here: https://personal.sron.nl/~pault/#sec:qualitative

"""

full_node_label_dict = {}


# pywr_{model} and {model} colors are the same
base_model_colors = {'obs': '#191919', # Dark grey
                'obs_pub':'#0f6da9', # grey-blue
                'nhmv10': '#B58800', # yellow
                'nwmv21': '#9E1145', # green
                'nwmv21_withLakes': '#9E1145', # green 
                'WEAP_24Apr2023_gridmet': '#1F6727', # teal cyan
                'pywr_obs_pub':'#0f6da9', # darker grey-blue
                'pywr_nhmv10': '#B58800', # darker yellow
                'pywr_nwmv21': '#9E1145', # darker pink
                'pywr_nwmv21_withLakes': '#9E1145', # darker pink
                'pywr_WEAP_24Apr2023_gridmet': '#1F6727'} # darker teal


# pywr_{model} colors are darker than {model} colors
paired_model_colors = {'obs': '#2B2B2B',
                'obs_pub': '#7db9e5', # grey-blue
                'nhmv10': '#FDE088', # yellow
                'nwmv21': '#FF91B9', # pink
                'nwmv21_withLakes': '#FF91B9', # green 
                'WEAP_24Apr2023_gridmet': '#95F19F', # lt grn
                'pywr_obs_pub':'#0f6da9', # darker grey-blue
                'pywr_nhmv10': '#B58800', # darker yellow
                'pywr_nwmv21': '#9E1145', # darker green
                'pywr_nwmv21_withLakes': '#9E1145', # darker green
                'pywr_WEAP_24Apr2023_gridmet': '#1F6727'} # drk grn
base_marker = "o"
pywr_marker = "x"

scatter_model_markers = {'obs': base_marker, # Dark grey
                'obs_pub': base_marker, # grey-blue
                'nhmv10': base_marker, # yellow
                'nwmv21': base_marker, # green
                'nwmv21_withLakes': base_marker, # green 
                'WEAP_24Apr2023_gridmet': base_marker, # mauve
                'pywr_obs_pub': pywr_marker, # darker grey-blue
                'pywr_nhmv10': pywr_marker, # darker yellow
                'pywr_nwmv21': pywr_marker, # darker green
                'pywr_nwmv21_withLakes': pywr_marker, # darker green
                'pywr_WEAP_24Apr2023_gridmet': pywr_marker} # darker mauve

node_colors = {}


model_hatch_styles = {'obs': '',
                        'obs_pub': '', 
                        'nhmv10': '', 
                        'nwmv21': '', 
                        'nwmv21_withLakes': '', 
                        'WEAP_24Apr2023_gridmet': '',
                        'pywr_nhmv10': '///',
                        'pywr_obs_pub': '///', 
                        'pywr_nwmv21': '///', 
                        'pywr_nwmv21_withLakes': '///',
                        'pywr_WEAP_24Apr2023_gridmet': '///'}

