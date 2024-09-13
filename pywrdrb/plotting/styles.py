"""
This file contains style specifications for figures.
"""
import matplotlib.cm as cm

full_node_label_dict = {}

syn_model_colors = {
    "syn_obs_pub_nhmv10_ObsScaled_ensemble": "orange",
    "syn_obs_pub_nwmv21_ObsScaled_ensemble": "blue",
    "pywr_syn_obs_pub_nhmv10_ObsScaled_ensemble": "orange",
    "pywr_syn_obs_pub_nwmv21_ObsScaled_ensemble": "cornflowerblue",
}


def get_model_color(model, colordict, default="darkorange"):
    """
    Return color for a specific model from a dictionary;
    if model is not in the dictionary, return the default color.

    Args:
        model (str): Model name.
        colordict (dict): Dictionary of model colors.
        default (str): Default color to use if model is not in the dictionary.

    Returns:
        str: Color for the model.
    """

    # check if model is in the dict
    if model in colordict.keys():
        return colordict[model]
    elif f"pywr_{model}" in colordict.keys():
        return colordict[f"pywr_{model}"]
    elif f"syn_{model}" in colordict.keys():
        return colordict[f"syn_{model}"]
    else:
        print(
            "Warning: Model not found in color dictionary. Using default color: {default}."
        )
        return default


# pywr_{model} and {model} colors are the same
base_model_colors = {
    "obs": "#191919",  # Dark grey
    "obs_pub_nhmv10": "#0f6da9",  # grey-blue
    "obs_pub_nwmv21": "#0f6da9",  # grey-blue
    "obs_pub_nhmv10_ObsScaled": "#0f6da9",  # grey-blue
    "obs_pub_nwmv21_ObsScaled": "#0f6da9",  # grey-blue
    "nhmv10": "#B58800",  # yellow
    "nwmv21": "#9E1145",  # green
    "nwmv21_withLakes": "#9E1145",  # green
    "WEAP_29June2023_gridmet": "#1F6727",  # teal cyan
    "pywr_obs_pub_nhmv10": "darkorange",
    "pywr_obs_pub_nwmv21": "forestgreen",
    "pywr_obs_pub_nhmv10_ObsScaled": "#0f6da9",  # darker grey-blue
    "pywr_obs_pub_nwmv21_ObsScaled": "#0f6da9",  # darker grey-blue
    "pywr_nhmv10": "#B58800",  # darker yellow
    "pywr_nwmv21": "#9E1145",  # darker pink
    "pywr_nwmv21_withLakes": "#9E1145",  # darker pink
    "pywr_WEAP_29June2023_gridmet": "#1F6727",
}  # darker teal


# pywr_{model} colors are darker than {model} colors
paired_model_colors = {
    "obs": "#2B2B2B",
    "obs_pub_nhmv10_ObsScaled": "#7db9e5",  # grey-blue
    "obs_pub_nwmv21_ObsScaled": "#7db9e5",  # grey-blue
    "obs_pub_nhmv10": "#7db9e5",  # grey-blue
    "obs_pub_nwmv21": "#7db9e5",  # grey-blue
    "nhmv10": "#FDE088",  # yellow
    "nwmv21": "#FF91B9",  # pink
    "nwmv21_withLakes": "#FF91B9",  # green
    "WEAP_29June2023_gridmet": "#95F19F",  # lt grn
    "pywr_obs_pub_nhmv10": "#0f6da9",  # darker grey-blue
    "pywr_obs_pub_nwmv21": "#0f6da9",  # darker grey-blue
    "pywr_obs_pub_nhmv10_ObsScaled": "#0f6da9",  # darker grey-blue
    "pywr_obs_pub_nwmv21_ObsScaled": "#0f6da9",  # darker grey-blue
    "pywr_nhmv10": "#B58800",  # darker yellow
    "pywr_nwmv21": "#9E1145",  # darker green
    "pywr_nwmv21_withLakes": "#9E1145",  # darker green
    "pywr_WEAP_29June2023_gridmet": "#1F6727",
}  # drk grn

model_colors_diagnostics_paper = {
    "obs": "0.5",
    "nhmv10": cm.get_cmap("Reds")(0.4),
    "nwmv21": cm.get_cmap("Oranges")(0.4),
    "nhmv10_withObsScaled": cm.get_cmap("Purples")(0.4),
    "nwmv21_withObsScaled": cm.get_cmap("Blues")(0.4),
    "pywr_nhmv10": cm.get_cmap("Reds")(0.8),
    "pywr_nwmv21": cm.get_cmap("Oranges")(0.8),
    "pywr_nhmv10_withObsScaled": cm.get_cmap("Purples")(0.8),
    "pywr_nwmv21_withObsScaled": cm.get_cmap("Blues")(0.9),
}


model_colors_diagnostics_paper2 = {
    "obs": "0.4",
    "nhmv10": cm.get_cmap("Greens")(0.6),
    "nwmv21": cm.get_cmap("Greens")(0.6),
    "nhmv10_withObsScaled": cm.get_cmap("Greens")(0.6),
    "nwmv21_withObsScaled": cm.get_cmap("Greens")(0.6),
    "pywr_nhmv10": cm.get_cmap("Purples")(0.6),
    "pywr_nwmv21": cm.get_cmap("Purples")(0.6),
    "pywr_nhmv10_withObsScaled": cm.get_cmap("Purples")(0.6),
    "pywr_nwmv21_withObsScaled": cm.get_cmap("Purples")(0.6),
}

model_colors_diagnostics_paper3 = {
    "obs": "0.4",
    "nhmv10": cm.get_cmap("Greens")(0.3),
    "nwmv21": cm.get_cmap("Greens")(0.5),
    "nhmv10_withObsScaled": cm.get_cmap("Greens")(0.7),
    "nwmv21_withObsScaled": cm.get_cmap("Greens")(0.9),
    "pywr_nhmv10": cm.get_cmap("Purples")(0.3),
    "pywr_nwmv21": cm.get_cmap("Purples")(0.5),
    "pywr_nhmv10_withObsScaled": cm.get_cmap("Purples")(0.7),
    "pywr_nwmv21_withObsScaled": cm.get_cmap("Purples")(0.9),
}


model_label_dict = {
    "obs": "Observed",
    "nhmv10": "NHMv1.0",
    "nwmv21": "NWMv2.1",
    "obs_pub_nhmv10": "PUB-NHM",
    "obs_pub_nhmv10_ensemble": "PUB-NHM Ensemble",
    "obs_pub_nwmv21": "PUB-NWM",
    "obs_pub_nwmv21_ensemble": "PUB-NWM Ensemble",
    "obs_pub_nhmv10_ObsScaled": "PUB-NHM",
    "obs_pub_nhmv10_ObsScaled_ensemble": "PUB-NHM Ensemble",
    "obs_pub_nwmv21_ObsScaled": "PUB-NWM",
    "obs_pub_nwmv21_ObsScaled_ensemble": "PUB-NWM Ensemble",
    "obs_pub_nhmv10_BC_ObsScaled": "PUB-NHM Bias Corrected",
    "obs_pub_nhmv10_BC_ObsScaled_ensemble": "PUB-NHM Bias Corrected Ensemble",
    "obs_pub_nwmv21_BC_ObsScaled": "PUB-NWM Bias Corrected",
    "obs_pub_nwmv21_BC_ObsScaled_ensemble": "PUB-NWM Bias Corrected Ensemble",
    "wrf1960s_calib_nlcd2016": "WRF 1960s Calibrated",
    "wrf2050s_calib_nlcd2016": "WRF 2050s Calibrated",
}

for l in list(model_label_dict.keys()):
    model_label_dict[f"pywr_{l}"] = "Pywr-DRB " + model_label_dict[l]

model_linestyle_dict = {
    "obs": "-",
    "pywr_nhmv10": (0, (1, 3)),
    "pywr_nwmv21": (0, (3, 6)),
    "pywr_nhmv10_withObsScaled": (0, (1, 1)),
    "pywr_nwmv21_withObsScaled": (0, (3, 1)),
}

node_label_dict = {
    "pepacton": "Pep",
    "cannonsville": "Can",
    "neversink": "Nev",
    "prompton": "Pro",
    "assunpink": "AspRes",
    "beltzvilleCombined": "Bel",
    "blueMarsh": "Blu",
    "mongaupeCombined": "Mgp",
    "fewalter": "FEW",
    "delLordville": "Lor",
    "delMontague": "Mtg",
    "delTrenton": "Tre",
    "outletAssunpink": "Asp",
    "outletSchuylkill": "Sch",
}

node_label_full_dict = {
    "pepacton": "Pepacton",
    "cannonsville": "Cannonsville",
    "neversink": "Neversink",
    "prompton": "Pro",
    "assunpink": "AspRes",
    "beltzvilleCombined": "Bel",
    "blueMarsh": "Blu",
    "mongaupeCombined": "Mgp",
    "fewalter": "FEW",
    "delLordville": "Lor",
    "delMontague": "Montague",
    "delTrenton": "Trenton",
    "outletAssunpink": "Asp",
    "outletSchuylkill": "Schuylkill",
    "NYCAgg": "NYC Total",
}

month_dict = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

base_marker = "o"
pywr_marker = "x"
scatter_model_markers = {
    "obs": base_marker,
    "nhmv10": base_marker,
    "nwmv21": base_marker,
    "nhmv10_withObsScaled": base_marker,
    "nwmv21_withObsScaled": base_marker,
    "obs_pub_nwmv21_ObsScaled": base_marker,
    "obs_pub_nhmv10_ObsScaled": base_marker,
    "obs_pub_nhmv10_ObsScaled_ensemble": base_marker,
    "obs_pub_nwmv21_ObsScaled_ensemble": base_marker,
    "pywr_nhmv10": pywr_marker,
    "pywr_nwmv21": pywr_marker,
    "pywr_nhmv10_withObsScaled": pywr_marker,
    "pywr_nwmv21_withObsScaled": pywr_marker,
    "pywr_obs_pub_nhmv10_ObsScaled": pywr_marker,
    "pywr_obs_pub_nwmv21_ObsScaled": pywr_marker,
    "pywr_obs_pub_nhmv10_ObsScaled_ensemble": pywr_marker,
    "pywr_obs_pub_nwmv21_ObsScaled_ensemble": pywr_marker,
}

node_colors = {}


model_hatch_styles = {
    "obs": "",
    "obs_pub_nhmv10_ObsScaled": "",
    "obs_pub_nwmv21_ObsScaled": "",
    "obs_pub_nhmv10": "",
    "obs_pub_nwmv21": "",
    "nhmv10": "",
    "nwmv21": "",
    "nwmv21_withLakes": "",
    "nhmv10_withObsScaled": "",
    "nwmv21_withObsScaled": "",
    "WEAP_29June2023_gridmet": "",
    "pywr_nhmv10": "///",
    "pywr_obs_pub_nhmv10_ObsScaled": "///",
    "pywr_obs_pub_nwmv21_ObsScaled": "///",
    "pywr_obs_pub_nhmv10": "///",
    "pywr_obs_pub_nwmv21": "///",
    "pywr_nwmv21": "///",
    "pywr_nwmv21_withLakes": "///",
    "pywr_WEAP_29June2023_gridmet": "///",
    "pywr_nhmv10_withObsScaled": "///",
    "pywr_nwmv21_withObsScaled": "///",
}

model_colors_historic_reconstruction = {
    "obs": "black",
    "nhmv10": "#925736",
    "nwmv21": "#385723",
    "obs_pub_nhmv10": "#F27300",
    "obs_pub_nhmv10_ObsScaled": "#F27300",
    "obs_pub_nhmv10_BC_ObsScaled": "#F27300",
    "obs_pub_nhmv10_ensemble": "#F9B572",
    "obs_pub_nhmv10_ObsScaled_ensemble": "#F9B572",
    "obs_pub_nhmv10_BC_ObsScaled_ensemble": "#F9B572",
    "obs_pub_nwmv21": "#0174BE",
    "obs_pub_nwmv21_ensemble": "#9CD2F6",
    "obs_pub_nwmv21_ObsScaled": "#0174BE",
    "obs_pub_nwmv21_BC_ObsScaled": "#0174BE",
    "obs_pub_nwmv21_ObsScaled_ensemble": "#9CD2F6",
    "obs_pub_nwmv21_BC_ObsScaled_ensemble": "#9CD2F6",
}

for m in list(model_colors_historic_reconstruction.keys()):
    model_colors_historic_reconstruction[
        f"pywr_{m}"
    ] = model_colors_historic_reconstruction[m]


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
# model_colors_diagnostics_paper2 = {'obs': '0.4',
#                                   'nhmv10': '#D8B70A',
#                                   'nwmv21': '#D8B70A',
#                                   'nhmv10_withObsScaled': '#D8B70A',
#                                   'nwmv21_withObsScaled': '#D8B70A',
#                                   'pywr_nhmv10': cm.get_cmap('Purples')(0.8),
#                                   'pywr_nwmv21': cm.get_cmap('Purples')(0.8),
#                                   'pywr_nhmv10_withObsScaled': cm.get_cmap('Purples')(0.8),
#                                   'pywr_nwmv21_withObsScaled': cm.get_cmap('Purples')(0.8)
#                                   }
