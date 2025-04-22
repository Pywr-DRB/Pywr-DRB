#Define gauges and mappings
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "observations", "_raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "observations")
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")


inflow_gauge_map = {
    "cannonsville": ["01423000", "0142400103"],
    "pepacton": ["01415000", "01414500", "01414000", "01413500"],
    "neversink": ["01435000"],
    "prompton": ["01428750"],
    "beltzvilleCombined": ["01449360"],
    "fewalter": ["01447720", "01447500"],
    "greenLane": ["01472199", "01472198"],
    "01425000": ["01425000"],
    "01417000": ["01417000"],
    "delLordville": ["01427207"],
    "01436000": ["01436000"],
    "01433500": ["01433500"],
    "delMontague": ["01438500"],
    "01449800": ["01449800"],
    "01447800": ["01447800"],
    "delDRCanal": ["01463500"],
    "delTrenton": ["01463500"],
    "01463620": ["01463620"],
    "outletAssunpink": ["01464000"],
    "01470960": ["01470960"],
    "outletSchuylkill": ["01474500"],
}

storage_gauge_map = {
    "beltzvilleCombined": ["01449790"],
    "fewalter": ["01447780"],
    "prompton": ["01428900"],
    "blueMarsh": ["01470870"],
    "cannonsville": ["01423910"],
    "pepacton": ["01414750"],
    "neversink": ["01435900"]
    # fill in others as needed
}

release_gauge_map = {
    "cannonsville": ["01425000"],
    "pepacton": ["01417000"],
    "neversink": ["01436000"],
    "wallenpaupack": ["01438500"],
    "prompton": ["01429000"],
    "shoholaMarsh": ["01438500"],
    "mongaupeCombined": ["01433500"],
    "beltzvilleCombined": ["01449800"],
    "fewalter": ["01447800"],
    "merrillCreek": ["01463500"],
    "hopatcong": ["01463500"],
    "nockamixon": ["01463500"],
    "assunpink": ["01463620"],
    "ontelaunee": ["01474500"],
    "stillCreek": ["01474500"],
    "blueMarsh": ["01470960"],
    "greenLane": ["01474500"],
    "01425000": ["01427207"],
    "01417000": ["01427207"],
    "delLordville": ["01438500"],
    "01436000": ["01438500"],
    "01433500": ["01438500"],
    "delMontague": ["01463500"],
    "01449800": ["01463500"],
    "01447800": ["01463500"],
    "delDRCanal": ["01463500"],
    "delTrenton": ["01463620"],
    "01463620": ["01464000"],
    "outletAssunpink": ["01474500"],
    "01470960": ["01474500"],
    "outletSchuylkill": ["01474500"],
}

storage_curves = {
    "01449790": f"{RAW_DATA_DIR}/beltzvilleCombined_storage_curve.csv", #"beltzvilleCombined"
    "01447780": f"{RAW_DATA_DIR}/fewalter_storage_curve.csv", #fewalter
    "01428900": f"{RAW_DATA_DIR}/prompton_storage_curve.csv", #prompton
    "01470870": f"{RAW_DATA_DIR}/blueMarsh_storage_curve.csv", #blueMarsh
    "01414750": f"{RAW_DATA_DIR}/pepacton_storage_curve.csv",       # Pepacton
    "01423910": f"{RAW_DATA_DIR}/cannonsville_storage_curve.csv",   # Cannonsville
    "01435900": f"{RAW_DATA_DIR}/neversink_storage_curve.csv"       # Neversink
}

nyc_reservoirs = ["pepacton", "cannonsville", "neversink"]
