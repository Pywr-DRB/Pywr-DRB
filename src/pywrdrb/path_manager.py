"""
Module aims to manage directories within the pywrdrb package.

Overview: 
The module adds a pathnavigator object to the pywrdrb package, which allows users to
easily manage and navigate directories related to the package's data files, as well as 
links to external customized input data. It provides functions to get the pathnavigator 
object, get the pathnavigator configuration, load a configuration from a file, and reset
the pathnavigator to its default state.

Links: 
- Pathnavigator GitHub repository: https://github.com/philip928lin/PathNavigator
 
Change Log:
Chung-Yi Lin, 2025-05-02, None
"""

import os
from copy import deepcopy
import pathnavigator

__all__ = [
    "get_pn_object", 
    "get_pn_config", 
    "load_pn_config", 
    "reset_pn"
    ]

##### Set directory config using pathnavigator
# Get the root directory
root_dir = os.path.realpath(os.path.dirname(__file__))

# Ensure root_dir points to the package directory
if not os.path.basename(root_dir) == "pywrdrb":
    root_dir = os.path.join(root_dir, "pywrdrb")

# To ensure the user can only access the data directory inside the package, not others.
root_dir = os.path.join(root_dir, "data")

# Create a global pathnavigator object with the root directory
global pn
pn = pathnavigator.create(root_dir)

def reset_pn():
    """
    Resets the pathnavigator object to the default configuration.
    
    Notes
    -----
        We only want to add shortcuts that matter to the user.
        For others, advanced users may retrieve from pn directly.
        If we use shortcuts for certain files or folders, we will need to use that shortcut 
        throughout the program to make it consistent.
    """
    global pn  # Ensure pn is modified globally
    # Add folder directories as shortcuts (can be accessed as pn.sc.get("folder name"))
    # Now we only allow users to add customized flows and diversions subfolders 
    # (i.e., flows/customized_flow_type and diversions/customized_diversion_type)
    # # Ignore folders/files starting with "_"
    pn.flows.set_all_to_sc(prefix="flows/", only_folders=True, overwrite=True, only_exclude = ["_*"])
    pn.diversions.set_all_to_sc(prefix="diversions/", only_folders=True, overwrite=True, only_exclude = ["_*"])

def get_pn_object(copy=False):
    """
    Returns the pathnavigator object.
    
    parameters
    ----------
    copy : bool, optional
        If True, returns a copy of the pathnavigator object. Default is False.
    
    Returns
    -------
    pathnavigator.PathNavigator
        The directories object.
    """
    global pn  # Ensure pn is modified globally
    if copy:
        return deepcopy(pn)
    else:
        return pn

def get_pn_config(filename=""):
    """
    Returns the pathnavigator configuration.
    
    Parameters
    ----------
    filename : str, optional
        If a filename is provided, saves the configuration to the file. The allowed file
        extensions are ".json" and ".yml".
    
    Returns
    -------
    dict or None
        If no filename is provided, returns the directories configuration as a dictionary.
        If a filename is provided, saves the configuration to the file and returns None.
    """
    global pn  # Ensure pn is modified globally
    if ".json" in filename:
        pn.sc.to_json(filename)
        print(f"Directories configuration saved to {filename}")
        return None
    elif ".yml" in filename:
        pn.sc.to_yaml(filename)
        print(f"Directories configuration saved to {filename}")
        return None
    else:
        return pn.sc.to_dict(to_str=True)

def load_pn_config(pn_config):
    """
    Loads the directories configuration from a file. Overwrites the existing directories
    with the given configuration. The allowed file extensions are ".json" and ".yml". 
    User may also provide a dictionary as input. Partial configurations can be provided.
    
    If users want to create a model with customized inflow files, they need to add the new
    folder to the directories configuration before creating the model with the customize
    flow_type. Otherwise, the model will not be able to find the inflow files.
    
    Alternatively, users can change the directories under the current directories 
    configuration. That way, the model will be created under the default flow_type while  
    using customized files in the given directories.
    
    Parameters
    ----------
    pn_config : str or dict
        The file path to the directories configuration.
    """
    global pn  # Ensure pn is modified globally
    if ".json" in pn_config:
        pn.sc.load_json(pn_config, overwrite=True)
    elif ".yml" in pn_config:
        pn.sc.load_yaml(pn_config, overwrite=True)
    else:
        pn.sc.load_dict(pn_config, overwrite=True)
