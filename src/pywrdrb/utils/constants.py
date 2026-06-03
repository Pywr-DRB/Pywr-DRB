"""
Contains constants used throughout the package.

Overview: 
It's pretty simple. Constants are used for conversions. 

Technical Notes: 
- This can probably be improved and/or removed in the future, but for now it is used in many places.
    
Links: 
- NA
 
Change Log:
TJA, 2025-05-05, Add docs.
"""

# Constants
cms_to_mgd = 22.82
cm_to_mg = 264.17 / 1e6
mcm_to_mg = 264.17
mg_to_mcm = 1 / mcm_to_mg
cfs_to_mgd = 0.645932368556
epsilon = 1e-5

ACRE_FEET_TO_MG = 0.325851  # Acre-feet to million gallons
GAL_TO_MG = 1 / 1_000_000   # Gallons to million gallons
