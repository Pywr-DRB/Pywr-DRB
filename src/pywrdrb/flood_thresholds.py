"""
Flood stage thresholds for monitoring locations.

Values in feet above gage datum.

Sources:
- NWS Advanced Hydrologic Prediction Service (AHPS)
- 2017 FFMP Appendix A, Section G.4.e-g
- USGS gage metadata

Change Log:
TJA, 2026-01-08, Initial implementation for flood monitoring.
"""

# Stage thresholds in feet above gage datum
flood_stage_thresholds = {
    # Hale Eddy (01426500) - Gage datum: 946.46 ft NGVD29
    "01426500": {
        "action": 9.0,      # FFMP L1 release cutoff
        "minor": 11.0,      # NWS flood stage
        "moderate": 13.0,   # NWS moderate flood
        "major": 15.0,      # NWS major flood
    },

    # Fishs Eddy (01421000) - Gage datum: 955.96 ft NGVD29
    "01421000": {
        "action": 11.0,     # FFMP L1 release cutoff
        "minor": 13.0,      # NWS flood stage
        "moderate": 15.0,   # NWS moderate flood
        "major": 18.0,      # NWS major flood
    },

    # Bridgeville (01436690)
    "01436690": {
        "action": 12.0,     # NWS action stage (conservative)
        "minor": 13.0,      # NWS flood stage
        "moderate": 17.0,   # NWS moderate flood
        "major": 19.0,      # NWS major flood
    },

    # Montague (01438500)
    "delMontague": {
        "action": 19.0,
        "minor": 25.0,      # NWS flood stage
        "moderate": 30.0,
        "major": 35.0,
    },

    # Trenton (01463500)
    "delTrenton": {
        "action": 20.0,
        "minor": 22.0,      # NWS flood stage
        "moderate": 25.0,
        "major": 28.0,
    },
}
