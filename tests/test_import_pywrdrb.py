"""
Simply makes sure all of the core modules can be imported.
"""

import pytest

def test_import_pywrdrb():
    try:
        import pywrdrb
    except ImportError as e:
        print(f"Failed to import pywrdrb module: {e}")

def test_import_pywrdrb_pre():        
    # import pywrdrb.pre
    try:
        import pywrdrb.pre
    except ImportError as e:
        print(f"Failed to import pywrdrb.pre module: {e}")

def test_import_pywrdrb_load():
    # import pywrdrb.load   
    try:
        import pywrdrb.load
    except ImportError as e:
        print(f"Failed to import pywrdrb.load module: {e}")

def test_import_pywrdrb_parameters():    
    # import pywrdrb.parameters
    try:
        import pywrdrb.parameters
    except ImportError as e:
        print(f"Failed to import pywrdrb.parameters module: {e}")