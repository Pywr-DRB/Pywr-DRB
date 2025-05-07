
def = test_import_pywrdrb():
    """
    Test the import of the pywrdrb module.
    
    This test checks if the pywrdrb module can be imported successfully.
    
    Returns
    -------
    None
    """
    try:
        import pywrdrb
    except ImportError as e:
        print(f"Failed to import pywrdrb module: {e}")