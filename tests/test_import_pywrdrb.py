"""
Simply makes sure all of the core modules can be imported.
"""


def test_import_pywrdrb():
    import pywrdrb

    assert hasattr(pywrdrb, "__version__")
    assert pywrdrb.__version__ != "unknown"


def test_import_pywrdrb_pre():
    import pywrdrb.pre


def test_import_pywrdrb_load():
    import pywrdrb.load


def test_import_pywrdrb_parameters():
    import pywrdrb.parameters


def test_import_pywrdrb_post():
    import pywrdrb.post
