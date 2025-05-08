import pytest
import os
import pywrdrb


def test_root_dir_contains_pywrdrb():
    directories = pywrdrb.get_directory()
    assert os.path.basename(directories.root_dir) == "pywrdrb", f"Expected root_dir to contain 'pywrdrb' but found {os.path.basename(directories.root_dir)}."
    return

def test_input_dir_exists():
    directories = pywrdrb.get_directory()
    assert os.path.exists(directories.input_dir), f"Expected input_dir to exist at {directories.input_dir} but it was not found."
    return

def test_model_data_dir_exists():
    directories = pywrdrb.get_directory()
    assert os.path.exists(directories.model_data_dir), f"Expected model_data_dir to exist at {directories.model_data_dir} but it was not found."
    return