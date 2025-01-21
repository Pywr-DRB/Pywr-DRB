import os

def verify_file_exists(file_path):
    """
    Verify that a file exists at the given path.

    Parameters:
    ----------
    file_path : str
        Full path to the file to be verified.

    Returns:
    -------
    bool
        True if the file exists, False otherwise.

    Raises:
    ------
    ValueError
        If the provided file_path is not a string or is empty.
    """
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("Invalid file path. It must be a non-empty string.")
    
    file_exists = os.path.exists(file_path)
    
    if file_exists:
        return True
    else:
        raise FileNotFoundError(f"File not found at path: {file_path}")