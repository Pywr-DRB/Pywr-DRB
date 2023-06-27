import numpy as np

class test(n):
    '''
    Custom Pywr parameter class to implement STARFIT reservoir policy.
    '''
    def __init__(self):
        """
        Initialize the STARFITReservoirRelease parameter.

        Args:
            n (int): a number.
        """
        self.n = n
