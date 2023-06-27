import os
import sys
print(f'CURRENT: {os.getcwd()}')
print(f'Adding path: {os.path.abspath("../")}')
print(f'Adding path: {os.path.abspath("../pywrdrb/")}')
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../pywrdrb'))