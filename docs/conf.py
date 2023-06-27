import os
import sys
### add error to verify that this isnt actually getting run
a = b


print(f'CURRENT: {os.getcwd()}')
print(f'Adding path: {os.path.abspath("../")}')
print(f'Adding path: {os.path.abspath("../pywrdrb/")}')
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../pywrdrb'))
sys.path.insert(0, os.path.abspath('./pywrdrb'))
