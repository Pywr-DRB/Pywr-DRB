### script to update the kernel for all Jupyter notebooks to user-specified name, given as argv
### this will be run by GitHub action bto set to .venv efore deploying website. Can also be run by user if their env has different name.
import glob
import json
import sys

venv_name = sys.argv[1]
nbpaths = glob.glob('*.ipynb') + glob.glob('*/*.ipynb') + glob.glob('*/*/*.ipynb')

for nbpath in nbpaths:
    ### load jupyter notebook into dict based on its json format
    with open(nbpath) as f:
        nbdict = json.load(f)
    ### update kernel metadata
    nbdict['metadata']['kernelspec']['display_name'] = venv_name
    nbdict['metadata']['kernelspec']['name'] = venv_name
    ### write updated dict to json/ipynb
    with open(nbpath, 'w') as f:
        json.dump(nbdict, f, indent=2)
