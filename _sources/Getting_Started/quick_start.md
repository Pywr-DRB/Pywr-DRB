# Quick Start

Pywr-DRB is an open-source Python model, and should run with Python >= 3.8 on Windows or Linux systems. 

## Accessing Pywr-DRB

Begin by cloning the Pywr-DRB GitHub repository, available here: [https://github.com/Pywr-DRB/Pywr-DRB](https://github.com/Pywr-DRB/Pywr-DRB)

To clone a copy of the repository to your local machine, run the following command:

```
git clone https://github.com/Pywr-DRB/Pywr-DRB.git
```

The folder contains the following content, see the [API References](../../API_References/api_references.md) for detail on how to use the specific functions within the various modules:

```Bash
Pywr-DRB/
├── DRB_spatial/
├── input_data/
│   ├── modeled_gages/
│   ├── usgs_gages/
│   └── WEAP_gridmet/
├── model_data/
├── output_data/
├── figs/
├── plotting/
├── prep_input_data.py
├── custom_pywr.py
├── drb_make_figs.py
├── drb_make_model.py
├── drb_run_sim.py
├── drb_run_all.sh        
├── README.md
└── requirements.txt
```


## Dependencies

Running Pywr-DRB requires several dependencies, listed in [`requirements.txt`](https://github.com/DRB_water_managment/requirements.txt), including:
- [`pywr`](https://pywr.github.io/pywr/index.html)
- `glob2`
- `h5py`
- `hydroeval`

You can install all of the necessary dependencies in a virtual environment:

```Bash
cd <your_local_directory>\Pywr-DRB
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick-Start:Run all simulations

Pywr-DRB is set up with quick-start bash executabe that will:
1. Prepare model input data using all [available inflow datasets](../Supplemental/data_summary.md)
2. Run the Pywr-DRB simulation using each dataset
3. Generate all figures used for analysis

To run all simulations through this quick-start option, execute the [`drb_run_all.sh`](../API_References/drb_run_all.md) script:

```Bash
source venv/bin/activate
sh drb_run_all.sh
```

Alternatively, the user can perform each of the steps listed above individually using individual executables as described in [Running Simulations](./Running_Simulations/run_all_simulations.md), and detailed in the [API References.](../API_References/api_references.md)

