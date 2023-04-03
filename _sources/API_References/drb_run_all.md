# `drb_run_all.sh`

This shell script is uses to execute the Pywr-DRB simulation using all available inflow data types, and then makes all figures.

For more information on the different inflow types see Data Summary page. #TODO: Add link.


```bash
### prep inputs from raw data
echo Prepping data...
time python3 -W ignore prep_input_data.py

### run simulation using multiple inpu data sources
for inflow_type in obs_pub nhmv10 nwmv21 nwmv21_withLakes WEAP_23Aug2022_gridmet
do
	backup_inflow_type=nhmv10
	echo Running simulation with $inflow_type ...
	time python3 -W ignore drb_run_sim.py $inflow_type $backup_inflow_type
done

### analyze results, make figures
echo Analyzing results...
time python3 -W ignore drb_make_figs.py
```
