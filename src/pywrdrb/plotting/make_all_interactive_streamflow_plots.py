from interactive_streamflow import plot_interactive_streamflow_stack

node = "delTrenton"
all_models = ["obs_pub", "nhmv10", "WEAP_24Apr2023_gridmet", "nwmv21"]

for model in all_models:
    plot_interactive_streamflow_stack(
        node, model, output_dir="../output_data/", fig_dir="../figs/"
    )
