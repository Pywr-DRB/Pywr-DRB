""" simple DRB representation, adapted from Thames example
"""
from pywr.model import Model
from pywr.recorders import TablesRecorder
import numpy as np
from matplotlib import pyplot as plt
import click

### import custom pywr params
from custom_pywr import FfmpNycRunningAvgParameter, FfmpNjRunningAvgParameter
FfmpNycRunningAvgParameter.register()  # register the name so it can be loaded from JSON
FfmpNjRunningAvgParameter.register()  # register the name so it can be loaded from JSON

MODEL_FILENAME = "model_data/drb_model_full.json"
OUTPUT_FILENAME = "output_data/drb_output_nwmv21_noScaled.hdf5"


@click.group()
def cli():
    pass


@cli.command()
def run():

    # Run the model
    model = Model.load(MODEL_FILENAME)

    # Add a storage recorder
    TablesRecorder(model, OUTPUT_FILENAME, parameters=[p for p in model.parameters if p.name])

    # Run the model
    stats = model.run()
    print(stats)
    stats_df = stats.to_dataframe()
    print(stats_df)

@cli.command()
@click.option("--ext", default="png")
@click.option("--show/--no-show", default=False)
def figures(ext, show):

    for name, df in TablesRecorder.generate_dataframes(OUTPUT_FILENAME):
        # df.columns = ["Very low", "Low", "Central", "High", "Very high"]
        df.columns = ["Central"]

        # if name.split('_')[0] in ("reservoir"):
        # if name.split('_')[0] in ("reservoir", "outflow", "flow", "link", "flowtarget", "demand"):
        # if 'target' in name.split('_'):
        # reservoir = 'neversink'
        # if name in ('reservoir_'+reservoir, 'flow_'+reservoir, 'outflow_'+reservoir, 'link_'+reservoir+'_nyc'):
        if name in ('max_flow_ffmp_delivery_nj', 'drought_factor_delivery_nyc', 'delivery_nj', 'demand_nj'):
        # if 'factor_trenton' in name or 'outflow_trenton' in name or 'target_trenton' in name: # or name == 'max_flow_delivery_nyc':
            fig, (ax1, ax2) = plt.subplots(
                figsize=(12, 4), ncols=2, sharey="row", gridspec_kw={"width_ratios": [3, 1]}
            )
            df.plot(ax=ax1)
            df.quantile(np.linspace(0, 1)).plot(ax=ax2)

            if name.startswith("reservoir"):
                ax1.set_ylabel("Volume [MG]")
            else:
                ax1.set_ylabel("Flow [MGD]")

            for ax in (ax1, ax2):
                ax.set_title(name)
                ax.grid(True)
            plt.tight_layout()
            print(name, df.min(), df.max())
            if ext is not None:
                fig.savefig(f"figs/{name}.{ext}", dpi=300)

    if show:
        plt.show()



if __name__ == "__main__":
    cli()
