""" simple DRB representation, adapted from Thames example
"""
from pywr.model import Model
from pywr.recorders import TablesRecorder
import numpy as np
from matplotlib import pyplot as plt
import click
import time
import json
import pandas

MODEL_FILENAME = "model_data/drb_model_full.json"
OUTPUT_FILENAME = "output_data/drb_output.h5"

@click.group()
def cli():
    pass


@cli.command()
def run():

    # Run the model
    model = Model.load(MODEL_FILENAME)

    # Add a storage recorder
    TablesRecorder(model, OUTPUT_FILENAME, parameters=[p for p in model.parameters])

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
        df.columns = ["Very low", "Low", "Central", "High", "Very high"]

        # if name.split('_')[0] in ("reservoir"):
        # if name.split('_')[0] in ("link", "demand"):
        if name.startswith("mrf_target") or name.startswith("demand_drought_level"):

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

        # if ext is not None:
        #     fig.savefig(f"{name}.{ext}", dpi=300)

    if show:
        plt.show()



if __name__ == "__main__":
    cli()
