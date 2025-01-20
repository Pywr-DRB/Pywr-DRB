import matplotlib.pyplot as plt


def timeseries(
    ax,
    output,
    results_set: str,
    models: list[str],
    scenarios: list[int],
    variable: list[str],
    **kwargs,
):
    for model in models:
        for scenario in scenarios:
            for variable in variables:
                if hasattr(output, variable):
                    data = output.__dict__[variable][model][scenario]
                else:
                    raise AttributeError(
                        f"Variable {variable} not found in Output object."
                    )

                if start_date and end_date:
                    data = data[(data.index >= start_date) & (data.index <= end_date)]

                ax.plot(data.index, data.values, label=f"{model}")

    return ax
