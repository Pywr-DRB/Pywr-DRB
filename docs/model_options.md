# ModelBuilder Options

The `pywrdrb.ModelBuilder` accepts an `options` dictionary which can be used to modify the model configuration. See the `ModelBuilder` class in the API reference for the full list of available options. This page describes two options added in v2.2.0.

## Flow prediction mode

The FFMP-based release rules for the NYC and lower basin reservoirs rely on multi-day predictions of flows at Montague and Trenton. The `flow_prediction_mode` option controls how these predictions are generated:

| Mode | Description |
|------|-------------|
| `"regression_disagg"` | (Default) Regression-based flow predictions, identical to prior versions. |
| `"perfect_foresight"` | Predictions constructed such that the model has perfect knowledge of future non-NYC flow contributions at Montague and Trenton. NYC reservoirs contribute zero to the predicted flows, STARFIT-controlled reservoirs contribute pre-simulated releases generated with `pywrdrb.pre.STARFITOfflineSimulator`, and all other nodes contribute their consumption-adjusted catchment inflows. |
| `"gage_flow"` | Predictions taken directly from the dataset gage flows. |

```python
import pywrdrb

mb = pywrdrb.ModelBuilder(
    start_date="1983-10-01",
    end_date="1985-12-31",
    inflow_type="nhmv10_withObsScaled",
    options={"flow_prediction_mode": "perfect_foresight"},
)
mb.make_model()
mb.write_model("./model_perfect_foresight.json")
```

The pre-packaged datasets include predicted flow columns for both the `regression_disagg` and `perfect_foresight` modes, so either mode can be used without re-running the preprocessors. When using a custom inflow dataset, the corresponding prediction columns must first be generated using `pywrdrb.pre.PredictedInflowPreprocessor` (and `pywrdrb.pre.PredictedDiversionPreprocessor`) with the desired modes.

## Custom STARFIT parameters

By default, the STARFIT reservoir rule parameters are loaded from the `istarf_conus.csv` file included in the package. The `starfit_params_filename` option allows the user to provide an alternative STARFIT parameter CSV:

```python
mb = pywrdrb.ModelBuilder(
    start_date="1983-10-01",
    end_date="1985-12-31",
    inflow_type="nhmv10_withObsScaled",
    options={"starfit_params_filename": "./my_starfit_params.csv"},
)
```

The file must follow the same format as `istarf_conus.csv` and contain rows for all Pywr-DRB reservoirs. The custom parameters are used for both the reservoir capacities and the STARFIT release rules.

Note that this option cannot be combined with the `run_starfit_sensitivity_analysis` option, which loads parameters from a separate scenario file.
