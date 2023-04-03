# `custom_pywr.py`

This code defines a class called `FfmpNycRunningAvgParameter` which is used to implement the Flexible Flow Management Program within the Pywr-DRB model.

The `__init__` method takes in several arguments:

-   `model`: an instance of a model
-   `node`: a node object
-   `max_avg_delivery`: a parameter object that represents the maximum average delivery

In the `setup` method, an empty array is created to hold the parameter state. The size of this array is the number of scenarios in the model.

The `reset` method sets the initial value of the parameter, calculates the timestep and datetime, and sets the max_delivery array to the max_avg_delivery multiplied by the timestep.

The `value` method returns the current volume remaining for the scenario.

The `after` method updates the max_delivery array based on the current datetime. If the current date is May 31st, the max_delivery array is reset to the max_avg_delivery multiplied by the timestep. If it is not May 31st, the max_delivery array is updated based on the running average formula: maxdel_t = maxdel_{t-1} - flow_{t-1} + max_avg_del. The max_delivery array is also updated to ensure that it cannot be less than zero. The datetime is also updated for the next day.

The `load` method is a class method that takes in the model, data, and loads the parameter object with the data passed in.
