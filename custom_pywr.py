### custom parameters for pywr
import numpy as np
import pandas as pd
from pywr.parameters import Parameter, load_parameter



class FfmpNycRunningAvgParameter(Parameter):
    def __init__(self, model, node, max_avg_delivery, **kwargs):
        super().__init__(model, **kwargs)
        self.node = node
        self.max_avg_delivery = max_avg_delivery.get_constant_value()
        self.children.add(max_avg_delivery)

    def setup(self):
        ### allocate an array to hold the parameter state
        super().setup()
        num_scenarios = len(self.model.scenarios.combinations)
        self.max_delivery = np.empty([num_scenarios], np.float64)

    def reset(self):
        ### reset the amount remaining in all states to the initial value
        self.timestep = self.model.timestepper.delta
        self.datetime = pd.to_datetime(self.model.timestepper.start)
        self.max_delivery[...] = self.max_avg_delivery * self.timestep

    def value(self, timestep, scenario_index):
        ### return the current volume remaining for the scenario
        return self.max_delivery[scenario_index.global_id]

    def after(self):
        ### if it is may 31, reset max delivery to original value (800)
        if self.datetime.month == 5 and self.datetime.day == 31:
            self.max_delivery[...] = self.max_avg_delivery * self.timestep
        ### else update the requirement based on running average: maxdel_t = maxdel_{t-1} - flow_{t-1} + max_avg_del
        else:
            self.max_delivery += (self.max_avg_delivery - self.node.flow) * self.timestep
            self.max_delivery[self.max_delivery < 0] = 0  # max delivery cannot be less than zero
        ### update date for tomorrow
        self.datetime += pd.Timedelta(1, 'd')

    @classmethod
    def load(cls, model, data):
        node = model.nodes[data.pop("node")]
        max_avg_delivery = load_parameter(model, data.pop("max_avg_delivery"))
        return cls(model, node, max_avg_delivery, **data)





class FfmpNjRunningAvgParameter(Parameter):
    def __init__(self, model, node, max_avg_delivery, max_daily_delivery, drought_factor, **kwargs):
        super().__init__(model, **kwargs)
        self.node = node
        self.max_avg_delivery = max_avg_delivery.get_constant_value()
        self.max_daily_delivery = max_daily_delivery.get_constant_value()
        self.drought_factor = drought_factor
        self.children.add(max_avg_delivery)
        self.children.add(max_daily_delivery)
        self.children.add(drought_factor)

    def setup(self):
        ### allocate an array to hold the parameter state
        super().setup()
        num_scenarios = len(self.model.scenarios.combinations)
        self.max_delivery = np.empty([num_scenarios], np.float64)
        self.current_drought_factor = np.empty([num_scenarios], np.float64)
        self.previous_drought_factor = np.empty([num_scenarios], np.float64)

    def reset(self):
        ### reset the amount remaining in all states to the initial value
        self.timestep = self.model.timestepper.delta
        self.datetime = pd.to_datetime(self.model.timestepper.start)
        self.max_delivery[...] = self.max_avg_delivery * self.timestep
        self.current_drought_factor[...] = 1.0
        self.previous_drought_factor[...] = 1.0

    def value(self, timestep, scenario_index):
        ### return the current volume remaining for the scenario
        return self.max_delivery[scenario_index.global_id]

    def after(self):
        self.current_drought_factor[...] = self.drought_factor.get_all_values()
        ### loop over scenarios
        for s, factor in enumerate(self.current_drought_factor):
            ### first check if today's drought_factor is same as yesterday's
            if factor == self.previous_drought_factor[s]:
                ### if today is same drought factor as yesterday, and factor is 1.0, we reset running avg on first day of each month
                if (self.datetime.day == 1) and (factor == 1.0):
                    self.max_delivery[s] = self.max_avg_delivery * factor * self.timestep
                ### else, update requirement based on running average: maxdel_t = maxdel_{t-1} - flow_{t-1} + max_avg_del
                else:
                    self.max_delivery[s] += (self.max_avg_delivery * factor - self.node.flow[s]) * self.timestep
            ### if today's drought factor is different from yesterday, we always reset running avg
            else:
                self.max_delivery[s] = self.max_avg_delivery * factor * self.timestep

        ### max delivery cannot be less than zero
        self.max_delivery[self.max_delivery < 0] = 0
        ### max delivery cannot be larger than daily limit
        self.max_delivery[self.max_delivery > self.max_daily_delivery] = self.max_daily_delivery
        ### update date & previous factor for tomorrow
        self.datetime += pd.Timedelta(1, 'd')
        self.previous_drought_factor[...] = self.current_drought_factor[...]

    @classmethod
    def load(cls, model, data):
        node = model.nodes[data.pop("node")]
        max_avg_delivery = load_parameter(model, data.pop("max_avg_delivery"))
        max_daily_delivery = load_parameter(model, data.pop("max_daily_delivery"))
        drought_factor = load_parameter(model, data.pop('drought_factor'))
        return cls(model, node, max_avg_delivery, max_daily_delivery, drought_factor, **data)