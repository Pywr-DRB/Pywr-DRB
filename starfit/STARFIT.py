"""
Trevor Amestoy
Cornell University

A class object which simulates reservoir storage and releases, using STARFIT inferred operating
rule parameters.

Assumes:
1. Daily timestep
2. Simulation period begins in October
    (This can be changed using the STARFIT.start_month; options include only Oct or Jan at the moment.)

"""

import numpy as np
import pandas as pd
from math import pi, sin, cos

class Reservoir:
    def __init__(self, starfit_params,
                        reservoir_name,
                        inflow,
                        S_initial,
                        **kwargs):

        """
        Parameters:
        ----------
        starfit_df : DataFrame
            A dataframe containing all starfit data for reservoirs in the basin.
        reservoir_name : str
            The name of the reservoir to be simulated.
        inflow : array [1 x 365-days]
            An array containing a timeseries of inflow values into the reservoir;
            daily time-step.
        S_initial : float
            The initial storage in the reservoir.
        """

        self.params = starfit_params
        self.inflow = inflow
        self.S_initial = S_initial
        self.reservoir_name = kwargs.get('reservoir_name', 'Reservoir')
        self.timestep = kwargs.get('timestep', 'daily')
        self.start_month = kwargs.get('start_month', 'oct')
        return

    def parse_starfit_params(self):

        # Define reservoir constant characteristics daily
        self.R_max = ((self.params['Release_max'] + 1) * self.params['GRanD_MEANFLOW_MGD']).values
        self.R_min = ((self.params['Release_min'] + 1) * self.params['GRanD_MEANFLOW_MGD']).values
        self.I_bar = self.params['GRanD_MEANFLOW_MGD'].values
        self.S_cap = self.params['GRanD_CAP_MG'].values
        return

    def sinNpi(self, day, N):
        if self.start_month == 'Oct':
            return sin(N * pi * (day)/52)
        elif self.start_month == 'Jan':
            return sin(N * pi * (day+39)/52)

    def cosNpi(self, day, N):
        if self.start_month == 'Oct':
            return cos(N * pi * (day)/52)
        elif self.start_month == 'Jan':
            return cos(N * pi * (day+39)/52)

    # Define the average daily release function
    def release_harmonic(self, time):
        if self.timestep == 'daily':
            time = time/7
        R_avg_t = (self.params['Release_alpha1'] * self.sinNpi(time, 2) +
                 self.params['Release_alpha2'] * self.sinNpi(time, 4) +
                 self.params['Release_beta1'] * self.cosNpi(time, 2) +
                 self.params['Release_beta2'] * self.cosNpi(time, 4))
        return R_avg_t.values[0]

    # Calculate daily values of the upper NOR bound
    def calc_NOR_hi(self, time):
        # NOR harmonic is at weekly step
        if self.timestep == 'daily':
            time = time/7

        NOR_hi = (self.params['NORhi_mu'] + self.params['NORhi_alpha'] * self.sinNpi(time, 2) +
                     self.params['NORhi_beta'] * self.cosNpi(time, 2))

        if (NOR_hi < self.params['NORhi_min']).bool():
            NOR_hi = self.params['NORhi_min']
        elif (NOR_hi > self.params['NORhi_max']).bool():
            NOR_hi = self.params['NORhi_max']
        return (NOR_hi.values/100)


    # Calculate daily values of the lower NOR bound
    def calc_NOR_lo(self, time):
        # NOR harmonic is at weekly step
        if self.timestep == 'daily':
            time = time/7

        NOR_lo = (self.params['NORlo_mu'] + self.params['NORlo_alpha'] * self.sinNpi(time, 2) +
                     self.params['NORlo_beta'] * self.cosNpi(time, 2))

        if (NOR_lo < self.params['NORlo_min']).bool():
            NOR_lo = self.params['NORlo_min']
        elif (NOR_lo > self.params['NORlo_max']).bool():
            NOR_lo = self.params['NORlo_max']
        return (NOR_lo.values/100)

    # Standardize inflow using annual average
    def standardize_inflow(self, I_t):
        return (I_t - self.I_bar) / self.I_bar


    # Calculate storage as % of S_cap
    def percent_storage(self, S_t):
        return (S_t / self.S_cap)


    # Define the daily release adjustement function
    def release_adjustment(self, S_hat, time):
        A_t = (S_hat - self.calc_NOR_lo(time)) / (self.calc_NOR_hi(time) - self.calc_NOR_lo(time))
        I_hat = self.standardize_inflow(self.inflow[time])

        epsilon = (self.params['Release_c'] + self.params['Release_p1']*A_t +
                   self.params['Release_p2']*I_hat)
        return epsilon.values

    # Calculate the conditional target release volume
    def target_release(self, S_hat, I_t, time):
        NOR_hi = self.calc_NOR_hi(time)
        NOR_lo = self.calc_NOR_lo(time)

        if (S_hat <= NOR_hi) and (S_hat >= NOR_lo):
            target_R = min(self.I_bar * (self.release_harmonic(time) +
                                    self.release_adjustment(S_hat, time))
                           + self.I_bar, self.R_max)
        elif (S_hat > NOR_hi):
            target_R = min(self.S_cap * (S_hat - NOR_hi) + I_t, self.R_max)
        else:
            tR = (self.I_bar * (self.release_harmonic(time) + self.release_adjustment(S_hat, time)) + self.I_bar) * (1 - (NOR_lo - S_hat)/NOR_lo)
            target_R = max(tR, self.R_min)
        return target_R

    # Calculate actual release subject to mass constraints
    def actual_release(self, target_R, I_t, S_t):
        return max(min(target_R, (I_t + S_t)), (I_t + S_t - self.S_cap))


    def run_model(self):
        """
        Simulates storage and release at daily timestep.
        """

        # Collect data
        self.source_starfit_data()

        # Initialize matrices
        S = np.zeros_like(self.inflow)
        S_hat = np.zeros_like(S)
        R = np.zeros_like(self.inflow)

        # Set initial storage
        S[0] = self.S_initial
        S_hat[0] = self.percent_storage(S[0])

        # Simulate at daily step
        for d in range(len(self.inflow) - 1):

            I = self.inflow[d]
            S_hat[d] = self.percent_storage(S[d])
            target_R = self.target_release(S_hat[d], I, d)
            R[d] = self.actual_release(target_R, I, S[d])

            S[d + 1] = S[d] + I - R[d]

        out = {'storage' : S, 'outflow':R}
        results = pd.DataFrame(out)

        self.results = results
        print('Simulation complete!')
        return
