# -*- coding: utf-8 -*-
"""
Trevor Amestoy

Cornell University
Spring 2022

Purpose:
    Contains the modified Fractional Gaussian Noise (mFGN) generator that is
    used to produce synthetic streamflow timeseries.

    Follows the synthetic streamflow generation as described by:

    Kirsch, B. R., Characklis, G. W., & Zeff, H. B. (2013). Evaluating the
    impact of alternative hydro-climate scenarios on transfer agreements:
    Practical improvement for generating synthetic streamflows. Journal of Water
    Resources Planning and Management, 139(4), 396-406.

    A detailed walk-trhough of the method is provided by Julie Quinn here:
    https://waterprogramming.wordpress.com/2017/08/29/open-source-streamflow-generator-part-i-synthetic-generation/

"""

# Core libraries
import numpy as np
import pandas as pd
import sys
import random
from scipy.linalg import cholesky, eig
from sklearn.preprocessing import StandardScaler

sys.path.append('..')

# Load custom functions of interest
from synthetic_generation.utils.transform import transform_daily_df_to_monthly_ndarray, transform_intraannual
from synthetic_generation.utils.math import cholskey_repair_and_decompose


class KirschNowakGenerator():
    def __init__(self, Q, **kwargs):
        """
        Follows the synthetic streamflow generation as described by:

        Kirsch, B. R., Characklis, G. W., & Zeff, H. B. (2013). Evaluating the
        impact of alternative hydro-climate scenarios on transfer agreements:
        Practical improvement for generating synthetic streamflows. Journal of Water
        Resources Planning and Management, 139(4), 396-406.

        A detailed walk-trhough of the method is provided by Julie Quinn here:
        https://waterprogramming.wordpress.com/2017/08/29/open-source-streamflow-generator-part-i-synthetic-generation/

        Parameters
        ----------
        historic_flow : matrix (N-years x T-timesteps)
            Historic flow data, arranged with yearly observations occupying the rows.
        n_years : int
            The number of synthetic years to produce.
        Standardized : bool
            Indication of whether the

        Returns
        -------
        if standardized:
            Matrices of 1,000 synthetic streamflow realizations ([1,0000, (N*T)]), Q_s, and standard streamflows, Z_s
        if not standardized:
            A matrix of 1,000 standard synthetic streamflow realizations ([1,0000, (N*T)]), Z_s

        """
        
        ## Assertions
        if not isinstance(Q, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        
        if not isinstance(Q.index, pd.DatetimeIndex):
            raise TypeError("Input index must be pd.DatetimeIndex.")
        
        if Q.index.freq != 'D':
            raise ValueError("Index frequency must be 'D' for daily")

        
        # Data transformations
        self.Q_hd = Q.to_numpy().transpose()
        self.Q_hm_3d = transform_daily_df_to_monthly_ndarray(Q)
        self.n_sites, self.n_days = self.Q_hd.shape
        self.n_months = self.Q_hm_3d.shape[2]
        self.n_years = self.Q_hm_3d.shape[1]


        # Handle **kwargs
        kwarg_keys = ('K',
                'historic_data_timestep',
                'sum_shifted_months',
                'modify_drought_frequency',
                'drought_frequency_probabilities',
                'drought_frequency_exceedances',
                'generate_using_log_flow',
                'matrix_repair_method')

        for kwarg in kwargs.keys():
            if kwarg not in kwarg_keys:
                raise Exception(kwarg + ' is not a valid key. Valid keys: ', kwarg_keys)

        self.K = kwargs.get('K', int((self.Q_hd.shape[1]/365)**0.5))  # Nearest neighbors for Nowak method
        self.hist_timestep = kwargs.get('historic_data_timestep', 'daily')
        self.sum_shifted_months = kwargs.get('sum_shifted_months', False)
        self.modify_drought_frequency = kwargs.get('modify_drought_frequency', False)
        self.drought_frequency_probabilities = kwargs.get('drought_frequency_probabilities', None)
        self.drought_frequency_exceedances = kwargs.get('drought_frequency_exceedances', None)
        self.generate_using_log_flow = kwargs.get('generate_using_log_flow', True)
        self.matrix_repair_method = kwargs.get('matrix_repair_method', 'spectral')
        
        # Constants
        self.DaysPerMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        self.MonthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',
                            'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        self.debugging = False
        self.print_status = False


    def kirsch_generator(self, return_timeseries = False):

        # Initialize
        Z_s = np.zeros((self.n_sites, self.n_years-1, self.n_months))  # Std. syn flow
        Q_s = np.zeros_like(Z_s)  # Syn. flow

        ## Bootstrapping ##
        # C is an uncorrelated randomly generated timeseries
        # M stores the bootstrap sampling for generating correlation later
        M = np.random.randint(low = 0,
                            high = self.n_years,
                            size = (self.n_years, self.n_months))

        # Generate synthetics
        for site in range(self.n_sites):

            # Scale flows according to month
            Z_h = np.zeros_like(self.Q_hm_3d[site,:,:])
            monthly_means = np.zeros(12)
            monthly_stds = np.zeros(12)

            if self.generate_using_log_flow:
                flow_to_scale = np.log(self.Q_hm_3d)
            else:
                flow_to_scale = self.Q_hm_3d.copy()

            for month in range(self.n_months):
                monthly_means[month] = np.mean(flow_to_scale[site, :, month])
                monthly_stds[month] =  np.std(flow_to_scale[site, :, month])
                Z_h[:,month] = (flow_to_scale[site, :, month] - monthly_means[month]) / monthly_stds[month]

            Z_h_prime = transform_intraannual(Z_h)

            C = np.zeros((self.n_years, self.n_months))
            for year in range(self.n_years):
                for month in range(self.n_months):
                    C[year, month] = Z_h[M[year, month], month]

            C_prime = transform_intraannual(C)

            # correlation matrix of Z_h (hsitoric log-std inflows), using columns!
            P_h = np.corrcoef(Z_h, rowvar = False)
            P_h_prime = np.corrcoef(Z_h_prime, rowvar = False)

            # Cholesky decomposition
            # Check if the corrlation matrix is positive definite; fix if not.
            U = cholskey_repair_and_decompose(P_h,
                                            max_iter = 100,
                                            method = self.matrix_repair_method,
                                            debugging = self.debugging).T

            U_prime = cholskey_repair_and_decompose(P_h_prime,
                                                    max_iter = 100,
                                                    method = self.matrix_repair_method,
                                                    debugging = self.debugging).T

            ## Impose correlation on X
            # Impose correlation on bootstrapped timeseries
            Z = C @ U
            Z_prime = C_prime @ U_prime

            Z_s[site, :, 0:6] = Z_prime[:, 6:12]
            Z_s[site, :, 6:12] = Z[1:, 6:12]

            # Convert back to normal flow
            if self.generate_using_log_flow:
                for month in range(self.n_months):
                    Q_s[site, :, month] = np.exp(Z_s[site, :, month] * monthly_stds[month] + monthly_means[month])
            else:
                for month in range(self.n_months):
                    Q_s[site, :, month] = Z_s[site, :, month] * monthly_stds[month] + monthly_means[month]

        self.Q_sm = Q_s
        self.Z_sm = Z_s
        if return_timeseries:
            return Q_s
        else:
            return

    def kirsch_ensemble_generator(self, return_ensemble = False):
        self.Q_sme = np.zeros((self.n_reals, self.n_sites, self.n_years-1, self.n_months))
        self.Z_sme = np.zeros_like(self.Q_sme)
        for r in range(self.n_reals):
            if self.print_status:
                print(f'Generating monthly data for realization {r+1} of {self.n_reals}.')
            self.Q_sme[r,:,:,:] = self.kirsch_generator(return_timeseries= True)
            self.Z_sme[r,:,:,:] = self.Z_sm
        if return_ensemble:
            return self.Q_sme
        else:
            return

    def calculate_shifted_monthly_totals(self):
        # Make an expanded dataset with extensions on either side +/- 7 days
        #expanded_Q_h = np.hstack((self.Q_h[:,self.n_days-7:], self.Q_h, self.Q_h[:, 0:8]))
        # Initalize monthly total storage
        # n_total_windows = (n_years * 15) + (n_years * 15 - 14)
        #for month in range(12):
        #    count = 1
        #    if month == 0 or month == 11:
        #        n_windows = n_years * 15 - 7
        #    else:
        #        n_windows = n_years*15

        #    Q_hm_shifted = np.zeros((self.n_sites, n_windows))
        #    indices = np.zeros((n_windows, 2))
            # Shift the timeseries 1 day at a time
            # Calculate monthly totals of shifted data
            #for shift in range(15):
            #    shifted_Q_h_daily = expanded_Q_h[:, shift:shift+self.n_days]
            #    shifted_Q_h_monthly = convert_daily_to_monthly(shifted_Q_h_daily)

            #    self.Q_h_monthly_totals= np.hstack((self.Q_h_monthly_totals, shifted_Q_h_monthly[site, :, month:month+1]))
            #    self.Q_h_monthly_indices= np.hstack((self.Q_h_monthly_indices, WHAT))

            #    for site in range(self.n_sites):

            #        self.Q_h_monthly_totals[site, month, :] =
            #        self.Q_h_monthly_indices[site, month, :] = month*np.ones(n)
                #for site in range(self.n_sites):
                #    if month == 0 and shift < 8:
                #        Qh[site, :] = Qh[site, 12:] # remove first year
                #
                #    elif month ==11 and shift > 8:
                #        Qh[site, :] = Qh[site, 0:-12] # remove last year
                #if month == 0 and shift < 8:
                #    indices[count:(count + Qh.shape[1]), 0] = np.arange(1:Qh.shape[1]+1)
                #else:
                #    indices[count:(count + Qh.shape[1]), 0] = np.arange(0:Qh.shape[1])
                # Q_hm_shifted[s, count:(count+n_years)] = Qh[s,]

        self.Q_h_monthly_totals = None
        self.Q_h_monthly_indices = None
        return


    def find_KNN(self, realization, syn_year, month):
        # Calculate difference between historic and synthetic month totals

        # Sum flows across sites
        monthly_hist_sums = np.sum(self.Q_hm_3d[:, :, month], axis = 0)

        monthly_syn_sum = np.sum(self.Q_sme[realization, :, syn_year, month])

        # Find nearest K
        delta = (monthly_hist_sums - monthly_syn_sum)**2
        KNN_year_month_id = np.argsort(delta)[0:self.K]

        # Weights
        w = (1/np.arange(1, self.K+1) / np.sum(1/np.arange(1,self.K+1)))

        return KNN_year_month_id, w


    def sample_KNN(self, KNN_year_month_id, month):
        r = np.random.rand()
        # Append a zero
        cummulative_w = np.hstack((np.array([0]), self.cummulative_wts.flatten()))

        # Probabilitically sample the KNNs
        for i in range(len(self.cummulative_wts)):
            if (r > cummulative_w[i]) and (r <= cummulative_w[i+1]):
                use_knn = i
                break
        hist_year_id = KNN_year_month_id[use_knn]

        # Extract daily flows from record period of interest
        start = 365*(hist_year_id) + sum(self.DaysPerMonth[0:month])
        end = start + self.DaysPerMonth[month]
        daily_flows = self.Q_hd[:, start:end]

        # Calculate proportions of monthly totals
        self.py = np.zeros_like(daily_flows)
        for site in range(self.n_sites):
            self.py[site, :] = daily_flows[site,:] / np.sum(daily_flows[site, :])

        return hist_year_id


    def nowak_disaggregation(self, return_ensemble = False):

        self.Q_sde = np.zeros((self.n_reals, self.n_sites, self.n_years-1, 365))

        # Find historic monthly totals
        if self.sum_shifted_months:
            self.calculate_shifted_monthly_totals()

        # Find and sample from KNN
        for real in range(self.n_reals):
            if self.print_status:
                print(f'Disaggregating data for realization {real+1} of {self.n_reals}.')
            for year in range(self.n_years - 1):  # The synthetic series are 1 year shorter
                for month in range(self.n_months):

                    start_day = sum(self.DaysPerMonth[0:month])
                    end_day = start_day + self.DaysPerMonth[month]

                    # Find month in historic record with nearest monthly total
                    knn_year_id, wts = self.find_KNN(real, year, month)
                    self.knn_year_id = knn_year_id
                    self.wts = wts
                    self.cummulative_wts = np.cumsum(wts)

                    # Sample 1 KNN from set, and calcualte flow proportions
                    year_id = self.sample_KNN(knn_year_id, month)

                    for site in range(self.n_sites):
                        self.Q_sde[real, site, year, start_day:end_day] = self.Q_sme[real, site, year, month] * self.py[site,:]
        if return_ensemble:
            return self.Q_sde
        else:
            return

    def generate_ensemble(self, N, return_monthly_data = False):
        """
        Runs the combined Kirsch-Nowak generation method.
        Returns an enesmble of daily synthetic flows at all sites.
        """
        self.n_reals = N

        self.kirsch_ensemble_generator()
        self.nowak_disaggregation()

        if return_monthly_data:
            return self.Q_sde, self.Q_sme
        else:
            return self.Q_sde