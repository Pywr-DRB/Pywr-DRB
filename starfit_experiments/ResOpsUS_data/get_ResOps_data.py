"""
Trevor Amestoy

Pulls reservoir data from the ResOpsUS database, using IDs in the
DRB reservoir info spreadsheet.
"""

import numpy as np
import pandas as pd



### Load STARFIT conus data for reservoirs in DRB
starfit = pd.read_csv('../../model_data/drb_model_istarf_conus.csv')
reservoirs = [res for res in starfit['reservoir']]
reservoir_ids = [id for id in starfit['GRanD_ID']]
reservoir_names = [name for name in starfit['GRanD_NAME']]


# load csv containing all ResOpsUS timeseries
resops_in = pd.read_csv('DAILY_AV_INFLOW_CUMECS.csv', delimiter = ',', infer_datetime_format = True, low_memory = False)
resops_out = pd.read_csv('DAILY_AV_OUTFLOW_CUMECS.csv', delimiter = ',', infer_datetime_format = True, low_memory = False)
resops_storage = pd.read_csv('DAILY_AV_STORAGE_MCM.csv', delimiter = ',', infer_datetime_format = True, low_memory = False)


# Store reservoirs not found
not_found = []

for res, id in zip(reservoirs, reservoir_ids):

    # Skip if no id available
    if type(id) != str:
        print(f'Improper ID format for {res}')

    # Check if more than one reservoir included
    elif len(id) > 5:

        many_ids = id.split()
        combined_data = pd.DataFrame({'Date':resops_in['date'], 'Inflow':np.zeros(len(resops_in['date'])), 'Outflow':np.zeros(len(resops_in['date'])), 'Storage':np.zeros(len(resops_in['date']))})

        # Loop through composite reservoirs
        for sub_id, counter in zip(many_ids, range(len(many_ids))):

            # Note if not found in database
            if sub_id not in resops_storage.columns:
                print(f'{res} with ID {sub_id} was not found in the ResOps database.')
                not_found.append(res)

            else:
                # Remove spaces
                sub_id = sub_id.strip()

                sub_data = pd.DataFrame({'Inflow':resops_in[sub_id].values, 'Outflow':resops_out[sub_id].values, 'Storage':resops_storage[sub_id].values})

                combined_data = pd.concat([combined_data, sub_data], axis = 1)

                # Only save if all sub-reservoirs have been addes
                if (counter + 1) == len(many_ids):
                    # Sum columns
                    combined_data = combined_data.groupby(combined_data.columns, axis=1).sum()
                    combined_data.to_csv(f'resops_{res}.csv', sep = ',', index = False)

    # Note if not found in database
    if id not in resops_storage.columns:
        print(f'{res} with ID {id} was not found in the ResOps database.')
        not_found.append(res)


    else:
        data = pd.DataFrame({'Date':resops_in['date'], 'Inflow':resops_in[id].values, 'Outflow':resops_out[id].values, 'Storage':resops_storage[id].values})

        #bluemarsh.reset_index(drop=True, inplace=True)
        #data.dropna(subset = 'Inflow', inplace = True)
        data.to_csv(f'resops_{res}.csv', sep = ',', index = False)
