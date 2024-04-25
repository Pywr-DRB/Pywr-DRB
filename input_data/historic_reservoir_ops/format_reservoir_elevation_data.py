"""
This script is used to read USGS reservoir elevation data, 
for fewalter, beltzville, bluemarsh,
then concatenate and save as a single CSV file.

Elevation timeseries were obtained from the USGS NWIS website. 
The following sites were used:

beltzvilleCombined: 01449790
blueMarsh: 01470870
fewalter: 01447780

"""
import sys
import pandas as pd

sys.path.append('./')
from pywrdrb.utils.directories import input_dir

def read_usgs_elev_txt(file_path):
    res_name= file_path.split('_')[0]
    df = pd.read_csv(file_path, delimiter='\t', skiprows=31)
    df = df.drop(0, axis=0)
    df=df.rename(columns={'14n': res_name,
                          '20d': 'datetime'})
    
    df=df[['datetime', res_name]]
    df.index=pd.to_datetime(df['datetime'])
    df=df.drop('datetime', axis=1)
    return df

if __name__ == '__main__':
    # Load elevation data
    beltzville = read_usgs_elev_txt(f'{input_dir}historic_reservoir_ops/beltzvilleCombined_01449790_elevation.txt')
    blueMarsh = read_usgs_elev_txt(f'{input_dir}historic_reservoir_ops/blueMarsh_01470870_elevation.txt')
    fewalter = read_usgs_elev_txt(f'{input_dir}historic_reservoir_ops/fewalter_01447780_elevation.txt')

    # Concat and save as single CSV
    all_reservoirs= pd.concat([beltzville, blueMarsh, fewalter])
    all_reservoirs.to_csv(f'{input_dir}historic_reservoir_ops/lower_basin_reservoir_elevation.csv', sep=',')
    