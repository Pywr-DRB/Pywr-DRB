import numpy as np
import pandas as pd
import urllib.request
import os

### get list of gages from WEAP
# ### West Branch Delaware River
# gages = ['01423000','01425000','01426500']
# ### Delaware River
# gages.extend(['01446500','01417500','01413500','01417000','01428500','01438500','01427207','01463500'])
# ### Beaver Kill
# gages.extend(['01420500'])
# ### Wallenpaupack Creek
# gages.extend(['01431270'])
# ### Lackawaxen River
# gages.extend(['01429000','01431500'])
# ### Mongaup River
# gages.extend(['01443350'])
# ### Neversink River
# gages.extend(['01436000','01435000','01437500'])
# ### Lehigh River
# gages.extend(['01454700','01453000','01447800','01449000'])
# ### Pohopoco Creek --- last one fails
# gages.extend(['01498000','01449360','01450000'])
# ### Musconetcong River
# gages.extend(['01457000'])
### Assunpink Creek
# gages = ['01464000','01463620']
# # gages.extend(['01464000','01463620'])
# ### Neshaminy Creek
# gages.extend(['01465645','01465500'])
# ### Rancocas River
# gages.extend(['01467000'])
# ### Schuylkill River
# gages.extend(['01474500','01470500','01472000','01473500'])
# ### Tulpehocken Creek
# gages.extend(['01471000','01470960','01470779'])
# ### Brandywine Creek
# gages.extend(['01481500'])


### get list of gages from pywr node-edge network spreadsheet (see drb_model_nodes.csv
# gages = ['0142400103','01423000','01415000','01414500','01414000','01413500','01427207','01428750','01432160','01432900',
#          '01435000','01438500','01449360','01447500','01447720','01457500','01463500','01470755','01470779','01475850',
#          '01480675','01482100','01425000','01417000','01429000','01432110','01431500','01433500','01436000','01449800',
#          '01447800','01455500','01459500','01463620','01470761','01469500','01470960','01473000','01480685']

### more gages needed for flow scaling (see explore_streamflows.ipynb
# gages = ['01423000', '01425000', '01413500', '01417000', '01435000', '01436000', '01428750', '01432110',  '01431500',
#          '01428750',  '01429000', '01428750', '01432495', '01428750', '01433500', '01449360', '01449800', '01447500', '01447800',
#          '01459500', '01459500', '01455500', '01455500', '01459500', '01459500', '01463620', '01463620', '01470755', '01470761',
#          '01469500', '01469500', '01470779', '01470960', '01472198', '+', '01472199', '01473000', '01480675', '01480685',
#          '01427207', '01438500', '01463500', '01464000', '01474500', '01481500', '01480015', '01479000', '01478000']
# ### remove dupulicates
# gages = list(set(gages))

### individuals
gages = ['01432110']

for gage in gages:
    try:
        start_date = '1980-01-01'
        end_date = '2021-12-31'
        url = f'https://nwis.waterdata.usgs.gov/usa/nwis/uv/?cb_00060=on&format=rdb&site_no={gage}&period=&begin_date={start_date}&end_date={end_date}'
        filename = f'input_data/usgs_gages/usgs_{gage}.txt'
        urllib.request.urlretrieve(url, filename)
    except:
        print('DOWNLOAD FAIL: GAGE ' + gage)

for gage in gages:
    try:
        filename = f'input_data/usgs_gages/usgs_{gage}.txt'
        with open(f'input_data/usgs_gages/usgs_{gage}.txt') as file:
            ### find line with gage ID and location
            lines = file.readlines()
            i = 0
            while i < 100:
                line = lines[i]
                if gage in line:
                    break
                i += 1
        ### rename file with name based on gage. abbreviate East/West/etc, Branch/Fork/River/Creek/Kill
        newname = line.lower().replace('#','').replace(',','').strip().replace(' ','_') + '.txt'
        for full, abbrev in [['eastern','e'], ['western','w'], ['northern','n'], ['southern', 's'],
                             ['east','e'], ['west','w'], ['north','n'], ['south', 's'],
                             ['branch','b'], ['fork','f'], ['river','r'], ['creek', 'c'], ['kill','k']]:
            newname = newname.replace(full, abbrev)
        os.replace(filename, 'input_data/usgs_gages/' + newname)
    except:
        print('CLEANING FAIL: GAGE ' + gage)