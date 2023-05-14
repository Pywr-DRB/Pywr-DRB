from pywr.model import Model
from pywr.recorders import TablesRecorder



### Load the model
model = Model.load('test_delay_json.json')

### Add a storage recorder
TablesRecorder(model, 'test_delay_out.hdf5', parameters=[p for p in model.parameters if p.name])

### Run the model
stats = model.run()
stats_df = stats.to_dataframe()

import h5py
f = h5py.File('test_delay_out.hdf5')
print('release')
print(f['release'][...])
print('mrf1')
print(f['mrf1'][...])
print('delay')
print(f['delay'][...])
print('mrf2')
print(f['mrf2'][...])
f.close()