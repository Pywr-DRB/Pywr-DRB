import json
import ast
import numpy as np
import pandas as pd

### first load baseline model
model_base_file = 'model_data/drb_model_base.json'
model_sheets_file = 'model_data/drb_model_sheets.xlsx'
model_full_file = 'model_data/drb_model_full.json'
model = json.load(open(model_base_file, 'r'))

### load nodes from spreadsheet & add elements as dict items
for sheet in ['nodes']:
    df = pd.read_excel(model_sheets_file, sheet_name=sheet)
    model[sheet] = []
    for i in range(df.shape[0]):
        nodedict = {}
        for j, col in enumerate(df.columns):
            val = df.iloc[i,j]
            if isinstance(val, int):
                nodedict[col] = val
            elif isinstance(val, str):
                ### if it's a list, need to convert from str to actual list
                if '[' in val:
                    val = ast.literal_eval(val)
                ### if it's "true" or "false", convert to bool
                if val in ["\"true\"", "\"false\""]:
                    val = {"\"true\"": True, "\"false\"": False}[val]
                nodedict[col] = val
            elif isinstance(val, float) or isinstance(val, np.float64):
                if not np.isnan(val):
                    nodedict[col] = val
            else:
                print(f'type not supported: {type(val)}, instance: {val}')
        model[sheet].append(nodedict)


### load edges from spreadsheet & add elements as dict items
for sheet in ['edges']:
    df = pd.read_excel(model_sheets_file, sheet_name=sheet)
    model[sheet] = []
    for i in range(df.shape[0]):
        edge = []
        for j, col in enumerate(df.columns):
            val = df.iloc[i,j]
            if isinstance(val, int):
                edge.append(val)
            elif isinstance(val, str):
                ### if it's a list, need to convert from str to actual list
                if '[' in val:
                    val = ast.literal_eval(val)
                ### if it's "true" or "false", convert to bool
                if val in ["\"true\"", "\"false\""]:
                    val = {"\"true\"": True, "\"false\"": False}[val]
                edge.append(val)
            elif isinstance(val, float) or isinstance(val, np.float64):
                if not np.isnan(val):
                    edge.append(val)
            else:
                print(f'type not supported: {type(val)}, instance: {val}')
        model[sheet].append(edge)


### load parameters from spreadsheet & add elements as dict items
for sheet in ['parameters']:
    df = pd.read_excel(model_sheets_file, sheet_name=sheet)
    model[sheet] = {}
    for i in range(df.shape[0]):
        name = df.iloc[i,0]
        model[sheet][name] = {}
        for j, col in enumerate(df.columns[1:], start=1):
            val = df.iloc[i,j]
            if isinstance(val, int):
                model[sheet][name][col] = val
            elif isinstance(val, str):
                ### if it's a list, need to convert from str to actual list
                if '[' in val:
                    val = ast.literal_eval(val)
                ### if it's "true" or "false", convert to bool
                if val in ["\"true\"", "\"false\""]:
                    val = {"\"true\"": True, "\"false\"": False}[val]
                model[sheet][name][col] = val
            elif isinstance(val, float) or isinstance(val, np.float64):
                if not np.isnan(val):
                    model[sheet][name][col] = val
            else:
                print(f'type not supported: {type(val)}, instance: {val}')


### save full model as json
with open(model_full_file, 'w') as o:
    json.dump(model, o, indent=4)

