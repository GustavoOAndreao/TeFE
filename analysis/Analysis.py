import pandas as pd
import numpy as np
import os, json

def search_json_for(_json, __csv=None):

    print(_json)

    _csv = pd.DataFrame() if __csv is None else __csv

    for period in _json:
        entries = _json[period]
        for entry_ in entries:
            _entry = entries[entry_]

            row = {'period': period}

            for i in _entry:
                row.update({i: _entry[i]})

            row = pd.DataFrame(row, index=[0])

            #_csv = _csv.append(row, ignore_index=True)

            _csv = pd.concat([_csv.loc[:], row]).reset_index(drop=True)

    return _csv

if __name__ == '__main__':

    path_to_json = 'analysis/..'
    json_files = {'mix': [pos_json for pos_json in os.listdir(path_to_json) if (pos_json.endswith('.json') and 'MIX' in pos_json)],
                  'contracts': [pos_json for pos_json in os.listdir(path_to_json) if (pos_json.endswith('.json') and 'CONTRACTS' in pos_json)],
                  'agents': [pos_json for pos_json in os.listdir(path_to_json) if (pos_json.endswith('.json') and 'AGENTS' in pos_json)],
                  'technologic': [pos_json for pos_json in os.listdir(path_to_json) if (pos_json.endswith('.json') and 'TECHNOLOGIC' in pos_json)]}

    # json_file = open(json_files['mix'][0])

    # print(json_file)
    #global df
    df = pd.DataFrame()
    print(json_files['mix'])
    for entry in range(0,len(json_files['mix'])-1):

        json_file = open(json_files['mix'][entry])
        data = search_json_for(json.load(json_file))

        df = pd.concat([df, data], ignore_index=True, sort=False)

    print(df)




