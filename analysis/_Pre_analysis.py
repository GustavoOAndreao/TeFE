import traceback

import pandas as pd
import feather
import numpy as np
import os
import json
import timeit
import pickle
import time
# import config
import pyarrow
import random
import winsound

from flatten_json import flatten
from itertools import zip_longest
from pyspark.sql import SparkSession
from itertools import chain, starmap
from pandas.compat import pyarrow


def flatten_json_iterative_solution__(dictionary, period, seed, max_level=2):
    """Flatten a nested json file"""

    list_o_dictionaries = []

    #level = 0

    def unpack(parent_key, parent_value):
        nonlocal level
        """Unpack one level of nesting in json file"""
        # Unpack one level only!!!

        # print(parent_key, parent_value)

        if level < max_level+1:  # or parent_key.isnumeric() is False:
            if isinstance(parent_value, dict):
                for key, value in parent_value.items():
                    temp1 = parent_key + '_' + key if level > 1 else key
                    yield temp1, value
            elif isinstance(parent_value, list):
                # print(parent_key, parent_value)
                i = 0
                for value in parent_value:
                    temp2 = parent_key # + '_' + str(i) if level > 1 else str(i)
                    i += 1
                    yield temp2, value
            else:
                yield parent_key, parent_value
        else:
            yield parent_key, str(parent_value)[:5]

            # Keep iterating until the termination condition is satisfied

    entries = list(dictionary.keys())

    for entry in entries:
        level = 0

        """if 'portfolio_of_projects' in dictionary[entry]:
            del dictionary[entry]['portfolio_of_projects']
            del dictionary[entry]['portfolio_of_plants']"""
        # print(dictionary[entry])



        while True:
            # Keep unpacking the json file until all values are atomic elements (not dictionary or list)
            _dictionary = dict(chain.from_iterable(starmap(unpack, dictionary[entry].items())))
            list_o_dictionaries.append(_dictionary)
            _dictionary['entry'] = entry
            _dictionary['period'] = period
            _dictionary['seed'] = seed
            # Terminate condition: not any value in the json file is dictionary or list
            if not any(isinstance(value, dict) for value in _dictionary.values()) and \
                    not any(isinstance(value, list) for value in _dictionary.values()):
                break
            level += 1

    return list_o_dictionaries


def search_json_for(_json, _seed, __csv=None, print_=False):
    """

    :param _seed:
    :param _json: Json file that we will mess with
    :param __csv: csv to concat, if None then we create one (leave it as None dude)
    :param print_: If you want to print what's
    :return:
    """
    print(_json) if print_ is True else None

    # _csv = pd.DataFrame() if __csv is None else __csv

    df_list = []
    for period in _json:
        entries = _json[period]
        if len(entries) > 0:
            _json_dict = flatten_json_iterative_solution__(entries, period, _seed)  # flatten_json_iterative_solution__(entries)
            # _json_dict['period'] = period
            df_list = df_list + _json_dict
    # _json_dict = pd.json_normalize(df_list)  # .transpose()
    # _json_dict['period'] = period
    # _csv = pd.concat([_csv.loc[:], _json_dict]).reset_index(drop=True)

    """
                    for entry_ in entries:
                _entry = entries[entry_]
                _json_dict = flatten_json_iterative_solution__(_entry)  # flatten_json_iterative_solution__(entries)
                _json_dict['period'] = period
                _json_dict = pd.DataFrame.from_dict(_json_dict, orient='index').transpose()
                _csv = pd.concat([_csv.loc[:], _json_dict]).reset_index(drop=True)
        """

    """     for entry_ in entries:

            _entry = entries[entry_]

            row = dict(_entry)

            row['period'] = period

            
            
            row = {'period': period}
            
            for i in _entry:
                if i != 'portfolio_of_plants' or 'portfolio_of_projects':
                    row.update({i: _entry[i]})
                else:
                    row.update({i: len(str(_entry[i]))})
            row = pd.DataFrame.from_dict(row, orient='index').transpose()

            # _csv = _csv.append(row, ignore_index=True)

            _csv = pd.concat([_csv.loc[:], row]).reset_index(drop=True)

           
            
                        row = {'period': period}

            for i in _entry:
                if i != 'portfolio_of_plants' or 'portfolio_of_projects':
                    row.update({i: _entry[i]})
                else:
                    row.update({i: len(str(_entry[i]))})
            row = pd.DataFrame.from_dict(row, orient='index').transpose()

            # _csv = _csv.append(row, ignore_index=True)

            _csv = pd.concat([_csv.loc[:], row]).reset_index(drop=True)
            
            previous solution. Current one is faster
            row = pd.DataFrame(row, index=[0])

            # _csv = _csv.append(row, ignore_index=True)

            _csv = pd.concat([_csv.loc[:], row]).reset_index(drop=True)
            """

    return df_list


def concat_csvs(json_files, file_type, name, _path, _format, big_files=None):
    # df = pd.DataFrame()
    df_big_list = []

    to_search = json_files[file_type]
    to_search = to_search if big_files is None else random.sample(to_search, big_files)
    start = timeit.default_timer()
    for entry in range(0, len(to_search)):
        print(str(entry+1) + ' of ' + str(len(to_search)) + ' is starting at ' + time.strftime("%H:%M:%S", time.localtime()))
        global path_to_json
        try:
            json_file = open(to_search[entry])
            """d = json.load(json_file)
            dd = pd.DataFrame.from_dict(d, orient='index')"""
            data = search_json_for(json.load(json_file), entry)
            json_file.close()
            # inter = timeit.default_timer()
            # print(str(entry) + " is done after " + str(((inter - start) / 60)) + ' minutes')
        except:
            traceback.print_exc()

        df_big_list += data

        # data['seed_name'] = str(entry)
        # df = pd.concat([df, data], ignore_index=True, sort=False)

        # print(data.shape[0])

    print('Done with them all, now to transform the list o list and save it in some format')

    df = pd.json_normalize(df_big_list) if _format != 'pickle' else df_big_list
    # _csv = pd.concat([_csv.loc[:], _json_dict]).reset_index(drop=True)

    name = _path + '__' + name

    if _format == 'csv':
        df.to_csv(name + '.csv', index=False)
    elif _format == 'feather':
        df.to_feather(name + '.feather', index=False)
    elif _format == 'pickle':
        picklefile = name + ".pkl"
        with open(picklefile, "wb") as pkl_wb_obj:
            pickle.dump(df, pkl_wb_obj)  # it's not really a df, it's a list, we'll use it only for the goddamn agents,
            # they're so damn big
    elif _format == 'parquet':
        from fastparquet import write
        write(name + '.parquet', df)
        # df.to_parquet(name + '.parquet', write_index=False)
    else:
        print("you forget to choose a format, my ganzirosis, but i'll save it a parquet just in case, ok?")

    # df.to_csv(name + '.csv', index=False) if _csv is True else df.to_pickle(name + '.pkl')  # else df.to_feather(name)
    stop = timeit.default_timer()

    return print(name + " is done after " + str(((stop - start)/60)) + ' minutes')


if __name__ == '__main__':

    list_of_dirs = next(os.walk('.'))[1]
    list_of_dirs.remove('Figures')
    try:
        list_of_dirs.remove('__pycache__')
    except:
        None

    # list_of_dirs = ["1_YES_YES_YES"]  # ['0_YES_YES']  # uncomment and put the directory name to override checking all directories

    for path in list_of_dirs:  # ['ALL_NO_NO', 'ALL_NO_YES', 'ALL_YES_NO', 'ALL_YES_YES']:
        path_to_json = 'analysis/..'
        path_to_json = path_to_json + '/' + path
        # config.name = 'YES_YES' # If you need to override the name uncomment
        just_normal = True
        json_files = {'mix': [path_to_json + '/' + pos_json for pos_json in os.listdir(path_to_json) if
                              (pos_json.endswith('.json') and 'MIX' in pos_json in pos_json and
                               'RANDOM_RUN' not in pos_json)],
                      'contracts': [path_to_json + '/' + pos_json for pos_json in os.listdir(path_to_json) if
                                    (pos_json.endswith('.json') and 'CONTRACTS' in pos_json in pos_json and
                                     'RANDOM_RUN' not in pos_json)],
                      'agents': [path_to_json + '/' + pos_json for pos_json in os.listdir(path_to_json) if
                                 (pos_json.endswith('.json') and 'AGENTS' in pos_json in pos_json and
                                  'RANDOM_RUN' not in pos_json)],
                      'technologic': [path_to_json + '/' + pos_json for pos_json in os.listdir(path_to_json) if
                                      (pos_json.endswith('.json') and 'TECHNOLOGIC' in pos_json in pos_json
                                       and 'RANDOM_RUN' not in pos_json)]}
        if just_normal is False:
            random_json_files = {'mix': [pos_json for pos_json in os.listdir(path_to_json) if
                                         (pos_json.endswith('.json') and 'MIX' in pos_json and
                                          'RANDOM_RUN' in pos_json)],
                                 'contracts': [pos_json for pos_json in os.listdir(path_to_json) if
                                               (pos_json.endswith(
                                                   '.json') and 'CONTRACTS' in pos_json and
                                                'RANDOM_RUN' in pos_json)],
                                 'agents': [pos_json for pos_json in os.listdir(path_to_json) if
                                            (pos_json.endswith('.json') and 'AGENTS' in pos_json and
                                             'RANDOM_RUN' in pos_json)],
                                 'technologic': [pos_json for pos_json in os.listdir(path_to_json) if
                                                 (pos_json.endswith(
                                                     '.json') and 'TECHNOLOGIC' in pos_json
                                                  and 'RANDOM_RUN' in pos_json)]}

        # json_file = open(json_files['mix'][0])

        # print(json_file)
        # global df

        _format = 'csv'  # Choose from 'pickle', 'csv', 'parquet' or 'feather

        runs = ['random', 'normal'] if just_normal is False else ['normal']

        for type_o_run in runs:
            files = json_files if type_o_run == 'normal' else random_json_files
            for type_o_dict in ['technologic', 'agents']:  # 'mix', 'contracts']:
                try:
                    print('STARTING ' + path.upper() + ' ' + type_o_run.upper() + " " + type_o_dict.upper() + ' FILES')
                    concat_csvs(files, type_o_dict, type_o_run + "_" + type_o_dict , path, _format)
                except:
                    print("Couldn't do it my ganzirosis, try again for the", type_of_dict)
            for type_o_dict in ['mix', 'contracts']:
                try:
                    print('STARTING ' + path.upper() + ' ' + type_o_run.upper() + " " + type_o_dict.upper() + ' FILES')
                    concat_csvs(files, type_o_dict, type_o_run + "_" + type_o_dict, path, _format)  # , big_files=True)
                except:
                    print("Couldn't do it my ganzirosis, try again for the", type_of_dict)

    duration = 1500  # milliseconds
    freq = 660  # Hz
    winsound.Beep(freq, duration)

    #merge_json_files(random_json_files['agents'])

    """dfs = []
    i=0
    for file in random_json_files['agents'][0:10]:
        print(i)
        with open(file) as f:
            json_data = pd.json_normalize(json.loads(f.read()))
            json_data['site'] = file.rsplit("/", 1)[-1]
        dfs.append(json_data)
        i+=1
    df = pd.concat(dfs)"""


    """
    BENCHMARKS (15 runs):
    
        PICKLE:
            
            agents: 19 mb
            mix: 448 mb 
            tech: 1,64 mb
            contracts: 430 mb
            
        CSV:
            
            agents: 17 mb
            mix: 339 mb
            tech: 1 mb
            contracts: 346
            
        PARQUET (had some troubles...):
            
            agents
            mix: 
            tech: 
            contracts:
            
        FEATHER: Couldn't do it, too big...
        
            agents
            mix: 
            tech: 
            contracts:
        
    """

    """print('STARTING MIX FILES')
    concat_csvs(json_files, 'mix', 'mix_test' + config.name, _format)
    concat_csvs(random_json_files, 'mix', 'random_mix_test' + config.name, _format)
    print('STARTING CONTRACTS FILES')
    concat_csvs(json_files, 'contracts', 'contracts_test' + config.name, _format)
    concat_csvs(random_json_files, 'contracts', 'random_contracts_test' + config.name, _format)
    print('STARTING TECHNOLOGIC FILES')
    concat_csvs(json_files, 'technologic', 'technologic_test' + config.name, _format)
    concat_csvs(random_json_files, 'technologic', 'random_technologic_test' + config.name, _format)
    print('STARTING AGENTS FILES')
    concat_csvs(json_files, 'agents', 'agents_test' + config.name, _format)
    concat_csvs(random_json_files, 'agents', 'random_agents_test' + config.name, _format)"""
