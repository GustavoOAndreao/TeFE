from importlib import reload
import json
import random
from types import ModuleType
import simpy
# import config
# from config import *
import subprocess, os, sys
import timeit
from matplotlib import pyplot as plt
import time
from icecream import ic
import numpy as np
import winsound

# import clear_cache
# from clear_cache import clear as clear_cache
# from IPython.lib.deepreload import reload as rld


def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)


def run_sim(_seed, _name='test', random_run=False):
    import config
    import simpy

    reload(config) if _seed > start_seed else None  # need to reload the file
    # import config

    config.env = simpy.Environment(initial_time=0)
    config.seed = _seed
    if random_run is True:
        config.INITIAL_RANDOMNESS = 1
        _name = 'RANDOM_RUN' + _name
    else:
        config.INITIAL_RANDOMNESS = config.INITIAL_RANDOMNESS

    config.STARTING_TIME = 0
    import agents
    reload(agents) if _seed > start_seed else None  # we also need to reload the agents
    import agents

    simulation_time = config.SIM_TIME if _seed != start_seed else 3

    # RUNNING

    config.env.run(until=simulation_time)

    if _seed > start_seed:

        #########

        # Removing the fuss periods from the dictionaries

        # Don't uncomment the following lines: it resets the dictionaries to the zeroth period, but creates some
        # problems due to this...

        for _dict in [config.MIX, config.CONTRACTS, config.AGENTS, config.TECHNOLOGIC]:
            for line in range(0, config.FUSS_PERIOD):
                del _dict[line]

                # Don't uncomment the following lines: it resets the dictionaries to the zeroth period, but creates some
                # problems due to this...
        """for line in range(config.FUSS_PERIOD, config.SIM_TIME):
                change_dict_key(_dict, line, line-config.FUSS_PERIOD)"""

        for period in config.AGENTS:
            for entry in list(config.AGENTS[period].keys()):
                agent = config.AGENTS[period][entry]

                if agent['genre'] != 'DD':
                    try:
                        # print('before', agent['genre'], agent)  # if agent['genre'] not in ['DBB', 'EPM'] else None
                        if agent['genre'] == 'EP':
                            del agent['portfolio_of_projects']
                            del agent['portfolio_of_plants']

                        for _entry in list(agent.keys()):
                            """print(agent['name'], _entry, list(agent.keys()), agent['genre'], config.ENTRIES_TO_SAVE[agent['genre']],
                                  _entry in config.ENTRIES_TO_SAVE[agent['genre']])"""
                            # print('keys', list(agent.keys()))
                            if _entry in agent['strikables_dict'] and _entry in config.ENTRIES_TO_SAVE[agent['genre']]:  # and _entry != 'source':
                                agent[_entry] = agent[_entry][0]
                            elif _entry not in config.ENTRIES_TO_SAVE[agent['genre']]:
                                # print(config.ENTRIES_TO_SAVE[agent['genre']])
                                del agent[_entry]

                        del agent['strikables_dict']

                    except:
                        None
                    # print('after the', agent)
                    """if agent['genre'] == 'EP':
                        # agent['portfolio_of_projects'] = str(agent['portfolio_of_projects'])
                        # agent['portfolio_of_plants'] = str(agent['portfolio_of_plants'])
                        del agent['portfolio_of_projects']
                        del agent['portfolio_of_plants']"""

                else:
                    for _entry in list(agent.keys()):
                        if _entry not in config.ENTRIES_TO_SAVE['DD']:
                            del agent[_entry]

            if len(config.MIX[period]) > 0:
                for entry in list(config.MIX[period].keys()):
                    plant = config.MIX[period][entry]
                    # ic(entry, list(plant.keys()))
                    # print('before the', plant)
                    for _entry in list(plant.keys()):
                        # ic(_entry, plant[_entry])
                        if _entry not in config.ENTRIES_TO_SAVE['MIX']:
                            # print('deleted', plant[_entry])
                            del plant[_entry]
                    # print('after the', plant)

        if 'RANDOM_RUN' not in _name:
            _path = 'analysis/' + _name.replace('____', '')
        else:
            __name = _name.replace('RANDOM_RUN', '')
            _path = 'analysis/' + __name.replace('____', '')

        if os.path.exists(_path) is False:
            os.makedirs(_path)

        if 'MIX' in config.ENTRIES_TO_SAVE['DICTS']:
            with open(_path +'/MIX_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
                json.dump(config.MIX, fp, sort_keys=True, indent=4)

        if 'CONTRACTS' in config.ENTRIES_TO_SAVE['DICTS']:
            with open(_path + '/CONTRACTS_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
                json.dump(config.CONTRACTS, fp)

        if 'AGENTS' in config.ENTRIES_TO_SAVE['DICTS']:
            with open(_path + '/AGENTS_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
                json.dump(config.AGENTS, fp)

        if 'TECHNOLOGIC' in config.ENTRIES_TO_SAVE['DICTS']:
            with open(_path + '/TECHNOLOGIC_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
                json.dump(config.TECHNOLOGIC, fp)

        # from commons import run_graphs
        # run_graphs(config.AGENTS, config.CONTRACTS, config.MIX, config.TECHNOLOGIC, save=True, name=_name + '_' + str(_seed),    show=False)

    printable = 'Simulation has thus ended at' + time.strftime("%H:%M:%S", time.localtime()) if _seed > start_seed else None

    return printable


if __name__ == '__main__':

    RUN_TIMES = 5  # 100
    global_start = timeit.default_timer()
    from config import name
    print('WE SHALL START THE ' + name + ' RUNS')

    """
    First simulation runs a little bizarre, so we always ignore it
    """

    list_o_runs = [False]  # , True]  # both runs  # True is random, False is normal

    start_seed = 0

    for seed in range(start_seed, RUN_TIMES + 1):
        for type_o_run in list_o_runs:
            random_run = type_o_run
            start = timeit.default_timer() if seed == start_seed + 1 else None
            print('Simulation is starting') if seed > start_seed and random_run is False else None
            print('Random run is starting') if seed > start_seed and random_run is True else None

            run_sim(seed, name) if random_run is False else run_sim(seed, name, random_run=True)
            if seed == start_seed:
                break
            stop = timeit.default_timer() if seed == start_seed + 1 else None

            printable_1 = 'SEED IS ' + str(seed) + " AND THIS IS RUN " + str(
                seed) + " OF " + str(RUN_TIMES) if seed > start_seed and random_run is False else None
            print(printable_1) if random_run is False else None

            cond = seed == start_seed + 1 and random_run is False
            printable_2 = " AND IT WILL TAKE ROUGHLY " + str(
                ((stop - start) / 60) * 2 * RUN_TIMES) + ' MINUTES (' + str(stop - start) + ' seconds for each run)' if cond is True else None
            print(printable_2) if cond is True else None

    global_stop = timeit.default_timer()
    print('IT TOOK ' + str(round((global_stop - global_start) / 60, 2)) + ' MINUTES TO RUN THE ' + str(RUN_TIMES) + " " + name +
          ' SIMULATIONS (double it for the random runs and give it to the next person)')

    duration = 3000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
