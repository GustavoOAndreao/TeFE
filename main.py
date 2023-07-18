from importlib import reload
import json
import random
from types import ModuleType
import simpy
#import config
#from config import *
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

    reload(config)  # need to reload the file
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
    reload(agents)  # we also need to reload the agents
    import agents

    config.SIM_TIME = config.SIM_TIME if _seed != 0 else 3

    # RUNNING

    config.env.run(until=config.SIM_TIME)

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
                    for _entry in list(agent.keys()):
                        if _entry in agent['strikables_dict']:  # and _entry != 'source':
                            agent[_entry] = agent[_entry][0]
                    del agent['strikables_dict']
                except:
                    None

                if agent['genre'] == 'EP':
                    # agent['portfolio_of_projects'] = str(agent['portfolio_of_projects'])
                    # agent['portfolio_of_plants'] = str(agent['portfolio_of_plants'])
                    del agent['portfolio_of_projects']
                    del agent['portfolio_of_plants']

    if _seed > 0:
        with open('analysis/json_MIX_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
            json.dump(config.MIX, fp, sort_keys=True, indent=4)

        with open('analysis/json_CONTRACTS_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
            json.dump(config.CONTRACTS, fp)

        with open('analysis/json_AGENTS_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
            json.dump(config.AGENTS, fp)

        with open('analysis/json_TECHNOLOGIC_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
            json.dump(config.TECHNOLOGIC, fp)

        # from commons import run_graphs
        # run_graphs(config.AGENTS, config.CONTRACTS, config.MIX, config.TECHNOLOGIC, save=True, name=_name + '_' + str(_seed),    show=False)

    printable = 'Simulation has thus ended at' + time.strftime("%H:%M:%S", time.localtime()) if _seed > 0 else None

    return printable


if __name__ == '__main__':

    RUN_TIMES = 100
    global_start = timeit.default_timer()
    from config import name

    """
    First simulation runs a little bizarre, so we always ignore it
    """
    for seed in range(0, RUN_TIMES):
        for type_o_run in [False, True]:
            random_run = type_o_run
            start = timeit.default_timer() if seed == 1 else None
            print('Simulation is starting') if seed > 0 and random_run is False else None
            print('Random run is starting') if seed > 0 and random_run is True else None

            run_sim(seed, name) if random_run is False else run_sim(seed, name, random_run=True)
            stop = timeit.default_timer() if seed == 1 else None

            printable_1 = 'SEED IS ' + str(seed) + " AND THIS IS RUN " + str(
                seed) + " OF " + str(RUN_TIMES) if seed > 0 and random_run is False else None
            print(printable_1) if random_run is False else None

            cond = seed == 1 and random_run is False
            printable_2 = " AND IT WILL TAKE ROUGHLY " + str(
                ((stop - start) / 60) * 2 * RUN_TIMES) + ' MINUTES' if cond is True else None
            print(printable_2) if cond is True else None

    global_stop = timeit.default_timer()
    print('IT TOOK ' + str(round((global_stop - global_start) / 60, 2)) + ' MINUTES TO RUN THE ' + str(RUN_TIMES) +
          ' SIMULATIONS (double it for the random runs and give it to the next person)')

    duration = 3000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
