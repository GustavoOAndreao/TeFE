import importlib
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
# from streamlit import caching
import functools
import types
import builtins
from importlib import reload
import reloader
reloader.enable()
# from IPython.lib import deepreload
# builtins.reload = deepreload.reload

# import clear_cache
# from clear_cache import clear as clear_cache
# from IPython.lib.deepreload import reload as rld

# config_name = None


def reload_package(package):
    assert(hasattr(package, "__package__"))
    fn = package.__file__
    fn_dir = os.path.dirname(fn) + os.sep
    module_visit = {fn}
    del fn

    def reload_recursive_ex(module):
        importlib.reload(module)

        for module_child in vars(module).values():
            if isinstance(module_child, types.ModuleType):
                fn_child = getattr(module_child, "__file__", None)
                if (fn_child is not None) and fn_child.startswith(fn_dir):
                    if fn_child not in module_visit:
                        # print("reloading:", fn_child, "from", module)
                        module_visit.add(fn_child)
                        reload_recursive_ex(module_child)

    return reload_recursive_ex(package)


def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)


def run_sim(_seed, _name='test', random_run=False):
    # config = importlib.import_module('_config____1_YES_YES')
    # config = __import__('_config____1_YES_YES')
    # caching.clear_cache()

    # del sys.modules['config']

    importlib.invalidate_caches()
    import simpy
    # print(vars(config)['config_name'])

    global start_seed


    # reload(__import__('_config____1_YES_YES')) if _seed > start_seed else None  # need to reload the file
    # importlib.invalidate_caches()
    reload(config) if _seed > start_seed else None  # need to reload the file
    reload(sys.modules[__init__.config_name[0]]) if _seed > start_seed else None
    sys.modules[__init__.config_name[0]] = config

    # print('run', __init__.config_name[0], config.config_name)
    # import config
    #
    # config = importlib.import_module('config')

    config.env = simpy.Environment(initial_time=0)
    config.seed = _seed
    if random_run is True:
        config.INITIAL_RANDOMNESS = 1
        _name = 'RANDOM_RUN' + _name
    else:
        config.INITIAL_RANDOMNESS = config.INITIAL_RANDOMNESS

    config.STARTING_TIME = 0

    # agents = importlib.import_module('_agents____1_YES_YES')
    # agents = __import__('_agents____1_YES_YES')
    import agents

    reload(agents) if _seed > start_seed else None  # we also need to reload the agents
    config.agents_name = __init__.agents_name[0]
    # reload(__import__('_agents____1_YES_YES')) if _seed > start_seed else None  # we also need to reload the agents
    # agents = importlib.import_module('_agents____1_YES_YES')
    # print(vars(config))
    sys.modules[__init__.agents_name[0]] = agents

    simulation_time = config.SIM_TIME if _seed != start_seed else 3  # config.SIM_TIME if _seed != start_seed else 3

    # RUNNING

    config.env.run(until=simulation_time)
    # print(_seed,  start_seed)
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
                    # print('before', agent['genre'], agent)  # if agent['genre'] not in ['DBB', 'EPM'] else None
                    if agent['genre'] == 'EP':
                        del agent['portfolio_of_projects']
                        del agent['portfolio_of_plants']
                    for _entry in list(agent.keys()):
                        """print(agent['name'], _entry, list(agent.keys()), agent['genre'], config.ENTRIES_TO_SAVE[agent['genre']],
                              _entry in config.ENTRIES_TO_SAVE[agent['genre']])"""
                        # print('keys', list(agent.keys()))
                        _to_save = config.ENTRIES_TO_SAVE[agent['genre']]
                        if ((isinstance(agent[_entry], list)) and _entry != 'strikables_dict') and _entry in _to_save:
                            agent[_entry] = agent[_entry][0]
                        if _entry not in _to_save and _entry != 'strikables_dict':
                            # print(config.ENTRIES_TO_SAVE[agent['genre']])
                            # print(config.ENTRIES_TO_SAVE[agent['genre']])
                            del agent[_entry]
                    print(agent['strikables_dict']) if agent['genre'] in ['EPM', 'TPM', 'DBB'] and entry == 300 else None
                    del agent['strikables_dict']
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

    """del config
    del agents
    del sys.modules[__init__.agents_name]
    del sys.modules[__init__.config_name]
    importlib.invalidate_caches()"""

    # sys.modules.pop('config')
    # sys.modules.pop('agents')
    # del sys.modules['config']
    # del sys.modules['agents']

    # reload_package(config)

    return printable


if __name__ == '__main__':
    file = os.environ['_name_o_run']
    RUN_TIMES = 100
    global_start = timeit.default_timer()
    # from config import name
    """try:
        del __init__
        del config
        del agents
        importlib.invalidate_caches()
        import __init__
        print('deleted')
    except:
        None"""
    import __init__
    reload(__init__)

    __init__.config_name = {0: '_config____' + file}
    # print('main', __init__.config_name)
    __init__.agents_name = {0: '_agents____' + file}

    # config = __import__('_config____1_YES_YES')
    # agents = __import__('_agents____1_YES_YES')
    from config import name
    # print('WE SHALL START THE ' + name + ' RUNS')

    import config

    """
    First simulation runs a little bizarre, so we always ignore it
    """

    list_o_runs = [False]  # , True]  # both runs  # True is random, False is normal

    start_seed = 90  # remember: we skip it

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
                ((stop - start) / 60) * len(list_o_runs) * RUN_TIMES) + ' MINUTES (' + str(stop - start) + \
                          ' seconds for each run)' if cond is True else None
            print(printable_2) if cond is True else None

    # reload(config)
    # sys.modules[__init__.config_name[0]] = agents
    # del sys.modules[__init__.config_name[0]]
    # import config
    # reload(__init__)

    global_stop = timeit.default_timer()
    print('IT TOOK ' + str(round((global_stop - global_start) / 60, 2)) + ' MINUTES TO RUN THE ' + str(RUN_TIMES) + " " + name +
          ' SIMULATIONS (double it for the random runs and give it to the next person)')

    duration = 1500  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
