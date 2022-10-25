from importlib import reload
import json
import sys
import random
import simpy
name = 'test'
import config
from config import *
import subprocess, os, sys
from importlib import reload
import timeit


def run_sim(seed):
    import config
    import simpy

    reload(config)  # need to reload the file


    config.env = simpy.Environment(initial_time=0)
    config.seed = seed
    config.STARTING_TIME = 0

    import agents
    reload(agents)  # we also need to reload the agents
    config.env.run(until=config.SIM_TIME)

    with open('analysis/json_MIX_' + str(seed) + '_' + str(name) + '.json', 'w') as fp:
        json.dump(config.MIX, fp, sort_keys=True, indent=4)

    with open('analysis/json_CONTRACTS_' + str(seed) + '_' + str(name) + '.json', 'w') as fp:
        json.dump(config.CONTRACTS, fp)

    with open('analysis/json_AGENTS_' + str(seed) + '_' + str(name) + '.json', 'w') as fp:
        json.dump(config.AGENTS, fp)

    with open('analysis/json_TECHNOLOGIC_' + str(seed) + '_' + str(name) + '.json', 'w') as fp:
        json.dump(config.TECHNOLOGIC, fp)


if __name__ == '__main__':

    RUN_TIMES = 100
    for seed in range(0, RUN_TIMES+1):
        start = timeit.default_timer() if seed == 0 else None
        run_sim(seed)
        stop = timeit.default_timer() if seed == 0 else None
        printable_1 = 'SEED IS ' + str(seed) + " AND THIS IS RUN " + str(seed) + " OF " + str(RUN_TIMES)
        print(printable_1)
        printable_2 = " AND IT WILL TAKE ROUGHLY " + str(((stop - start)/60)*RUN_TIMES) + ' MINUTES' if seed == 0 else None
        print(printable_2) if seed == 0 else None




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
