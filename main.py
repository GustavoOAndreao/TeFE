from importlib import reload
import json
import sys
import random
import simpy
name = 'test'
import config
from config import *


def run_sim(seed):
    import config
    import simpy

    config.env = simpy.Environment(initial_time=0)
    config.seed = seed

    import agents
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

    RUN_TIMES = 10
    for seed in range(0, RUN_TIMES):
        run_sim(seed)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
