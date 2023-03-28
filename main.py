from importlib import reload
import json
import sys
import random
import simpy
import config
from config import *
import subprocess, os, sys
from importlib import reload
import timeit
from matplotlib import pyplot as plt
from icecream import ic
import numpy as np


def run_sim(_seed, _name='test'):
    import config
    import simpy

    reload(config)  # need to reload the file

    config.env = simpy.Environment(initial_time=0)
    config.seed = _seed
    config.STARTING_TIME = 0

    import agents
    reload(agents)  # we also need to reload the agents
    config.env.run(until=config.SIM_TIME)

    with open('analysis/json_MIX_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
        json.dump(config.MIX, fp, sort_keys=True, indent=4)

    with open('analysis/json_CONTRACTS_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
        json.dump(config.CONTRACTS, fp)

    with open('analysis/json_AGENTS_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
        json.dump(config.AGENTS, fp)

    with open('analysis/json_TECHNOLOGIC_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
        json.dump(config.TECHNOLOGIC, fp)

    run_graphs(config.AGENTS, config.CONTRACTS, config.MIX, config.TECHNOLOGIC,
               save=True, name=_name + '_' + str(_seed))


def run_graphs(agents_dict, contracts_dict, mix_dict, technologic_dict, save=False,
               name='test', pathfile='analysis/Figures/', show=False):
    """
    Function for producing specific graphs for checking how things are going

    :param save:
    :param name:
    :param pathfile:
    :param agents_dict:
    :param contracts_dict:
    :param mix_dict:
    :param technologic_dict:
    :return:
    """

    #################################################################
    #                                                               #
    #                          FIRST GRAPH:                         #
    #                 Adaptation in relation to time                #
    #                                                               #
    #################################################################

    _EP_adaptation = {}
    DBB_adaptation = []
    time = list(range(0, len(agents_dict)))

    for period in agents_dict:
        _EP_adaptation[period] = []
        for entry in list(agents_dict[period].keys()):
            agent = agents_dict[period][entry]

            if agent['genre'] == 'EP':
                _EP_adaptation[period].append(agent['LSS_tot'])
            if agent['genre'] == 'DBB':
                DBB_adaptation.append(agent['LSS_tot'])

    # ic(_EP_adaptation)
    # ic(DBB_adaptation)
    EP_adaptation = []
    for period in _EP_adaptation:
        adaptations = sum(_EP_adaptation[period])
        EP_adaptation.append(adaptations/len(config.EP_NAME_LIST))

    fig = plt.figure()
    ax = plt.axes()

    LSS_EPs = ax.plot(time, EP_adaptation, color='g', label='Adaptation of EPs')
    LSS_DBB = ax.plot(time, DBB_adaptation, color='b', label='Adaptation of DBB')

    ax.legend()

    plt.xlabel("Time")
    plt.ylabel("Adaptation")

    title = "Adaptation of EPs and DBB"
    plt.title(title)

    fig.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        fig.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         SECOND GRAPH:                         #
    #            Speed of Adaptation in relation to time            #
    #                                                               #
    #################################################################

    fig2 = plt.figure()
    ax2 = plt.axes()

    EP_speed = []
    DBB_speed = []
    for period in range(len(_EP_adaptation)):
        if period > config.FUSS_PERIOD:
            EP_speed_append = (EP_adaptation[period] - EP_adaptation[period-1]) / EP_adaptation[period-1] if EP_adaptation[period-1] > 0 else 0
            DBB_speed_append = (DBB_adaptation[period] - DBB_adaptation[period-1]) / DBB_adaptation[period-1] if DBB_adaptation[period-1] > 0 else 0

            EP_speed.append(EP_speed_append)
            DBB_speed.append(DBB_speed_append)

    speed_EPs = ax2.plot(list(range(0, len(EP_speed))), EP_speed, color='g', label='Speed of Adaptation of EPs')
    speed_DBB = ax2.plot(list(range(0, len(DBB_speed))), DBB_speed, color='b', label='Speed of Adaptation of DBB')

    # Program to calculate moving average
    arrEP = EP_speed
    arrDBB = DBB_speed
    window_size = 12

    i = 0
    # Initialize an empty list to store moving averages
    EP_speed_WA = []
    DBB_speed_WA = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arrEP) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        windowEP = arrEP[i: i + window_size]
        windowDBB = arrDBB[i: i + window_size]

        # Calculate the average of current window
        window_averageEP = round(sum(windowEP) / window_size, 2)
        window_averageDBB = round(sum(windowDBB) / window_size, 2)

        # Store the average of current
        # window in moving average list
        EP_speed_WA.append(window_averageEP)
        DBB_speed_WA.append(window_averageDBB)

        # Shift window to right by one position
        i += 1

    # print(EP_speed_WA)
    # print(DBB_speed_WA)

    speed_EPs_cumsum = ax2.plot(list(range(0, len(EP_speed_WA))), EP_speed_WA, color='g', label='Speed of Adaptation of EPs (weighted average)', linestyle='dashed')
    speed_DBB_cumsum = ax2.plot(list(range(0, len(DBB_speed_WA))), DBB_speed_WA, color='b', label='Speed of Adaptation of DBB (weighted average)', linestyle='dashed')

    ax2.legend()

    plt.xlabel("Time")
    plt.ylabel("Adaptation")

    title="Speed of adaptation of EPs and DBB"

    plt.title(title)

    fig2.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        fig2.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                          THIRD GRAPH:                         #
    #             Goal achievement in relation to time              #
    #                                                               #
    #################################################################

    #################################################################
    #                                                               #
    #                         FOURTH GRAPH:                         #
    #         Speed of Goal achievement in relation to time         #
    #                                                               #
    #################################################################

    #################################################################
    #                                                               #
    #                         FIFTH GRAPH:                          #
    #           Goal achievement in relation to Adaptation          #
    #                                                               #
    #################################################################

    #################################################################
    #                                                               #
    #                         SIXTH GRAPH:                          #
    #            Speed of Goal achievement in relation to           #
    #                    the speed of Adaptation                    #
    #                                                               #
    #################################################################


if __name__ == '__main__':

    RUN_TIMES = 1
    global_start = timeit.default_timer()
    for seed in range(0, RUN_TIMES):
        start = timeit.default_timer() if seed == 0 else None
        run_sim(seed, 'test')
        """print('Demand', config.AGENTS[config.SIM_TIME - 1]['DD']['Demand'], 'Remaining_demand',
              config.AGENTS[config.SIM_TIME - 1]['DD']['Remaining_demand'])"""
        stop = timeit.default_timer() if seed == 0 else None
        printable_1 = 'SEED IS ' + str(seed) + " AND THIS IS RUN " + str(seed) + " OF " + str(RUN_TIMES)
        print(printable_1)
        printable_2 = " AND IT WILL TAKE ROUGHLY " + str(
            ((stop - start) / 60) * RUN_TIMES) + ' MINUTES' if seed == 0 else None
        print(printable_2) if seed == 0 else None

    global_stop = timeit.default_timer()
    print('IT TOOK ' + str(round((global_stop - global_start) / 60, 2)) + ' MINUTES TO RUN THE ' + str(RUN_TIMES) + ' SIMULATIONS')


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
