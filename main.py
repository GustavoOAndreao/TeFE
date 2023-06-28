from importlib import reload
import json
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
import winsound

def change_dict_key(d, old_key, new_key, default_value=None):
    d[new_key] = d.pop(old_key, default_value)


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

    # Removing the fuss periods from the dictionaries

    for _dict in [config.MIX, config.CONTRACTS, config.AGENTS, config.TECHNOLOGIC]:
        for line in range(0, config.FUSS_PERIOD):
            del _dict[line]

            # Don't uncomment the following lines: it resets the dictionaries to the zeroth period, but creates some
            # problems due to this...
        """for line in range(config.FUSS_PERIOD, config.SIM_TIME):
            change_dict_key(_dict, line, line-config.FUSS_PERIOD)"""

    with open('analysis/json_MIX_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
        json.dump(config.MIX, fp, sort_keys=True, indent=4)

    with open('analysis/json_CONTRACTS_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
        json.dump(config.CONTRACTS, fp)

    with open('analysis/json_AGENTS_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
        json.dump(config.AGENTS, fp)

    with open('analysis/json_TECHNOLOGIC_' + str(_seed) + '_' + str(_name) + '.json', 'w') as fp:
        json.dump(config.TECHNOLOGIC, fp)

    run_graphs(config.AGENTS, config.CONTRACTS, config.MIX, config.TECHNOLOGIC,
               save=True, name=_name + '_' + str(_seed), show=True)


def run_graphs(agents_dict, contracts_dict, mix_dict, technologic_dict, save=False,
               name='test', pathfile='analysis/Figures/', show=False, norm=False, weak=False):
    """
    Function for producing specific graphs for checking how things are going

    :param weak:
    :param show:
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
    #                         BEFORE GRAPHS:                        #
    #                                                               #
    #################################################################

    """
    First we have to produce the four (actually eight) time series that we use:
    
    1- private adaptation
    2- public adaptation
    3- private goal
    4- public goal
    
    And their respective speeds (5 through 8)
    """

    """
    1- private adaptation: priv_adaptation
    2- public adaptation: publ_adaptation
    3- private goal
    4- public goal
    """

    _priv_adaptation = {}
    _publ_adaptation = {}
    _priv_goal = {}
    _publ_goal = {}
    time = list(range(0, len(agents_dict)))
    number_of_priv = config.EP_NUMBER + config.TP_NUMBER + config.BB_NUMBER
    number_of_publ = config.PUB_NUMBER

    name = name + "_weak_" if weak is True else name

    for period in agents_dict:
        _priv_adaptation[period] = []
        _publ_adaptation[period] = []
        _priv_goal[period] = []
        _publ_goal[period] = []
        for entry in list(agents_dict[period].keys()):
            agent = agents_dict[period][entry]

            if agent['genre'] != 'DD':
                append = agent['LSS_tot'] if weak == False else agent['LSS_weak']
            else:
                append = None

            if agent['genre'] == ('EP' or 'TP' or 'BB'):
                _priv_adaptation[period].append(append)
                _priv_goal[period].append(agent['profits'])
            elif agent['genre'] == ('DBB' or 'TPM' or 'EPM'):
                _publ_adaptation[period].append(append)
                _publ_goal[period].append(agent['current_state'])

    # ic(_priv_adaptation)
    # ic(DBB_adaptation)
    priv_adaptation = []
    publ_adaptation = []
    priv_goal = []
    publ_goal = []
    for period in _priv_adaptation:
        adaptations = sum(_priv_adaptation[period])
        goals = sum(_priv_goal[period])
        priv_adaptation.append(adaptations / number_of_priv)
        priv_goal.append(goals / number_of_priv)

    for period in _publ_adaptation:
        adaptations = sum(_publ_adaptation[period])
        goals = sum(_publ_goal[period])
        publ_adaptation.append(adaptations / number_of_publ)
        publ_goal.append(goals / number_of_publ)
    print(priv_adaptation)
    print(priv_goal)
    print(publ_adaptation)
    print(publ_goal)
    """
    5 - Speed of private adaptation : priv_adaptation_speed
    6 - Speed of public adaptation : publ_adaptation_speed
    7 - Speed of private goal : priv_goal_speed 
    8 - Speed of public goal : publ_goal_speed
    """

    priv_goal_speed = []
    publ_goal_speed = []
    priv_adaptation_speed = []
    publ_adaptation_speed = []

    for period in range(1, len(_priv_adaptation)):
        priv_goal_speed_append = (
                                    priv_goal[period] - priv_goal[period - 1]
                            ) / priv_goal[period - 1] if priv_goal[period - 1] > 0 else 0
        publ_goal_speed_append = (
                                    publ_goal[period] - publ_goal[period - 1]
                            ) / publ_goal[period - 1] if publ_goal[period - 1] > 0 else 0

        priv_adaptation_speed_append = (
                                               priv_adaptation[period] - priv_adaptation[period - 1]
                                       ) / priv_adaptation[period - 1] if priv_adaptation[period - 1] > 0 else 0
        publ_adaptation_speed_append = (
                                               publ_adaptation[period] - publ_adaptation[period - 1]
                                       ) / publ_adaptation[period - 1] if publ_adaptation[period - 1] > 0 else 0

        priv_goal_speed.append(priv_goal_speed_append)
        publ_goal_speed.append(publ_goal_speed_append)
        priv_adaptation_speed.append(priv_adaptation_speed_append)
        publ_adaptation_speed.append(publ_adaptation_speed_append)

    # Program to calculate moving average
    arrpriv_goal = priv_goal_speed
    arrpubl_goal = publ_goal_speed
    arrpriv_adaptation = priv_adaptation_speed
    arrpubl_adaptation = publ_adaptation_speed
    window_size = 3

    i = 0
    # Initialize an empty list to store moving averages
    priv_goal_speed_WA = []
    publ_goal_speed_WA = []
    priv_adaptation_speed_WA = []
    publ_adaptation_speed_WA = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arrpriv_goal) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        windowpriv_goal = arrpriv_goal[i: i + window_size]
        windowpubl_goal = arrpubl_goal[i: i + window_size]

        windowpriv_adaptation = arrpriv_adaptation[i: i + window_size]
        windowpubl_adaptation = arrpubl_adaptation[i: i + window_size]

        # Calculate the average of current window
        window_averagepriv_goal = round(sum(windowpriv_goal) / window_size, 2)
        window_averagepubl_goal = round(sum(windowpubl_goal) / window_size, 2)

        window_averagepriv_adaptation = round(sum(windowpriv_adaptation) / window_size, 2)
        window_averagepubl_adaptation = round(sum(windowpubl_adaptation) / window_size, 2)

        # Store the average of current
        # window in moving average list
        priv_goal_speed_WA.append(window_averagepriv_goal)
        publ_goal_speed_WA.append(window_averagepubl_goal)

        priv_adaptation_speed_WA.append(window_averagepriv_adaptation)
        publ_adaptation_speed_WA.append(window_averagepubl_adaptation)

        # Shift window to right by one position
        i += 1

    priv_goal = [(val - min(priv_goal)) / (max(priv_goal) - min(priv_goal)) for val in priv_goal]
    publ_goal = [(val - min(publ_goal)) / (max(publ_goal) - min(publ_goal)) for val in publ_goal]

    if norm is True:
        # publ_goal = [val / max(publ_goal) for val in publ_goal]

        max_adapt = max(max(priv_adaptation), max(publ_adaptation))
        min_adapt = min(min(priv_adaptation), min(publ_adaptation))

        """priv_adaptation = [(val - min(priv_adaptation)) / (max(priv_adaptation) - min(priv_adaptation)) for val in priv_adaptation]
        publ_adaptation = [(val - min(publ_adaptation)) / (max(publ_adaptation) - min(publ_adaptation)) for val in publ_adaptation]"""
        priv_adaptation = [(val - min_adapt) / (max_adapt - min_adapt) for val in priv_adaptation]
        publ_adaptation = [(val - min_adapt) / (max_adapt - min_adapt) for val in publ_adaptation]

    names_dict = {
        str(priv_adaptation): "private adaptation",
        str(priv_goal): "private goal achievement",
        str(publ_adaptation): "public adaptation",
        str(publ_goal): "public goal achievement",
        str(priv_goal_speed): "speed of private goal achievement",
        str(publ_goal_speed): "speed of public goal achievement",
        str(priv_adaptation_speed): "speed of private adaptation",
        str(publ_adaptation_speed): "speed of public adaptation",
        str(priv_goal_speed_WA): "speed of private goal achievement (weighted average)",
        str(publ_goal_speed_WA): "speed of public goal achievement (weighted average)",
        str(priv_adaptation_speed_WA): "speed of private adaptation (weighted average)",
        str(publ_adaptation_speed_WA): "speed of private adaptation (weighted average)"
    }

    #################################################################
    #                                                               #
    #                          FIRST GRAPH:                         #
    #                 Adaptation in relation to time                #
    #                                                               #
    #################################################################

    fig = plt.figure()
    ax = plt.axes()

    x = priv_adaptation
    y = publ_adaptation

    ax.plot(time, x, color='g', label=names_dict[str(x)])
    ax.plot(time, y, color='b', label=names_dict[str(y)])

    ax.legend()

    plt.xlabel("Time")
    plt.ylabel("Adaptation")

    title = "Adaptation of Private and Public agents over time"
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

    # print(EP_speed_WA)
    # print(DBB_speed_WA)

    ax2.plot(list(range(0, len(priv_adaptation_speed))), priv_adaptation_speed, color='g',
             label='Speed of Adaptation of EPs')
    ax2.plot(list(range(0, len(publ_adaptation_speed))), publ_adaptation_speed, color='b',
             label='Speed of Adaptation of DBB')

    ax2.plot(list(range(0, len(priv_adaptation_speed_WA))), priv_adaptation_speed_WA, color='g',
             label='Speed of Adaptation of EPs (weighted average)', linestyle='dashed')
    ax2.plot(list(range(0, len(publ_adaptation_speed_WA))),publ_adaptation_speed_WA, color='b',
             label='Speed of Adaptation of DBB (weighted average)', linestyle='dashed')

    ax2.legend()

    plt.xlabel("Time")
    plt.ylabel("Adaptation")

    title = "Speed of adaptation of private and public agents over time"

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

    fig3 = plt.figure()
    ax3 = plt.axes()

    ax3.plot(time, priv_goal, color='g', label='Goal achievement of Private agents')
    ax3.plot(time, publ_goal, color='b', label='Goal achievement of Public agents')

    ax3.legend()

    plt.xlabel("Time")
    plt.ylabel("Goal Achievement")

    title = "Goal achievement of Private and Public agents over time"
    plt.title(title)

    fig3.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        fig3.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         FOURTH GRAPH:                         #
    #         Speed of Goal achievement in relation to time         #
    #                                                               #
    #################################################################

    fig4 = plt.figure()
    ax4 = plt.axes()

    # print(EP_speed_WA)
    # print(DBB_speed_WA)

    ax4.plot(list(range(0, len(priv_goal_speed))), priv_adaptation_speed, color='g',
             label='Speed of Adaptation of EPs')
    ax4.plot(list(range(0, len(publ_goal_speed))), publ_adaptation_speed, color='b',
             label='Speed of Adaptation of DBB')

    ax4.plot(list(range(0, len(priv_goal_speed_WA))), priv_adaptation_speed_WA, color='g',
             label='Speed of Adaptation of EPs (weighted average)', linestyle='dashed')
    ax4.plot(list(range(0, len(publ_goal_speed_WA))),publ_adaptation_speed_WA, color='b',
             label='Speed of Goal achievement of DBB (weighted average)', linestyle='dashed')

    ax4.legend()

    plt.xlabel("Time")
    plt.ylabel("Adaptation")

    title = "Speed of adaptation of private and public agents over time"

    plt.title(title)

    fig4.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        fig4.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         FIFTH GRAPH:                          #
    #                                                               #
    #################################################################

    """
    Private adaptation in relation to public goals: are private agents adapting towards public goals?
    """

    figA = plt.figure()
    axA = plt.axes()

    color = range(0, len(publ_goal))

    x = publ_goal
    y = priv_adaptation

    points = axA.scatter(x, y, c=color, cmap='viridis')

    figA.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    axA.axline([0, 0], [max(x), max(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figA.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figA.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         SIXTH GRAPH:                          #
    #                                                               #
    #################################################################

    """
    Private adaptation in relation to public goals in terms of speeds
    """

    figAs = plt.figure()
    axAs = plt.axes()

    x = publ_goal_speed
    y = priv_adaptation_speed

    color = range(0, len(x))

    points = axAs.scatter(x, y, c=color, cmap='viridis')

    figAs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)]+ " in relation to " + names_dict[str(x)]
    plt.title(title)

    figAs.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figAs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                    SIXTH AND A HALF GRAPH:                    #
    #                                                               #
    #################################################################

    """
    Private adaptation in relation to public goals in terms of speeds
    """

    figAss = plt.figure()
    axAss = plt.axes()

    x = publ_goal_speed_WA
    y = priv_adaptation_speed_WA

    color = range(0, len(x))

    points = axAss.scatter(x, y, c=color, cmap='viridis')

    figAss.colorbar(points)

    axAss.axline([0, 0], [max(x), max(y)])

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)]+ " in relation to " + names_dict[str(x)]
    plt.title(title)

    figAss.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figAss.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                        SEVENTH GRAPH:                         #
    #                                                               #
    #################################################################

    """
    Public goal achievement in relation to public adaptation: are the adaptation processes leading towards goal completion for policy makers?
    """

    figB = plt.figure()
    axB = plt.axes()

    x = publ_goal
    y = priv_adaptation

    color = range(0, len(publ_adaptation))

    points = axB.scatter(publ_adaptation, publ_goal, c=color, cmap='viridis')

    axB.axline([0, 0], [max(x), max(y)])

    figB.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)]+ " in relation to " + names_dict[str(x)]
    plt.title(title)

    figB.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figB.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         EIGHTH GRAPH:                         #
    #                                                               #
    #################################################################

    """
    Public goal achievement in relation to public adaptation (speeds)
    """

    figBs = plt.figure()
    axBs = plt.axes()

    x = publ_goal_speed
    y = priv_adaptation_speed

    color = range(0, len(publ_goal_speed))

    points = axBs.scatter(publ_goal_speed, publ_adaptation_speed, c=color, cmap='viridis')

    axBs.axline([0, 0], [max(x), max(y)])

    figBs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)]+ " in relation to " + names_dict[str(x)]
    plt.title(title)

    figBs.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figBs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                    EIGHTH AND A HALF GRAPH:                   #
    #                                                               #
    #################################################################

    """
    Public goal achievement in relation to public adaptation (speeds weighted average)
    """

    figBss = plt.figure()
    axBss = plt.axes()

    x = publ_goal_speed_WA
    y = priv_adaptation_speed_WA

    color = range(0, len(x))

    points = axBss.scatter(x, y, c=color, cmap='viridis')

    axBss.axline([0, 0], [max(x), max(y)])

    figBs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figBss.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figBss.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         NINTH GRAPH:                          #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to private adaptation: are the adaptation processes leading towards goal completion for private agents?
    """

    figC = plt.figure()
    axC = plt.axes()

    x = priv_goal
    y = priv_adaptation

    color = range(0, len(publ_adaptation))

    points = axC.scatter(x, y, c=color, cmap='viridis')

    axC.axline([0, 0], [max(x), max(y)])

    figC.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)]+ " in relation to " + names_dict[str(x)]
    plt.title(title)

    figC.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figC.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                          TENTH GRAPH:                         #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to private adaptation: (speeds)
    """

    figCs = plt.figure()
    axCs = plt.axes()

    x = priv_adaptation_speed
    y = priv_goal_speed

    color = range(0, len(publ_goal_speed))

    points = axCs.scatter(x, y, c=color, cmap='viridis')

    axCs.axline([0, 0], [max(x), max(y)])

    figCs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)]+ " in relation to " + names_dict[str(x)]
    plt.title(title)

    figCs.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figCs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                     TENTH AND A HALF GRA                      #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to private adaptation: (speeds weighted average)
    """

    figCss = plt.figure()
    axCss = plt.axes()

    x = priv_adaptation_speed_WA
    y = priv_goal_speed_WA

    color = range(0, len(x))

    points = axCss.scatter(x, y, c=color, cmap='viridis')

    axCss.axline([0, 0], [max(x), max(y)])

    figCss.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figCss.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figCss.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                        ELEVENTH GRAPH:                        #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public adaptation: Is the public adaptation clashing with the private goals?
    """

    figD = plt.figure()
    axD = plt.axes()

    x = publ_adaptation
    y = priv_goal

    color = range(0, len(publ_adaptation))

    points = axD.scatter(x, y, c=color, cmap='viridis')

    axD.axline([0, 0], [max(x), max(y)])

    figD.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)]+ " in relation to " + names_dict[str(x)]
    plt.title(title)

    figD.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figD.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         TWELFTH GRAPH:                        #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public adaptation: (speeds)
    """

    figDs = plt.figure()
    axDs = plt.axes()

    x = publ_adaptation_speed
    y = priv_goal_speed

    color = range(0, len(publ_goal_speed))

    points = axDs.scatter(x, y, c=color, cmap='viridis')

    axDs.axline([0, 0], [max(x), max(y)])

    figDs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figDs.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figDs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                    TWELFTH AND A HALF GRAPH:                  #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public adaptation: (speeds)
    """

    figDss = plt.figure()
    axDss = plt.axes()

    x = publ_adaptation_speed_WA
    y = priv_goal_speed_WA

    color = range(0, len(x))

    points = axDss.scatter(x, y, c=color, cmap='viridis')

    axDss.axline([0, 0], [max(x), max(y)])

    figDss.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figDss.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figDss.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                       THIRTEENTH GRAPH:                       #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public goal achivement: are private and public goals clashing?
    """

    figE = plt.figure()
    axE = plt.axes()

    x = priv_goal
    y = publ_goal

    color = range(0, len(publ_adaptation))

    points = axE.scatter(x, y, c=color, cmap='viridis')

    figE.colorbar(points)

    axE.axline([0, 0], [max(x), max(y)])

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figE.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figE.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         TWELFTH GRAPH:                        #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public goal achievement: (speeds)
    """

    figEs = plt.figure()
    axEs = plt.axes()

    x = priv_goal_speed
    y = publ_goal_speed

    color = range(0, len(publ_goal_speed))

    points = axEs.scatter(x, y, c=color, cmap='viridis')

    axEs.axline([0, 0], [max(x), max(y)])

    figEs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figEs.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figEs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                   TWELFTH AND A HALF GRAPH:                   #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public goal achievement: (speeds)
    """

    figEss = plt.figure()
    axEss = plt.axes()

    x = priv_goal_speed_WA
    y = publ_goal_speed_WA

    color = range(0, len(x))

    points = axEss.scatter(x, y, c=color, cmap='viridis')

    axEss.axline([0, 0], [max(x), max(y)])

    figEss.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figEss.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figEs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                       THIRTEENTH GRAPH:                       #
    #                                                               #
    #################################################################

    """
    Private adaptation in relation to public adaptation: who is accumulating more adaptation?
    """

    figF = plt.figure()
    axF = plt.axes()

    x = priv_adaptation
    y = publ_adaptation

    color = range(0, len(publ_adaptation))

    points = axF.scatter(x, y, c=color, cmap='viridis')

    axF.axline([0, 0], [max(x), max(y)])

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    figF.colorbar(points)

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figF.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figF.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                        FOURTEENTH GRAPH:                      #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public adaptation: (speeds)
    """

    figFs = plt.figure()
    axFs = plt.axes()

    x = priv_adaptation_speed
    y = publ_adaptation_speed

    color = range(0, len(publ_goal_speed))

    points = axFs.scatter(x, y, c=color, cmap='viridis')

    axFs.axline([0, 0], [max(x), max(y)])

    figFs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)]+ " in relation to " + names_dict[str(x)]
    plt.title(title)

    figFs.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figFs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                 FOURTEENTH AND A HALF GRAPH:                  #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public adaptation: (speeds)
    """

    figFss = plt.figure()
    axFss = plt.axes()

    x = priv_adaptation_speed_WA
    y = publ_adaptation_speed_WA

    color = range(0, len(x))

    points = axFss.scatter(x, y, c=color, cmap='viridis')

    axFss.axline([0, 0], [max(x), max(y)])

    figFss.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figFss.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figFss.savefig(pathfile + file_name)


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

    duration = 3000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
