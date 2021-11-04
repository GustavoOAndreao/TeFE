#################################################################
#                                                               #
#                        Packages go here                       #
#                                                               #
#################################################################

# check before importing

import random
import feather
import simpy
# !pip install simpy #on colab it must be this pip install thing, dunno why
import numpy as np
import pandas as pd
import math as math
import scipy
import networkx as nx
import time
from matplotlib import pyplot as plt
import itertools
import uuid
import pdb
import statistics
from icecream import ic
from statistics import median, mean
from collections import Counter

blem = [10]  # example value

SIM_TIME = 12 * 30
random.seed(1)
STARTING_TIME = 0
env = simpy.Environment(initial_time=STARTING_TIME)
EP_NAME_LIST = []
TP_NAME_LIST = []
BB_NAME_LIST = []
CONTRACTS = {}
CONTRACTS_r = {}
AGENTS = {}
AGENTS_r = {}
DEMAND = {}
MIX = {}
TECHNOLOGIC = {}
TECHNOLOGIC_r = {}
rev_dict = {}
r = 0.001
POLICY_EXPIRATION_DATE = 12 * 10
TACTIC_DISCOUNT = 0.99
rNd_INCREASE = 0.5
M_CONTRACT_LIMIT = 2 * 12
AUCTION_WANTED_SOURCES = []
AMMORT = 20 * 12
NPV_THRESHOLD = 0
NPV_THRESHOLD_DBB = 0
INSTRUMENT_TO_SOURCE_DICT = {1: [1], 2: [2], 12: [1, 2], 4: [4], 5: [5], 45: [4, 5], 1245: [1, 2, 4, 5]}
BASEL = 0.105
MARGIN = .1
INITIAL_DEMAND = {'E': 100, 'M': 80}
KICKSTART_ADDITION = {'E': INITIAL_DEMAND.get('E') / 5,
                      'M': INITIAL_DEMAND.get('M')}
STARTING_PRICE = 25
RADICAL_THRESHOLD = 2
RISKS = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

THERMAL = {"CAPEX": 20000000,
           'OPEX': 30000,
           "MW": 20,
           'CF': .7,
           "lifetime": 20 * 12,
           'building_time': 12}

GAS = {"CAPEX": 20000000,
       'OPEX': 30000,
       "CF": .9,
       "MW": 5,
       "lifetime": 20 * 12,
       'building_time': 12}

for j in range(SIM_TIME):
    CONTRACTS.update({j: {}})
    AGENTS.update({j: {}})
    MIX.update({j: {}})
    TECHNOLOGIC.update({j: {'TP_thermal': {'subgenre': 0,
                                           "EorM": 'E',
                                           'name': 'TP_thermal',
                                           "green": False,
                                           "source": 0,
                                           "source_name": 'thermal',
                                           "CAPEX": THERMAL.get("CAPEX"),
                                           'OPEX': THERMAL.get('OPEX'),
                                           "dispatchable": True,
                                           "transport": False,
                                           "CF": THERMAL.get("CF"),
                                           "MW": THERMAL.get("MW"),
                                           "lifetime": THERMAL.get("lifetime"),
                                           'building_time': THERMAL.get('building_time'),
                                           'last_radical_innovation': 0,
                                           'last_marginal_innovation': 0,
                                           'emissions': 5000,
                                           'avoided_emissions': 0},

                            'TP_gas': {'subgenre': 3,
                                       "EorM": 'M',
                                       "green": False,
                                       "source_name": 'gas',
                                       'name': 'TP_gas',
                                       "source": 3,
                                       "CAPEX": GAS.get("CAPEX"),
                                       'OPEX': GAS.get('OPEX'),
                                       "dispatchable": True,
                                       "transport": False,
                                       "CF": GAS.get("CF"),
                                       "MW": GAS.get("MW"),
                                       "lifetime": GAS.get("lifetime"),
                                       'building_time': GAS.get('building_time'),
                                       'last_radical_innovation': 0,
                                       'last_marginal_innovation': 0,
                                       'emissions': 5000,
                                       'avoided_emissions': 0}}})
