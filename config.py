#################################################################
#                                                               #
#                        Packages go here                       #
#                                                               #
#################################################################

# check before importing

import random

import simpy

# !pip install simpy #on colab it must be this pip install thing, dunno why

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
r = 0.001
POLICY_EXPIRATION_DATE = 12 * 10
rNd_INCREASE = 0.5
M_CONTRACT_LIMIT = 2 * 12
AUCTION_WANTED_SOURCES = []
AMMORT = 20 * 12
NPV_THRESHOLD = 0
NPV_THRESHOLD_DBB = 0
INSTRUMENT_TO_SOURCE_DICT = {1: [1], 2: [2], 12: [1, 2], 4: [4], 5: [5], 45: [4, 5], 1245: [1, 2, 4, 5]}
BASEL = 0.105
MARGIN = .1
INITIAL_DEMAND = 100
STARTING_PRICE = 25
RADICAL_THRESHOLD = 2
RISKS = {0: 0, 1: 0, 2: 0}

THERMAL = {"CAPEX": 100000000,
           'OPEX': 30000,
           "MW": 100,
           'CF': .7,
           "lifetime": 25 * 12,
           'building_time': 24}

"""
For fixed technologies only, if using TPs comment it
"""

WIND = {"CAPEX": 20000000,
           'OPEX': 30000,
           "MW": 20,
           'CF': .29,
           "lifetime": 25 * 12,
           'building_time': 12}

SOLAR = {"CAPEX": 10000,
           'OPEX': 200,
           "MW": 1,
           'CF': .1,
           "lifetime": 20 * 12,
           'building_time': 6}

for j in range(SIM_TIME):
    CONTRACTS.update({j: {}})
    AGENTS.update({j: {}})
    MIX.update({j: {}})
    TECHNOLOGIC.update({j: {'TP_thermal': {'name': 'TP_thermal',
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
                            'TP_solar' : {'name': 'TP_solar',
                                          "green": True,
                                          "source": 2,
                                          "source_name": 'solar',
                                          "CAPEX": SOLAR['CAPEX'],
                                          'OPEX': SOLAR['OPEX'],
                                          "dispatchable": True,
                                          "transport": True,
                                          "CF": SOLAR['CF'],
                                          "MW": SOLAR['MW'],
                                          "lifetime": SOLAR['lifetime'],
                                          'building_time': SOLAR['building_time'],
                                          'last_radical_innovation': 0,
                                          'last_marginal_innovation': 0,
                                          'emissions': 0,
                                          'avoided_emissions': 50},
                            'TP_wind' : {'name': 'TP_wind',
                                          "green": True,
                                          "source": 2,
                                          "source_name": 'solar',
                                          "CAPEX": WIND['CAPEX'],
                                          'OPEX': WIND['OPEX'],
                                          "dispatchable": True,
                                          "transport": True,
                                          "CF": WIND['CF'],
                                          "MW": WIND['MW'],
                                          "lifetime": WIND['lifetime'],
                                          'building_time': WIND['building_time'],
                                          'last_radical_innovation': 0,
                                          'last_marginal_innovation': 0,
                                          'emissions': 0,
                                          'avoided_emissions': 250}}
                            })
