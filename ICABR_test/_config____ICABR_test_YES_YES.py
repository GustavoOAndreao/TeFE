#################################################################
#                                                               #
#                        Packages go here                       #
#                                                               #
#################################################################

# check before importing

import simpy
# from __main__ import env

# !pip install simpy #on colab it must be this pip install thing, dunno why

blem = [10]  # example value


STARTING_TIME = None
env = simpy.Environment(initial_time=0)
seed = None
name = "____ICABR_TEST_YES_YES"
EP_NUMBER = 0
TP_NUMBER = 0
BB_NUMBER = 0
PUB_NUMBER = 0
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
# M_CONTRACT_LIMIT = 2 * 12
AUCTION_WANTED_SOURCES = []
AMMORT = 25 * 12
NPV_THRESHOLD = 0
NPV_THRESHOLD_DBB = 0
INSTRUMENT_TO_SOURCE_DICT = {1: [1], 2: [2], 12: [1, 2], 120: [0, 1, 2]}
BASEL = 0.105
MARGIN = .1
INITIAL_DEMAND = 100
STARTING_PRICE = 25
RADICAL_THRESHOLD = 2
RISKS = {0: 0, 1: 0, 2: 0}
FUSS_PERIOD = 60
INITIAL_RANDOMNESS = 0.15
RANDOMNESS = INITIAL_RANDOMNESS
TP_THERMAL_PROD_CAP_PCT = 0.5
SIM_TIME = 12 * 30 + FUSS_PERIOD

THERMAL = {
    "CAPEX": 1000000,
    'OPEX': 30000,
    "MW": 30,
    'CF': .7,
    "lifetime": 25 * 12,
    'building_time': 24
}

"""
For fixed technologies only, if using TPs comment it
"""

WIND = {
    "CAPEX": 20000000,
    'OPEX': 3000,
    "MW": 15,
    'CF': .29,
    "lifetime": 30 * 12,
    'building_time': 12
}

SOLAR = {
    "CAPEX": 10000,
    'OPEX': 200,
    "MW": 1,
    'CF': .1,
    "lifetime": 25 * 12,
    'building_time': 6
}

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
                                           'avoided_emissions': -5000}}
                            })  # ,
"""                            'TP_solar' : {'name': 'TP_solar',
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
                                          "source": 1,
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
                                          'avoided_emissions': 250}
                            }
                            }) """
