#################################################################
#                                                               #
#                        Packages go here                       #
#                                                               #
#################################################################

# check before importing

import simpy
# from __main__ import env

# !pip install simpy #on colab it must be this pip install thing, dunno why

"""
    Name of all the files comes from the name here
"""

name = "____1_YES_NO"

"""
    Description
"""

descrpt = "No homogeneity between policymakers, just BNDES"

blem = [10]  # example value

"""
Variables that we want to save:
"""

ENTRIES_TO_SAVE = {
    'DD': [
        'name',
        'genre',
        'Demand',
        'Remaining_demand',
        # 'Demand_by_source',
        'Price',
        'homo',
    ],
    'EP': [
        "name",
        "genre",
        "wallet",
        # "portfolio_of_plants",
        # "portfolio_of_projects",
        # "periodicity",
        # "tolerance",
        # "last_acquisition_period",
        "source",
        "decision_var",
        # "LSS_thresh",
        # "impatience",
        # "memory",
        # "discount",
        # "past_weight",
        # "current_weight",
        # "index",
        # "strikables_dict",
        # "verdict",
        "profits",
        # "dd_profits",
        "LSS_tot",
        "shareholder_money",
        "LSS_weak"
    ],
    'TP': [
        "name",
        "genre",
        "wallet",
        "capacity",
        # "Technology",
        "profits",
        # "RnD_threshold",
        "RandD",
        # "capacity_threshold",
        "decision_var",
        # "cap_conditions",
        # "verdict",
        # "self_NPV",
        # "capped",
        # "impatience",
        # "past_weight",
        # "LSS_thresh",
        # "memory",
        # "discount",
        # "strategy",
        "innovation_index",
        "shareholder_money",
        "true_innovation_index",
        "LSS_tot",
        "prod_cap_pct",
        "LSS_weak"
    ],
    'BB': [
        "financing_index",
        "Portfolio",
        "receivable",
        "accepted_sources",
        "car_ratio",
        "name",
        "genre",
        "subgenre",
        "wallet",
        "profits",
        "dd_profits",
        "dd_source",
        "decision_var",
        "verdict",
        "dd_kappas",
        "dd_qual_vars",
        "dd_backwardness",
        "dd_avg_time",
        "dd_discount",
        "dd_strategies",
        "source",
        "kappa",
        "qual_vars",
        "backwardness",
        "avg_time",
        "discount",
        "strategy",
        "index",
        "value",
        "interest_rate",
        "LSS_weak"
    ],
    'EPM': [
        "genre",
        # "subgenre",
        "name",
        "wallet",
        # "PPA_expiration",
        # "PPA_limit",
        # "auction_countdown",
        # "auction_time",
        # "COUNTDOWN",
        # "decision_var",
        "disclosed_var",
        # "verdict",
        # "index_per_source",
        # "auction_capacity",
        # "instrument",
        "source",
        # "LSS_thresh",
        # "impatience",
        # "disclosed_thresh",
        # "past_weight",
        # "memory",
        # "discount",
        # "policies",
        # "rationale",
        "LSS_tot",
        "strikables_dict",
        "current_state",
        "LSS_weak"
    ],
    'TPM': [
        "genre",
        # "subgenre",
        "name",
        "wallet",
        # "instrument",
        # "source",
        # "decision_var",
        "disclosed_var",
        # "verdict",
        # "LSS_thresh",
        # "impatience",
        # "disclosed_thresh",
        # "past_weight",
        # "memory",
        # "discount",
        # "policies",
        # "rationale",
        "current_state",
        "LSS_tot",
        "LSS_weak"
    ],
    'DBB': [
        # "NPV_THRESHOLD_DBB",
        # "guaranteed_contracts",
        "genre",
        "name",
        "wallet",
        # "instrument",
        "source",
        # "decision_var",
        "disclosed_var",
        # "verdict",
        # "LSS_thresh",
        # "impatience",
        # "disclosed_thresh",
        # "past_weight",
        # "memory",
        # "discount",
        # "policies",
        # "rationale",
        # "financing_index",
        # "receivable",
        # "car_ratio",
        # "strikables_dict",
        "current_state",
        "LSS_tot",
        # "interest_rate",
        "LSS_weak"
    ],
    'MIX': [
        # "BB",
        # "CAPEX",
        # "CF",
        # "EP",
        # "Lumps",
        # "MW",
        "MWh",
        # "OPEX",
        # "TP",
        # "ammortisation",
        # "auction_contracted",
        "avoided_emissions",
        # "building_time",
        "capacity",
        # "code",
        # "completion",
        # "dispatchable",
        "emissions",
        # "green",
        # "guarantee",
        # "last_marginal_innovation",
        # "last_radical_innovation",
        # "lifetime",
        # "limit",
        # "name",
        # "old_CAPEX",
        # "old_OPEX",
        "price",
        # "principal",
        # "receiver",
        # "retirement",
        # "sender",
        "source",
        # "source_name",
        "status",
        # "transport",
        # "reason",
        # "risk"
    ],
    'DICTS': [
        'MIX',
        'AGENTS'
    ]
}

"""
Global constants and variables
"""

STARTING_TIME = None
env = simpy.Environment(initial_time=0)
seed = None
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
MARGIN = 4.5
INITIAL_DEMAND = 3.5 * 10 ** 4
STARTING_PRICE = 11
RADICAL_THRESHOLD = 2
RISKS = {0: 0, 1: 0, 2: 0}
SIM_TIME = 12 * 20
FUSS_PERIOD = int(SIM_TIME * 0.3)
SIM_TIME += FUSS_PERIOD
INITIAL_RANDOMNESS = 0.1
RANDOMNESS = INITIAL_RANDOMNESS
TP_THERMAL_PROD_CAP_PCT = 0.5

"""
Technology
"""


THERMAL = {
    "CAPEX": 1000000,
    'OPEX': 30000,
    "MW": 30,
    'CF': .5,
    "lifetime": 30 * 12,
    'building_time': 24,
    'emissions': 100
}

WIND = {
    "CAPEX": 250000,
    'OPEX': 3000,
    "MW": 10,
    'CF': .29,
    "lifetime": 25 * 12,
    'building_time': 12
}

SOLAR = {
    "CAPEX": 10000,
    'OPEX': 150,
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
                                           "dispatchable": False,
                                           "transport": False,
                                           "CF": THERMAL.get("CF"),
                                           "MW": THERMAL.get("MW"),
                                           "lifetime": THERMAL.get("lifetime"),
                                           'building_time': THERMAL.get('building_time'),
                                           'last_radical_innovation': 0,
                                           'last_marginal_innovation': 0,
                                           'emissions': THERMAL['emissions'],
                                           'avoided_emissions': 0}}
                        })  # ,

    # If using fixed technology uncomment!

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
