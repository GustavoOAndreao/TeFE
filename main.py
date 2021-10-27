# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


####################################################
# First we start the environment and global things #
####################################################
####################################################

env = simpy.Environment(initial_time=0)  # change initial time to allow replication
random.seed(1)
SIM_TIME = 30 * 12  # how many months will the simulation run?
EP_NAME_LIST = []
TP_NAME_LIST = []
BB_NAME_LIST = []  # Initializing the llist with the names of the agents
CONTRACTS = {}  # initializing the dictionary of contracts
AGENTS = {}  # initializing the dictionary of agents
MIX = {}  # initializing the dictionary of infrastructure
TECHNOLOGIC = {}  # initializing the dictionary of technologies
rev_dict = {}
r = 0.001  # monthly real tax rate (already including inflation)
POLICY_EXPIRATION_DATE = 12 * 10  # after how many months are the policies discarted?
TACTIC_DISCOUNT = 0.99  # how much of past votes are carried out to the next period (1 is all, zero is none)
rNd_INCREASE = 0.5  # how much more difficult is to innovate after one innovation?
M_CONTRACT_LIMIT = 2 * 12  # for how many months are molecule contracts contracted?
AUCTION_WANTED_SOURCES = []  # initializing the list of auctions currently on auction
AMMORT = 20 * 12  # how many months do companies have to pay for their projects?
NPV_THRESHOLD = 0  # how much NPV should a project have to be accepted by a private bank?
INSTRUMENT_TO_SOURCE_DICT = {1: [1], 2: [2], 12: [1, 2], 4: [4], 5: [5], 45: [4, 5], 1245: [1, 2, 4, 5]}
BASEL = 0.105  # how much car_ratio can the banks have
MARGIN = .1  # how much margin should energy producers get for electricity projects?
INITIAL_DEMAND = {'E': 100, 'M': 80}  # demmand in MW
KICKSTART_ADDITION = {'E': INITIAL_DEMAND.get('E') / 5,
                      'M': INITIAL_DEMAND.get('M')}  # how much MW to start adding to the system at t=1?
STARTING_PRICE = 25  # if there is no capacity yet, what should be the price for the sources?
RADICAL_THRESHOLD = 2  # how large should the a (the result of the probability distribution) be for that to be considered radical nnovation?
RISKS = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # risk dictionary for sources
NAMES = {'EP': [], 'TP': [], 'BB': []}

##########################################
# Now, we start the Technology producers #
##########################################
##########################################

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
    rev_dict({j: 0})
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
                                           'last_marginal_innovation': 0},
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
                                       'last_marginal_innovation': 0}}})

flagSHIPS = [4, 4, 4, 4]  # we want four technology providers for each source

Technologies_list = [{
    "source": 1,
    "CAPEX": 20000000,
    'OPEX': 30000,
    "dispatchable": False,
    "CF": .29,
    "MW": 10,
    "lifetime": 20 * 12,
    'building_time': 12
}, {
    "source": 2,
    "CAPEX": 100000,
    'OPEX': 200,
    "dispatchable": False,
    "CF": .1,
    "MW": 1,
    "lifetime": 20 * 12,
    'building_time': 6
}, {
    "source": 4,
    "CAPEX": 20000000,
    'OPEX': 30000,
    "dispatchable": False,
    "CF": .74,
    "MW": 2,
    "lifetime": 20 * 12,
    'building_time': 12
}, {
    "source": 5,
    "CAPEX": 20000000,
    'OPEX': 30000,
    "dispatchable": False,
    "CF": .75,
    "MW": 50,
    "lifetime": 20 * 12,
    'building_time': 12
}]

TP_DICT = {'Strategies': [
    ['capacity', 'keep', .25],
    ['capacity', 'keep', .25],
    ['capacity', 'keep', .25],
    ['capacity', 'keep', .25]
],
    'Capacities': [5000000, 10000000, 20000000, 15000000],
    'Technologies': Technologies_list,
    'Wallets': [10000000, 10000000, 10000000, 10000000],
    'RnD_thresholds': [5000000, 5000000, 5000000, 5000000],
    'Capacity_thresholds': [3, 3, 3, 3]
}

NUMBER_OF_TP_DICT = {0: 0, 1: 0, 2: 0,
                     3: 0, 4: 0, 5: 0,
                     12: 0, 45: 0, 1245: 0}

for i in range(len(flagSHIPS)):
    for j in range(flagSHIPS[i]):
        traits = {
            'wallet': TP_DICT.get('Wallets')[i],
            'name': 'TP_' + str(i) + str(j),
            'subgenre': TP_DICT.get('Technologies')[i].get('subgenre'),
            'strategy': TP_DICT.get('Strategies')[i],
            'capacity': TP_DICT.get('Capacities')[i],
            'technology': TP_DICT.get('Technologies')[i],
            'RnD_threshold': TP_DICT.get('RnD_thresholds')[i],
            'capacity_threshold': TP_DICT.get('Capacity_thresholds')[i]
        }
        NAMES.get('TP').append(traits.get('name'))
        Create('TP', traits)

NUMBER_OF_TP_DICT.update({
    12: NUMBER_OF_TP_DICT.get(1) + NUMBER_OF_TP_DICT.get(2)
})
NUMBER_OF_TP_DICT.update({
    45: NUMBER_OF_TP_DICT.get(4) + NUMBER_OF_TP_DICT.get(5)
})
NUMBER_OF_TP_DICT.update({
    1245: NUMBER_OF_TP_DICT.get(12) + NUMBER_OF_TP_DICT.get(45)
})

NUMBER_OF_TP = sum(flagSHIPS) + 2

##################################
# And we start the private Banks #
##################################
##################################

# each entry is a different type of firm and the number is how many copies it will have
flagSHIPS = [3, 1]  # we want 3 banks that only accept fossils and one that also accepts wind
BB_DICT = {'Strategies': [[1, 'keep', .1], [2, 'keep', .25]],
           'Accepted_sources': [
               {0: True, 1: False, 2: False, 3: True, 4: False, 5: False},
               {0: True, 1: True, 2: False, 3: True, 4: False, 5: False}],
           'Wallets': [10000000, 10000000],
           'Portfolios': [[], []],
           'Rationales': ['project_finance', 'project_finance']}

for i in range(len(flagSHIPS)):
    for j in range(flagSHIPS[i]):
        traits = {
            'wallet': BB_DICT.get('Wallets')[i],
            'name': 'BB_' + str(i) + str(j),
            'subgenre': 'BB',
            'strategy': BB_DICT.get('Strategies')[i],
            'portfolio': BB_DICT.get('Portfolios')[i],
            'accepted_sources': BB_DICT.get('Accepted_sources')[i],
            'rationale': BB_DICT.get('Rationales')[i]
        }
        NAMES.get('BB').append(traits.get('name'))
        Create('BB', traits)
NUMBER_OF_BB = sum(flagSHIPS)

##########################################
# And then we start the energy producers #
##########################################
##########################################

flagSHIPS = [9, 1, 9,
             1]  # we want 9 electricity producers that use fossil and one that uses wind, and 9 molecule producers that use fossil and one that uses biogas

EP_Dict = {'Strategies': [
    [0, 'keep', .1],
    [1, 'keep', .25],
    [3, 'keep', .1],
    [4, 'keep', .25]
],
    'Wallets': [10000000, 10000000, 10000000, 10000000],
    'Periodicities': [3, 3, 3, 3],
    'Tolerances': [12, 12, 12, 12]
}

for i in range(len(flagSHIPS)):
    for j in range(flagSHIPS[i]):
        traits = {
            'wallet': EP_Dict.get('Wallets')[i],
            'strategy': EP_Dict.get('Strategies')[i],
            'name': 'EP_' + str(i) + str(j),
            'periodicity': EP_Dict.get('Periodicities')[i],
            'tolerance': EP_Dict.get('Tolerances')[i]
        }
        NAMES.get('EP').append(traits.get('name'))
        Create('EP', traits)
NUMBER_OF_EP = sum(flagSHIPS)

#######################
# the demmand #
#######################
#######################

# INITIAL_DEMAND = {'E' : 3000*10**9, 'M' : 800*10**9} It was put at the beggining
DEMAND_SPECIFICITIES = {'when': 60, 'green_awareness': 0.1, 'EorM': 0.5, 'increase': 0.75}
DD = Demand(env)

#############################################
# Now, we start the technology policy maker #
#############################################
#############################################

traits = {'wallet': 4.5 * 10 ** 9,
          'meta_strategy': {
              'first_to_change': 'T',
              'max_strikes_for_the_first_option': 5,
              'second_to_change': 'I',
              'max_strikes_for_the_second_option': 3,
              'third_to_change': 'R',
              'max_strikes_for_the_third_option': 2},
          'RIVAT': {'Rationale': 'innovation',
                    'Instrument': 'supply',
                    'Value': 0.5,
                    'Action': 'keep',
                    'Target': 1245},
          'ranks': {'rationale': {'innovation': 1,
                                  'capacity': 0,
                                  'expansion': 2},
                    'instrument': {'supply': 0}}
          }

Create('TPM', traits)

#########################################
# Now, we start the energy policy maker #
#########################################
#########################################

traits = {'wallet': 4.5 * 10 ** 9,
          'meta_strategy': {
              'first_to_change': 'T',
              'max_strikes_for_the_first_option': 5,
              'second_to_change': 'I',
              'max_strikes_for_the_second_option': 3,
              'third_to_change': 'R',
              'max_strikes_for_the_third_option': 2},
          'RIVAT': {'Rationale': 'innovation',
                    'Instrument': 'carbon_tax',
                    'Value': 0.5,
                    'Action': 'keep',
                    'Target': 1245},
          'ranks': {'rationale': {'innovation': 1,
                                  'capacity': 0,
                                  'expansion': 2},
                    'instrument': {'carbon_tax': 0,
                                   'FiT': 1,
                                   'auction': 2}},
          'PPA_expiration': 25 * 12,
          'PPA_limit': 5 * 12,
          'auction_countdown': 6,
          }

Create('EPM', traits)

#######################################
# Now, we start the  Development Bank #
#######################################
#######################################

traits = {'wallet': 4.5 * 10 ** 9,
          'meta_strategy': {
              'first_to_change': 'T',
              'max_strikes_for_the_first_option': 5,
              'second_to_change': 'I',
              'max_strikes_for_the_second_option': 3,
              'third_to_change': 'R',
              'max_strikes_for_the_third_option': 2},
          'RIVAT': {'Rationale': 'capacity',
                    'Instrument': 'guarantee',
                    'Value': 0.5,
                    'Action': 'keep',
                    'Target': 1245},
          'ranks': {'rationale': {'innovation': 1,
                                  'capacity': 0,
                                  'expansion': 2},
                    'instrument': {'finance': 0,
                                   'guarantee': 1}},
          'NPV_threshold_for_the_DBB': 0,
          'financing_index': {
              0: 0,
              1: 0,
              2: 0,
              3: 0,
              4: 0,
              5: 0}
          }

Create('DBB', traits)

###################
# And now, we run #
###################
###################
t = 40
# t=SIM_TIME
env.run(until=t)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    env.run(until=t)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
