blem = [10] #example value

SIM_TIME = 12*30
env = simpy.Environment(initial_time=SIM_TIME)
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
NAMES = {'EP': [], 'TP': [], 'BB': []}
