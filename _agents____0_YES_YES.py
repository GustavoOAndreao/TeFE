#################################################################
#                                                               #
#                  This is the list of agents:                  #
#                    First, the public agents                   #
#                                                               #
#################################################################
import random
import config

from config import *
from classes import *

TECHNOLOGY_PROVIDERS = []

for n in list(range(0, 10)):
    TECHNOLOGY_PROVIDERS.append(make_tp(env))

config.TP_NUMBER = len(TECHNOLOGY_PROVIDERS)

# ic(TECHNOLOGY_PROVIDERS)

config.TP_NUMBER = len(TECHNOLOGY_PROVIDERS)

ENERGY_PRODUCERS = []

for n in list(range(0, 50)):
    ENERGY_PRODUCERS.append(make_ep(env))

config.EP_NUMBER = len(ENERGY_PRODUCERS)

PRIVATE_BANK = []

config.BB_NUMBER = len(PRIVATE_BANK)

DEMAND = [Demand(env=env,
                 initial_demand=INITIAL_DEMAND,
                 when=1,
                 increase=INITIAL_DEMAND * random.uniform(0.001, 0.005))]

# ic(DEMAND)

POLICY_MAKERS = [DBB(env=env,
                     name='BNDES',
                     wallet=random.uniform(1, 5) * 10 ** 9,
                     instrument=['finance'],  # ['finance', 'guarantee']
                     source=random.sample([{2: random.uniform(0, 1000)}, {1: random.uniform(0, 1000)}], 2),
                     decision_var=random.uniform(0, 1),
                     LSS_thresh=random.sample([.25, .5, .75, .1, .9], 5),
                     past_weight=random.sample([0.5, 0.25, 0.75], 3),
                     memory=random.sample([int(random.uniform(12,24))], 1),
                     discount=[int(random.uniform(0.00001, 0.0001))],
                     policies=[],
                     impatience=[int(random.uniform(1, 8))],
                     disclosed_thresh=random.sample([0.01, 0.1, 0.25], 3),
                     rationale=['capacity']),
                 EPM(env=env,
                     wallet=random.uniform(1, 5) * 10 ** 9,
                     PPA_expiration=350,  # int(random.uniform(1, 20))  * 12,
                     PPA_limit=12,
                     COUNTDOWN=int(random.uniform(12, 24)),
                     decision_var=random.uniform(0, 1),
                     auction_capacity=random.uniform(0.5, 1) * 1.5,
                     instrument=['auction'],  # random.sample(['auction', 'carbon_tax', 'FiT'], 3),  # ['auction', 'carbon_tax', 'FiT'],
                     source=random.sample([{2: random.uniform(0, 1000)}, {1: random.uniform(0, 1000)}], 2),
                     LSS_thresh=random.sample([.25, .5, .75, .1, .9], 5),
                     impatience=[int(random.uniform(1, 8))],
                     disclosed_thresh=random.sample([0.01, 0.1, 0.25], 3),
                     past_weight=random.sample([0.5, 0.25, 0.75], 1),
                     memory=random.sample([int(random.uniform(12, 24))], 1),
                     discount=[int(random.uniform(0.00001, 0.0001))],
                     policies=[],
                     rationale=['green'])
                 ]

"""DBB(env=env,
                     name='BNDES',
                     wallet=0,
                     instrument=['finance'],  # ['finance', 'guarantee']
                     source=random.sample([{2: random.uniform(0, 1000)}, {1: random.uniform(0, 1000)}], 2),
                     decision_var=1,
                     LSS_thresh=random.sample([.1, .5, .75], 3),
                     past_weight=[1],
                     memory=random.sample([6, 12, 24], 3),
                     discount=random.sample([0.001, 0.005, 0.01], 2),
                     policies=[],
                     impatience=random.sample([1, 3, 4], 3),
                     disclosed_thresh=random.sample([0.1, 0.25, 0.5], 3),
                     rationale=['capacity'])]"""
"""                 EPM(env=env,
                     wallet=random.uniform(0, 5) * 10 ** 9,
                     PPA_expiration=350,  # int(random.uniform(1, 20))  * 12,
                     PPA_limit=12,
                     COUNTDOWN=int(random.uniform(3, 12)),
                     decision_var=random.uniform(0, 1),
                     auction_capacity=random.uniform(0.5, 1) * config.INITIAL_DEMAND,
                     instrument=['auction'],  # random.sample(['auction', 'carbon_tax', 'FiT'], 3),  # ['auction', 'carbon_tax', 'FiT'],
                     source=random.sample([{2: random.uniform(0, 1000)}, {1: random.uniform(0, 1000)}], 2),
                     LSS_thresh=random.sample([.1, .5, .75], 3),
                     impatience=random.sample([1, 3, 4], 3),
                     disclosed_thresh=random.sample([0.1, 0.25, 0.5], 3),
                     past_weight=[0.5, 0.25, 0.75],
                     memory=random.sample([6, 12, 24], 3),
                     discount=random.sample([0.001, 0.005, 0.01], 2),
                     policies=[],
                     rationale=['green'])]"""  # ,
"""                 TPM(env=env,
                     wallet=random.uniform(0, 10) * 10 ** 7,
                     boundness_cost=random.betavariate(3,1),
                     instrument=random.sample(['unbound', 'bound'], 2),
                     source=random.sample([{2: random.uniform(0, 1000)}, {1: random.uniform(0, 1000)}], 2),
                     decision_var=random.uniform(0, 1),
                     LSS_thresh=random.sample([.1, .5, .75], 3),
                     impatience=random.sample([1, 3, 4], 3),
                     disclosed_thresh=random.sample([0.1, 0.25, 0.5], 3),
                     past_weight=[0.5, 0.25, 0.75],
                     memory=random.sample([6, 12, 24], 3),
                     discount=random.sample([0.001, 0.005, 0.01], 2),
                     policies=[],
                     rationale=['innovation'])]"""

# config.BB_NUMBER += 1  # comment if there is no DBB

#ic(POLICY_MAKERS)

config.PUB_NUMBER = len(POLICY_MAKERS)

HETEROGENEITY = [Hetero(
    env=env,
    hetero_threshold=0
)]
