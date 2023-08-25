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
from params import PARAMS

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
                 when=12,
                 increase=0)] # INITIAL_DEMAND * random.uniform(0.00005, 0.00015))]

# ic(DEMAND)

POLICY_MAKERS = [DBB(env=env,
                     name='BNDES',
                     wallet=min(max(PARAMS['DBB']['wallet']['min'],
                                    random.normalvariate(PARAMS['DBB']['wallet']['mean'],
                                                         PARAMS['DBB']['wallet']['std'])),
                                PARAMS['DBB']['wallet']['max']),
                     instrument=random.sample(PARAMS['DBB']['instrument'], len(PARAMS['DBB']['instrument'])),
                     source=random.sample(
                         [{2: min(max(PARAMS['DBB']['source']['min'],
                                      random.normalvariate(PARAMS['DBB']['source']['mean'],
                                                           PARAMS['DBB']['source']['std'])),
                                  PARAMS['DBB']['source']['max'])},
                          {1: min(max(PARAMS['DBB']['source']['min'],
                                      random.normalvariate(PARAMS['DBB']['source']['mean'],
                                                           PARAMS['DBB']['source']['std'])),
                                  PARAMS['DBB']['source']['max'])}], 2),
                     decision_var=min(max(PARAMS['DBB']['decision_var']['min'],
                                          random.normalvariate(PARAMS['DBB']['decision_var']['mean'],
                                                               PARAMS['DBB']['decision_var']['std'])),
                                      PARAMS['DBB']['decision_var']['max']),
                     LSS_thresh=random.sample(PARAMS['DBB']['LSS_thresh'], len(PARAMS['DBB']['LSS_thresh'])),
                     past_weight=random.sample(PARAMS['DBB']['past_weight'], len(PARAMS['DBB']['past_weight'])),
                     memory=[int(min(max(PARAMS['DBB']['memory']['min'],
                                          random.normalvariate(PARAMS['DBB']['memory']['mean'],
                                                               PARAMS['DBB']['memory']['std'])),
                                      PARAMS['DBB']['memory']['max']))],
                     discount=[min(max(PARAMS['DBB']['discount']['min'],
                                       random.normalvariate(PARAMS['DBB']['discount']['mean'],
                                                            PARAMS['DBB']['discount']['std'])),
                                   PARAMS['DBB']['discount']['max'])],
                     impatience=[int(min(max(PARAMS['DBB']['impatience']['min'],
                                       random.normalvariate(PARAMS['DBB']['impatience']['mean'],
                                                            PARAMS['DBB']['impatience']['std'])),
                                   PARAMS['DBB']['impatience']['max']))],
                     disclosed_thresh=random.sample(PARAMS['DBB']['disclosed_thresh'], len(PARAMS['DBB']['disclosed_thresh'])),
                     rationale=random.sample(PARAMS['DBB']['rationale'], len(PARAMS['DBB']['rationale'])),
                     policies=[]),
                 EPM(env=env,
                     PPA_expiration=350,  # int(random.uniform(1, 20))  * 12,
                     PPA_limit=min(max(PARAMS['EPM']['PPA_limit']['min'],
                                    random.normalvariate(PARAMS['EPM']['PPA_limit']['mean'],
                                                         PARAMS['EPM']['PPA_limit']['std'])),
                                PARAMS['EPM']['PPA_limit']['max']),
                     COUNTDOWN=min(max(PARAMS['EPM']['COUNTDOWN']['min'],
                                    random.normalvariate(PARAMS['EPM']['COUNTDOWN']['mean'],
                                                         PARAMS['EPM']['COUNTDOWN']['std'])),
                                PARAMS['EPM']['COUNTDOWN']['max']),
                     auction_capacity=random.uniform(0.5, 1) * 1.5,

                     wallet=min(max(PARAMS['EPM']['wallet']['min'],
                                    random.normalvariate(PARAMS['EPM']['wallet']['mean'],
                                                         PARAMS['EPM']['wallet']['std'])),
                                PARAMS['EPM']['wallet']['max']),
                     instrument=random.sample(PARAMS['EPM']['instrument'], len(PARAMS['EPM']['instrument'])),
                     source=random.sample(
                         [{2: min(max(PARAMS['EPM']['source']['min'],
                                      random.normalvariate(PARAMS['EPM']['source']['mean'],
                                                           PARAMS['EPM']['source']['std'])),
                                  PARAMS['EPM']['source']['max'])},
                          {1: min(max(PARAMS['EPM']['source']['min'],
                                      random.normalvariate(PARAMS['EPM']['source']['mean'],
                                                           PARAMS['EPM']['source']['std'])),
                                  PARAMS['EPM']['source']['max'])}], 2),
                     decision_var=min(max(PARAMS['EPM']['decision_var']['min'],
                                          random.normalvariate(PARAMS['EPM']['decision_var']['mean'],
                                                               PARAMS['EPM']['decision_var']['std'])),
                                      PARAMS['EPM']['decision_var']['max']),
                     LSS_thresh=random.sample(PARAMS['EPM']['LSS_thresh'], len(PARAMS['EPM']['LSS_thresh'])),
                     past_weight=random.sample(PARAMS['EPM']['past_weight'], len(PARAMS['EPM']['past_weight'])),
                     memory=[int(min(max(PARAMS['EPM']['memory']['min'],
                                     random.normalvariate(PARAMS['EPM']['memory']['mean'],
                                                          PARAMS['EPM']['memory']['std'])),
                                 PARAMS['EPM']['memory']['max']))],
                     discount=[min(max(PARAMS['EPM']['discount']['min'],
                                       random.normalvariate(PARAMS['EPM']['discount']['mean'],
                                                            PARAMS['EPM']['discount']['std'])),
                                   PARAMS['EPM']['discount']['max'])],
                     impatience=[int(min(max(PARAMS['EPM']['impatience']['min'],
                                         random.normalvariate(PARAMS['EPM']['impatience']['mean'],
                                                              PARAMS['EPM']['impatience']['std'])),
                                     PARAMS['EPM']['impatience']['max']))],
                     disclosed_thresh=random.sample(PARAMS['EPM']['disclosed_thresh'],
                                                    len(PARAMS['EPM']['disclosed_thresh'])),
                     rationale=random.sample(PARAMS['EPM']['rationale'], len(PARAMS['EPM']['rationale'])),
                     policies=[])
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
    hetero_threshold=0.5
)]
