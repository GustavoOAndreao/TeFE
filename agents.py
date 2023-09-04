#################################################################
#                                                               #
#                  This is the list of agents:                  #
#                    First, the public agents                   #
#                                                               #
#################################################################
import os
import random
import sys
import builtins
# from IPython.lib import deepreload
# builtins.reload = deepreload.reload
import importlib
from importlib import reload

import params
reload(params)
from params import PARAMS

reload(sys.modules[__name__])

importlib.invalidate_caches()
import __init__
reload(__init__)
from __init__ import config_name

# print('agents', agents_name)

# import config
# from config import *
# from classes import *

import config
from config import *

import classes
reload(classes)
from classes import *
# print('agents', seed, config.seed)
random.seed(config.seed)

TECHNOLOGY_PROVIDERS = []

for n in list(range(0, config.number_o_tps)):
    TECHNOLOGY_PROVIDERS.append(make_tp(env))

config.TP_NUMBER = len(TECHNOLOGY_PROVIDERS)

# ic(TECHNOLOGY_PROVIDERS)

config.TP_NUMBER = len(TECHNOLOGY_PROVIDERS)

ENERGY_PRODUCERS = []

for n in list(range(0, config.number_o_eps)):
    ENERGY_PRODUCERS.append(make_ep(env))

config.EP_NUMBER = len(ENERGY_PRODUCERS)

PRIVATE_BANK = []

config.BB_NUMBER = len(PRIVATE_BANK)

DEMAND = [Demand(env=env,
                 initial_demand=min(max(PARAMS['DD']['initial_demand']['min'],
                                    random.normalvariate(PARAMS['DD']['initial_demand']['mean'],
                                                         PARAMS['DD']['initial_demand']['std'])),
                                PARAMS['DD']['initial_demand']['max']),
                 when=min(max(PARAMS['DD']['when']['min'],
                                    random.normalvariate(PARAMS['DD']['when']['mean'],
                                                         PARAMS['DD']['when']['std'])),
                                PARAMS['DD']['when']['max']),
                 increase=min(max(PARAMS['DD']['increase']['min'],
                                    random.normalvariate(PARAMS['DD']['increase']['mean'],
                                                         PARAMS['DD']['increase']['std'])),
                                PARAMS['DD']['increase']['max']))]

ic(DEMAND)


POLICY_MAKERS = []

dbb = False
epm = False
tpm = False

if "YES_YES_YES" in config_name:
    dbb = True
    epm = True
    tpm = True

elif "YES_NO_NO"  in config_name:
    dbb = True
    # epm = False
    # tpm = False

elif "NO_YES_NO" in config_name:
    # dbb = False
    epm = True
    # tpm = False

elif "NO_NO_YES" in config_name:
    # dbb = False
    # epm = False
    tpm = True

elif "YES_YES_NO" in config_name:
    dbb = True
    epm = True
    # tpm = False

elif "NO_YES_YES" in config_name:
    # dbb = False
    epm = True
    tpm = True

elif "YES_NO_YES" in config_name:
    dbb = True
    # epm = False
    tpm = True

possible_pms = [dbb, epm, tpm]

number = 0
for pm in possible_pms:
    if pm is True:
        if number == 0:
            POLICY_MAKERS.append(DBB(env=env,
                     name='BNDES',
                     wallet=min(max(PARAMS['DBB']['wallet']['min'],
                                    random.normalvariate(PARAMS['DBB']['wallet']['mean'],
                                                         PARAMS['DBB']['wallet']['std'])),
                                PARAMS['DBB']['wallet']['max']),
                     instrument=random.sample(PARAMS['DBB']['instrument'], len(PARAMS['DBB']['instrument'])),
                     periodicity= random.sample(PARAMS['periodicity'], len(PARAMS['periodicity'])),
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
                     memory=random.sample(PARAMS['DBB']['memory'], len(PARAMS['DBB']['memory'])),
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
                     policies=[]))
        if number == 1:
            POLICY_MAKERS.append(EPM(env=env,
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
                     periodicity= random.sample(PARAMS['periodicity'], len(PARAMS['periodicity'])),
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
                     memory=random.sample(PARAMS['EPM']['memory'], len(PARAMS['EPM']['memory'])),
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
                     policies=[]))

        if number == 2:
            POLICY_MAKERS.append(TPM(env=env,
                     boundness_cost=min(max(PARAMS['TPM']['boundness_cost']['min'],
                                    random.normalvariate(PARAMS['TPM']['boundness_cost']['mean'],
                                                         PARAMS['TPM']['boundness_cost']['std'])),
                                PARAMS['TPM']['boundness_cost']['max']),

                     wallet=min(max(PARAMS['TPM']['wallet']['min'],
                                    random.normalvariate(PARAMS['TPM']['wallet']['mean'],
                                                         PARAMS['TPM']['wallet']['std'])),
                                PARAMS['TPM']['wallet']['max']),
                     instrument=random.sample(PARAMS['TPM']['instrument'], len(PARAMS['TPM']['instrument'])),
                     periodicity= random.sample(PARAMS['periodicity'], len(PARAMS['periodicity'])),

                     source=random.sample(
                         [{2: min(max(PARAMS['TPM']['source']['min'],
                                      random.normalvariate(PARAMS['TPM']['source']['mean'],
                                                           PARAMS['TPM']['source']['std'])),
                                  PARAMS['TPM']['source']['max'])},
                          {1: min(max(PARAMS['TPM']['source']['min'],
                                      random.normalvariate(PARAMS['TPM']['source']['mean'],
                                                           PARAMS['TPM']['source']['std'])),
                                  PARAMS['TPM']['source']['max'])}], 2),
                     decision_var=min(max(PARAMS['TPM']['decision_var']['min'],
                                          random.normalvariate(PARAMS['TPM']['decision_var']['mean'],
                                                               PARAMS['TPM']['decision_var']['std'])),
                                      PARAMS['TPM']['decision_var']['max']),
                     LSS_thresh=random.sample(PARAMS['TPM']['LSS_thresh'], len(PARAMS['TPM']['LSS_thresh'])),
                     past_weight=random.sample(PARAMS['TPM']['past_weight'], len(PARAMS['TPM']['past_weight'])),
                     memory=random.sample(PARAMS['TPM']['memory'], len(PARAMS['TPM']['memory'])),
                     discount=[min(max(PARAMS['TPM']['discount']['min'],
                                       random.normalvariate(PARAMS['TPM']['discount']['mean'],
                                                            PARAMS['TPM']['discount']['std'])),
                                   PARAMS['TPM']['discount']['max'])],
                     impatience=[int(min(max(PARAMS['TPM']['impatience']['min'],
                                         random.normalvariate(PARAMS['TPM']['impatience']['mean'],
                                                              PARAMS['TPM']['impatience']['std'])),
                                     PARAMS['TPM']['impatience']['max']))],
                     disclosed_thresh=random.sample(PARAMS['EPM']['disclosed_thresh'],
                                                    len(PARAMS['TPM']['disclosed_thresh'])),
                     rationale=random.sample(PARAMS['TPM']['rationale'], len(PARAMS['TPM']['rationale'])),

                     policies=[]))
    number += 1

ic(POLICY_MAKERS)

print(PARAMS['TPM']['rationale'], PARAMS['DBB']['rationale'], PARAMS['EPM']['rationale'])

config.PUB_NUMBER = len(POLICY_MAKERS)

if config_name.split('_')[0] != "1":
    name += '_homo_' if PARAMS['HH']['heterogeneous'] is False else '_hetero_'
    print(name)
    name += str(os.environ['HH_check'])
    if config_name.split('_')[0] == "0":
        HETEROGENEITY = [Hetero(
            env=env,
            threshold=0,
            heterogeneous=PARAMS['HH']['heterogeneous'],
            check=PARAMS['HH']['check']
    )]

    elif config_name.split('_')[0] == "025":
        HETEROGENEITY = [Hetero(
            env=env,
            threshold=0.25,
            heterogeneous=PARAMS['HH']['heterogeneous'],
            check=PARAMS['HH']['check']
    )]

    elif config_name.split('_')[0] == "05":
        HETEROGENEITY = [Hetero(
            env=env,
            threshold=0.5,
            heterogeneous=PARAMS['HH']['heterogeneous'],
            check=PARAMS['HH']['check']
    )]

    elif config_name.split('_')[0] == "075":
        HETEROGENEITY = [Hetero(
            env=env,
            threshold=0.75,
            heterogeneous=PARAMS['HH']['heterogeneous'],
            check=PARAMS['HH']['check']
    )]

# print(os.environ['HH_heterogeneous'] == 'False', os.environ['HH_heterogeneous'])

# print(PARAMS['HH']['heterogeneous'], isinstance(PARAMS['HH']['heterogeneous'], bool))

config.name = name

"""file2 = __import__(agents_name[0])  # importlib.import_module('_agents____1_YES_YES')
reload(file2)  # reload(__import__('_agents____1_YES_YES'))
# importlib.invalidate_caches()
# file2 = importlib.import_module('_agents____1_YES_YES')


for attr in vars(file2):
    try:
        locals()[attr] = vars(file2)[attr]
    except:
        None"""

# del sys.modules['_agents____1_YES_YES']
# sys.modules['_agents____1_YES_YES'] = importlib.import_module('_agents____1_YES_YES')

# import _agents____1_YES_YES
# reload(_agents____1_YES_YES)
# from   _agents____1_YES_YES import *
