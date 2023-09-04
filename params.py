import os
import random

from config import INITIAL_DEMAND
from config import config_name
from config import seed

# print(seed)

random.seed(seed)

LSS_thresh = [.25, .5, .75, .1, .9]
decision_var = {'mean': 0, 'std': 1, 'max': 1, 'min': 0}
past_weight = [0.5, 0.25, 0.75, 0.1, 0.9]
public_source = {'mean': 800, 'std': 800, 'max': 2000, 'min': 500}
memory = [3, 6, 12, 24, 48]  # {'mean': 12, 'std': 12, 'max': 36, 'min': 3}
periodicity = [3, 6, 12, 24, 48]
discount = {'mean': 0.0005, 'std': 0.0005, 'max': 0.0001, 'min': 0.00001}
impatience = {'mean': 3, 'std': 5, 'max': 8, 'min': 1}
disclosed_thresh = [0.01, 0.1, 0.25, 0.5]
rationale = ['green', 'innovation', 'capacity']

rationale_dict = {'epm': None,
                      'tpm': None,
                      'dbb': None}

if os.environ.get('HH_heterogeneous') is not None:
    # basically thresh in zero and it's homogeneity
    chosen = random.choice(rationale)
    for pm in list(rationale_dict.keys()):
        rationale_dict[pm] = chosen

    if eval(os.environ['HH_heterogeneous']) is False:
        # we are enforcing homogeneity to a certain degree
        if config_name.split('_')[0] == "025":
            print('homogeneous with threshold .25 incoming')
            for pm in list(rationale_dict.keys())[1:]:
                if random.uniform(0,1) < 0.25:
                    while rationale_dict[pm] == chosen:
                        rationale_dict[pm] = random.choice(rationale)

        elif config_name.split('_')[0] == "05":
            print('homogeneous with threshold .5 incoming')
            for pm in list(rationale_dict.keys())[1:]:
                if random.uniform(0,1) < 0.5:
                    while rationale_dict[pm] == chosen:
                        rationale_dict[pm] = random.choice(rationale)


        elif config_name.split('_')[0] == "075":
            print('homogeneous with threshold .75 incoming')
            for pm in list(rationale_dict.keys())[1:]:
                if random.uniform(0,1) < 0.75:
                    while rationale_dict[pm] == chosen:
                        rationale_dict[pm] = random.choice(rationale)

        else:
            print('homogeneous with threshold 0 incoming')

    elif eval(os.environ['HH_heterogeneous']) is True:
        # we are enforcing heterogeneity to a certain degree

        random.shuffle(rationale)
        n = 0
        for pm in list(rationale_dict.keys()):
            rationale_dict[pm] = rationale[n]
            n += 1

        if config_name.split('_')[0] == "025":
            print('heterogeneous runs with threshold .25 incoming')
            n = 1
            for pm in list(rationale_dict.keys())[1:]:
                if random.uniform(0, 1) < 0.25:
                    rationale_dict[pm] = rationale[n]
                else:
                    rationale_dict[pm] = random.choice(rationale)
                n += 1

        elif config_name.split('_')[0] == "05":
            print('heterogeneous runs with threshold .5 incoming')
            n = 1
            for pm in list(rationale_dict.keys())[1:]:
                if random.uniform(0, 1) < 0.5:
                    rationale_dict[pm] = rationale[n]
                else:
                    rationale_dict[pm] = random.choice(rationale)
                n += 1

        elif config_name.split('_')[0] == "075":
            print('heterogeneous runs with threshold .75 incoming')
            n = 1
            for pm in list(rationale_dict.keys())[1:]:
                if random.uniform(0, 1) < 0.75:
                    rationale_dict[pm] = rationale[n]
                else:
                    rationale_dict[pm] = random.choice(rationale)
                n += 1
        else:
            print('heterogeneous runs with threshold 0 incoming')

else:
    print('baseline run incoming')
    for pm in list(rationale_dict.keys()):
        rationale_dict[pm] = random.choice(rationale)

# print(rationale_dict)


PARAMS = {
    'periodicity': periodicity,
    'DD': {
        "initial_demand": {'mean': INITIAL_DEMAND, 'std': INITIAL_DEMAND*0.1, 'max': 2*INITIAL_DEMAND, 'min': INITIAL_DEMAND/2},
        "when": {'mean': 12, 'std': 6, 'max': 24, 'min': 6},
        "increase": {'mean': 0.01, 'std': 0.01, 'max': 0.02, 'min': 0.05},
    },
    'EP': {

    },
    'TP': {

    },
    'BB': {

    },
    'EPM': {
        "PPA_limit": {'mean': 12, 'std': 6, 'max': 24, 'min': 6},
        "COUNTDOWN": {'mean': 6, 'std': 6, 'max': 12, 'min': 3},
        'wallet': {'mean': 9 * 10 ** 9, 'std': 3 * 10 ** 9, 'max': 5 * 10 ** 9, 'min': 1 * 10 ** 9},
        'source': public_source,
        'decision_var': decision_var,
        'instrument': ['auction'],  # ['auction', 'carbon_tax', 'FiT']
        'LSS_thresh': LSS_thresh,
        "public_source": public_source,
        "memory": memory,
        "discount": discount,
        "impatience": impatience,
        "disclosed_thresh": disclosed_thresh,
        "rationale": [rationale_dict['epm']],  # ['green'],
        "past_weight": past_weight
    },
    'DBB': {
        'wallet': {'mean': (9 * 10 ** 9), 'std': (3 * 10 ** 9), 'max': (5 * 10 ** 9), 'min': (1 * 10 ** 9)},
        'source': public_source,
        'decision_var': decision_var,
        'instrument': ['finance'],  # ['finance', 'guarantee']
        'LSS_thresh': LSS_thresh,
        "public_source" : public_source,
        "memory" : memory,
        "discount" : discount,
        "impatience" : impatience,
        "disclosed_thresh" : disclosed_thresh,
        "rationale": [rationale_dict['dbb']],  # ['capacity'],
        "past_weight": past_weight
    },
    'TPM': {
        "boundness_cost": {'mean': 0, 'std': 1, 'max': 0.99, 'min': 0.01},
        'wallet': {'mean': 3 * 10 ** 7, 'std': 3 * 10 ** 7, 'max': 5 * 10 ** 7, 'min': 1 * 10 ** 7},
        'source': public_source,
        'decision_var': decision_var,
        'instrument': ['unbound'],  # ['unbound', 'bound']
        'LSS_thresh': LSS_thresh,
        "public_source": public_source,
        "memory": memory,
        "discount": discount,
        "impatience": impatience,
        "disclosed_thresh": disclosed_thresh,
        "rationale": [rationale_dict['tpm']],  # ['innovation'],
        "past_weight": past_weight
    },
 }

if config_name.split('_')[0] != "1":
    HH_heterogeneous = eval(os.environ['HH_heterogeneous']) #  True if os.environ['HH_heterogeneous'] == 'True' else False
    print(HH_heterogeneous)
    HH_check = os.environ['HH_check']

    PARAMS['HH'] = {'heterogeneous': HH_heterogeneous,
                    'check': HH_check}
