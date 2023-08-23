LSS_thresh = [.25, .5, .75, .1, .9]
decision_var = {'mean': 0, 'std': 1, 'max': 1, 'min': 0}
past_weight = [0.5, 0.25, 0.75, 0.1, 0.9]
public_source = {'mean': 800, 'std': 800, 'max': 2000, 'min': 500}
memory = {'mean': 12, 'std': 12, 'max': 36, 'min': 3}
discount = {'mean': 0.0005, 'std': 0.0005, 'max': 0.0001, 'min': 0.00001}
impatience = {'mean': 3, 'std': 5, 'max': 8, 'min': 1}
disclosed_thresh = [0.01, 0.1, 0.25]

PARAMS = {
    'EP': {

    },
    'TP': {

    },
    'BB': {

    },
    'EPM': {
        "PPA_limit": {'mean': 12, 'std': 6, 'max': 24, 'min': 6},
        "COUNTDOWN": {'mean': 6, 'std': 12, 'max': 24, 'min': 3},
        'wallet': {'mean': 3 * 10 ** 9, 'std': 3 * 10 ** 9, 'max': 5 * 10 ** 9, 'min': 1 * 10 ** 9},
        'source': public_source,
        'decision_var': decision_var,
        'instrument': ['auction'],  # ['auction', 'carbon_tax', 'FiT']
        'LSS_thresh': LSS_thresh,
        "public_source": public_source,
        "memory": memory,
        "discount": discount,
        "impatience": impatience,
        "disclosed_thresh": disclosed_thresh,
        "rationale": ['green'],
        "past_weight": past_weight
    },
    'DBB': {
        'wallet' : {'mean': 3 * 10 ** 9, 'std': 3 * 10 ** 9, 'max': 5 * 10 ** 9, 'min': 1 * 10 ** 9},
        'source': public_source,
        'decision_var': decision_var,
        'instrument': ['finance'],  # ['finance', 'guarantee']
        'LSS_thresh': LSS_thresh,
        "public_source" : public_source,
        "memory" : memory,
        "discount" : discount,
        "impatience" : impatience,
        "disclosed_thresh" : disclosed_thresh,
        "rationale": ['capacity'],
        "past_weight": past_weight
    },
    'TPM': {
        "boundness_cost": {'mean': 0, 'std': 1, 'max': 0.9, 'min': 0.1},
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
        "rationale": ['innovation'],
        "past_weight": past_weight
    }
 }
