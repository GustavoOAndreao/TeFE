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


def random_list(value, percentage=False):

    My_list = [*range(1, value)] if percentage is False else np.arange(0, 1, value).tolist()

    return random.sample(My_list, random.choice([*range(1, len(My_list)-1)]))


def make_ep(name,
            wallet=None,
            tolerance=None,
            source=None,
            decision_var=None,
            LSS_thresh=None,
            impatience=None,
            memory=None,
            discount=None,
            past_weight=None,
            current_weight=None,
            periodicity=None
            ):

    """

    :param periodicity:
    :param name:
    :param wallet:
    :param tolerance:
    :param source:
    :param decision_var:
    :param LSS_thresh:
    :param impatience:
    :param memory:
    :param discount:
    :param past_weight:
    :param current_weight:
    :return:
    """
    if wallet is None:
        wallet = random.uniform(0, 2) * 10 ** 8
    if decision_var is None:
        decision_var = random.uniform(0, 1)
    if current_weight is None:
        current_weight = random_list(0.25, percentage=True)
    if past_weight is None:
        past_weight = random_list(0.25, percentage=True)
    if memory is None:
        memory = random_list(24)
    if impatience is None:
        impatience = random_list(5)
    if LSS_thresh is None:
        LSS_thresh = random_list(0.25, percentage=True)
    if source is None:
        source = [{0: random.uniform(0, 1000)}, {1: random.uniform(0, 1000)}, {2: random.uniform(0, 1000)}]
        random.shuffle(source)
    if tolerance is None:
        tolerance = random_list(24)
    if periodicity is None:
        periodicity = int(random.uniform(1, 12))
    if discount is None:
        discount = [0.001, 0.005, 0.01]
        random.shuffle(discount)

    ic(name, wallet, tolerance, source, decision_var, LSS_thresh, impatience, memory, discount, past_weight,
       current_weight, periodicity)

    return EP(env=env,
              name=name,
              wallet=wallet,
              portfolio_of_plants={},
              portfolio_of_projects={},
              periodicity=periodicity,
              tolerance=tolerance,
              last_acquisition_period=0,
              source=source,
              decision_var=decision_var,
              LSS_thresh=LSS_thresh,
              impatience=impatience,
              memory=memory,
              discount=discount,
              past_weight=past_weight,
              current_weight=current_weight)


TECHNOLOGY_PROVIDERS = [TP(env=env,
                           name='TP_0',
                           wallet=random.uniform(0, 1)*10**6,
                           capacity=random.uniform(10, 20)*10**3,
                           Technology={'name': 'TP_0',
                                       "green": True,
                                       "source": 1,
                                       "source_name": 'wind',
                                       "CAPEX": config.WIND['CAPEX'],
                                       'OPEX': config.WIND['OPEX'],
                                       "dispatchable": False,
                                       "transport": False,
                                       "CF": config.WIND['CF'],
                                       "MW": config.WIND['MW'],
                                       "lifetime": config.WIND['lifetime'],
                                       'building_time': config.WIND['building_time'],
                                       'last_radical_innovation': 0,
                                       'last_marginal_innovation': 0,
                                       'emissions': 0,
                                       'avoided_emissions': 250},
                           RnD_threshold=random.uniform(2, 5),
                           capacity_threshold=random.uniform(2, 5),
                           decision_var=random.uniform(0, 1),
                           cap_conditions={},
                           impatience=random.sample([1, 3, 5], 2),
                           past_weight=random.sample([0.25, 0.5, 0.75], 2),
                           LSS_thresh=random.sample([0.25, 0.1, 0.5, 0.75], 2),
                           memory=random.sample([12, 6, 24], 2),
                           discount=random.sample([0.005, 0.001, 0.002], 2),
                           strategy=random.sample(['capacity', "innovation"], 2),
                           starting_tech_age=3),
                        TP(env=env,
                           name='TP_1',
                           wallet=3*10**5,
                           capacity=20000,
                           Technology={'name': 'TP_1',
                                       "green": True,
                                       "source": 2,
                                       "source_name": 'solar',
                                       "CAPEX": config.SOLAR['CAPEX'],
                                       'OPEX': config.SOLAR['OPEX'],
                                       "dispatchable": False,
                                       "transport": True,
                                       "CF": config.SOLAR['CF'],
                                       "MW": config.SOLAR['MW'],
                                       "lifetime": config.SOLAR['lifetime'],
                                       'building_time': config.SOLAR['building_time'],
                                       'last_radical_innovation': 0,
                                       'last_marginal_innovation': 0,
                                       'emissions': 0,
                                       'avoided_emissions': 50},
                           RnD_threshold=random.uniform(2, 5),
                           capacity_threshold=random.uniform(2, 5),
                           decision_var=random.uniform(0, 1),
                           cap_conditions={},
                           impatience= random.sample([1, 3, 5], 2),
                           past_weight=random.sample([0.25, 0.5, 0.75], 2),
                           LSS_thresh=random.sample([0.25, 0.1, 0.5, 0.75], 2),
                           memory=random.sample([12, 6, 24], 2),
                           discount=random.sample([0.005, 0.001, 0.002], 2),
                           strategy=random.sample(['capacity', "innovation"], 2),
                           starting_tech_age=1
                           )]

config.TP_NUMBER = len(TECHNOLOGY_PROVIDERS)

ENERGY_PRODUCERS = [make_ep('EP_00'),
                    make_ep('EP_01'),
                    make_ep('EP_02'),
                    make_ep('EP_03')]

config.EP_NUMBER = len(ENERGY_PRODUCERS)

PRIVATE_BANK = []

config.BB_NUMBER = len(PRIVATE_BANK)

DEMAND = Demand(env=env,
                initial_demand=INITIAL_DEMAND,
                when=1,
                increase=INITIAL_DEMAND * random.uniform(0.001, 0.005))

POLICY_MAKERS = [DBB(env=env,
                     name='BNDES',
                     wallet=random.uniform(0, 5) * 10 ** 9,
                     instrument=['finance'],  # ['finance', 'guarantee']
                     source=random.sample([{2: random.uniform(0, 1000)}, {1: random.uniform(0, 1000)}], 2),
                     decision_var=random.uniform(0, 1),
                     LSS_thresh=random.sample([.1, .5, .75], 3),
                     past_weight=random.sample([0.1, 0.5, 0.25, 0.75, 0.9], 3),
                     memory=random.sample([6, 12, 24], 3),
                     discount=random.sample([0.001, 0.005, 0.01], 2),
                     policies=[],
                     impatience=random.sample([1, 3, 4], 3),
                     disclosed_thresh=random.sample([0.1, 0.25, 0.5], 3),
                     rationale=['capacity']),
                 EPM(env=env,
                     wallet=random.uniform(0, 5) * 10 ** 9,
                     PPA_expiration=350,  # int(random.uniform(1, 20))  * 12,
                     PPA_limit=12,
                     COUNTDOWN=int(random.uniform(3, 12)),
                     decision_var=random.uniform(0, 1),
                     auction_capacity=random.uniform(0.5, 1) * config.INITIAL_DEMAND,
                     instrument=['FiT'],  # random.sample(['auction', 'carbon_tax', 'FiT'], 3),  # ['auction', 'carbon_tax', 'FiT'],
                     source=random.sample([{2: random.uniform(0, 1000)}, {1: random.uniform(0, 1000)}], 2),
                     LSS_thresh=random.sample([.1, .5, .75], 3),
                     impatience=random.sample([1, 3, 4], 3),
                     disclosed_thresh=random.sample([0.1, 0.25, 0.5], 3),
                     past_weight=[0.5, 0.25, 0.75],
                     memory=random.sample([6, 12, 24], 3),
                     discount=random.sample([0.001, 0.005, 0.01], 2),
                     policies=[],
                     rationale=['green']),
                 TPM(env=env,
                     wallet=random.uniform(0, 10) * 10 ** 9,
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
                     rationale=['innovation'])]

config.PUB_NUMBER = len(POLICY_MAKERS)
