import random
import feather
import simpy
import numpy as np
import pandas as pd
import math as math
import scipy
import networkx as nx
import time
from matplotlib import pyplot as plt
import itertools
import uuid
import pdb
import statistics
from icecream import ic
from statistics import median, mean
from collections import Counter

from commons import *
import config
from config import seed

random.seed(seed)


class TP(object):
    def __init__(self, env, name, wallet, capacity, Technology, RnD_threshold, capacity_threshold,
                 decision_var, cap_conditions, impatience, past_weight, LSS_thresh, memory, discount,
                 strategy, starting_tech_age):
        """
        Technology provider. Provides productive capacity for energy producers to produce energy (molecules or
        electricity)

        :param env:
        :param name: name of the agent, is a string, normally something like TP_01
        :param wallet: wallet or reserves, or savings, etc. How much of its resource does the agent have? is a number
        :param capacity: productive capacity of the agent, is a number used to determine the CAPEX of its technology
        :param Technology: the technology dict. Is a dictionary with the characteristics of the technology
        :param RnD_threshold: what is the threshold of R&D expenditure necessary to innovate? Is a number
        :param capacity_threshold: what is the threshold of investment in productive capacity in order to start
        decreasing CAPEX costs?
        :param dd_source: This, my ganzirosis, used to be the Tactics. It is the first of the ranked dictionaries. It
        goes a little sumthing like dis: dd = {'current' : 2, 'ranks' : {0: 3500, 1: 720, 2: 8000}}.
        With that we have the current decision for the variable or thing and on the ranks we have the score for
        :param decision_var: this is the value of the decision variable. Is a number between -1 and 1. Notice: private
        entities have, by default, their decision_var equal to their disclosed_var
        :param dd_kappas: this is the kappa, follows the current ranked dictionary
        :param dd_qual_vars: this tells the agent the qualitative variables in a form
        {0 : 'name of the zeroth variable', 1 : 'name of the first variable', 2 : 'name of the second variable'}
        :param dd_backwardness: also a ranked dictionary, this one tells the backwardness of agents
        :param dd_avg_time: also a ranked dictionary, this one tells the average time for deciding if change is necessary
        :param dd_discount: discount factor. Is a ranked dictionary
        :param cap_conditions: there are the cap conditions for this technology, being a dictionary following this
        example {'char' : 'CAPEX', 'cap' : 20000, 'increase' : 0.5}
        :param strategy: the initial strategy of the agent, can be to reivnest into producetive capacity or R&D
        :param dd_strategies: initial strategy for the technology provider. Is a ranked dictionary
        """

        self.env = env
        self.name = name
        self.genre = 'TP'  # genre, we do not use type, because type is a dedicated command of python, is also a string
        self.Technology = Technology
        # self.subgenre = Technology['source']  # subgenre or source, we used to use subgenre a lot, now it's kind of a
        # legacy. Is a number, 1 is wind, 2 is solar, 4 is biomass and 5 is hydrogen. 0 and 3 are not used because they
        # are the fossil options
        self.wallet = wallet
        self.profits = 0  # profits of the agent at that certain period, is a number
        self.capacity = capacity
        self.RandD = 0  # how much money was put into R&D? Is a number
        # self.EorM = Technology['EorM']  # does the agent use electricity or molecules? is a string (either 'E' or 'M')
        self.innovation_index = 0  # index of innovation. Kind of a legacy, was used to analyze innovation
        self.self_NPV = {'value': 0, 'MWh': 0}  # The NPV of a unit of investment. Is a dictionary:
        # e.g. self_NPV={'value' : 2000, 'MWh' : 30}
        self.RnD_threshold = RnD_threshold
        self.capacity_threshold = capacity_threshold
        self.dd_profits = {0: 0, 1: 0, 2: 0} # if Technology['EorM'] == 'E' else {3: 0, 4: 0, 5: 0}  # same as profits,
        # but as dict. Makes accounting faster and simpler
        # self.dd_source = dd_source
        self.decision_var = decision_var
        self.verdict = "keep"  # this is the verdict variable. It can be either 'keep', 'change' or 'add'
        # self.dd_kappas = dd_kappas
        # self.dd_qual_vars = dd_qual_vars
        # self.dd_backwardness = dd_backwardness
        # self.dd_avg_time = dd_avg_time
        # self.dd_discount = dd_discount
        self.cap_conditions = cap_conditions
        self.capped = False  # boolean variable to make the capping easier
        # self.strategy = strategy
        # self.dd_strategies = dd_strategies
        self.impatience = impatience
        self.past_weight = past_weight
        self.LSS_thresh = LSS_thresh
        self.memory = memory
        self.discount = discount
        self.strategy = strategy
        self.LSS_tot = 0
        self.innovation_index = 0
        self.shareholder_money = 0
        self.true_innovation_index = 0

        strikables_dict = {'impatience': impatience,
                           'LSS_thresh': LSS_thresh,
                           'memory': memory,
                           'discount': discount,
                           'strategy': strategy,
                           'past_weight': past_weight
                           }

        self.strikables_dict = strikable_dicting(strikables_dict)
        self.starting_tech_age = starting_tech_age
        self.prod_cap_pct = [0, 1]
        self.radical = 0
        self.marginal = 0
        self.min_price = config.STARTING_PRICE

        # ic(name, wallet, capacity, Technology, RnD_threshold, capacity_threshold,decision_var, cap_conditions, impatience, past_weight, LSS_thresh, memory, discount,strategy, starting_tech_age)

        self.action = env.process(run_TP(
            self.name,
            self.genre,
            self.wallet,
            self.capacity,
            self.Technology,
            self.profits,
            self.RnD_threshold,
            self.RandD,
            self.capacity_threshold,
            self.decision_var,
            self.cap_conditions,
            self.verdict,
            self.self_NPV,
            self.capped,
            self.impatience,
            self.past_weight,
            self.LSS_thresh,
            self.memory,
            self.discount,
            self.strategy,
            self.LSS_tot,
            self.innovation_index,
            self.shareholder_money,
            self.true_innovation_index,
            self.strikables_dict,
            self.dd_profits,
            self.starting_tech_age,
            self.prod_cap_pct,
            self.radical,
            self.marginal,
            self.min_price
        ))


def run_TP(name,
           genre,
           wallet,
           capacity,
           Technology,
           profits,
           RnD_threshold,
           RandD,
           capacity_threshold,
           decision_var,
           cap_conditions,
           verdict,
           self_NPV,
           capped,
           impatience,
           past_weight,
           LSS_thresh,
           memory,
           discount,
           strategy,
           LSS_tot,
           innovation_index,
           shareholder_money,
           true_innovation_index,
           strikables_dict,
           dd_profits,
           starting_tech_age,
           prod_cap_pct,
           radical,
           marginal,
           min_price
):
    CONTRACTS, MIX, AGENTS, AGENTS_r, TECHNOLOGIC, TECHNOLOGIC_r, r, AMMORT, rNd_INCREASE, RADICAL_THRESHOLD, env = config.CONTRACTS, config.MIX, config.AGENTS, config.AGENTS_r, config.TECHNOLOGIC, config.TECHNOLOGIC_r, config.r, config.AMMORT, config.rNd_INCREASE, config.RADICAL_THRESHOLD, config.env  # globals

    while True:

        ###############################################################################################################
        #                                                                                                             #
        #                            Before anything, we must the current values of each of                           #
        #                               the dictionaries that we use and other variables                              #
        #                                                                                                             #
        ###############################################################################################################

        _LSS_thresh = LSS_thresh[0] if env.now == 0 else AGENTS[env.now - 1][name]['LSS_thresh'][0]
        _memory = memory[0] if env.now == 0 else AGENTS[env.now - 1][name]['memory'][0]
        _discount = discount[0] if env.now == 0 else AGENTS[env.now - 1][name]['discount'][0]
        _impatience = impatience[0] if env.now == 0 else AGENTS[env.now - 1][name]['impatience'][0]
        _past_weight = past_weight[0] if env.now == 0 else AGENTS[env.now - 1][name]['past_weight'][0]
        _strategy = strategy[0] if env.now == 0 else AGENTS[env.now - 1][name]['strategy'][0]
        value = decision_var
        LSS_tot = LSS_tot if env.now == 0 else AGENTS[env.now - 1][name]['LSS_tot']
        profits = 0  # in order to get the profits of this period alone

        RandD = int(RandD)
        RnD_threshold = int(RnD_threshold)

        # radical = 0
        # marginal = 0

        ###############################################################################################################
        #                                                                                                             #
        #                           First, the Technology provider closes any new deals and                           #
        #                                               collect profits                                               #
        #                                                                                                             #
        ###############################################################################################################

        dd_profits[Technology['source']] = 0

        if env.now > 0 and len(CONTRACTS[env.now - 1]) > 0:
            for _ in CONTRACTS[env.now - 1]:
                i = CONTRACTS[env.now - 1][_]
                if i['receiver'] == name and i['status'] == 'payment':
                    wallet += i['value']
                    profits += i['value']
                    dd_profits[Technology['source']] += i['value']
                    prod_cap_pct[1] += i['value']

        ###############################################################################################################
        #                                                                                                             #
        #                           Now, the Technology provider gets any incentive that is                           #
        #                                               available to it                                               #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 0:
            for _ in CONTRACTS[env.now - 1]:
                i = CONTRACTS[env.now - 1][_]
                if i['receiver'] == name and i['sender'] == 'TPM':
                    incentive = i['value']
                    prod_cap_pct[1] += incentive
                    if i['bound'] == 'capacity':
                        capacity += incentive
                        prod_cap_pct[0] += incentive
                    elif i['bound'] == 'innovation':
                        RandD += incentive
                    else:
                        wallet += incentive

        ###############################################################################################################
        #                                                                                                             #
        #                        If it is the end of the year, then the TP shares its profits                         #
        #                                            with its shareholders                                            #
        #                                                                                                             #
        ###############################################################################################################

        if env.now % 12 == 0 and env.now > 0 and wallet > 0:
            profits_to_shareholders = wallet*(1-value)
            wallet -= profits_to_shareholders
            shareholder_money += profits_to_shareholders

        ###############################################################################################################
        #                                                                                                             #
        #                                The TP then has to 1) adjust the base capex to                               #
        #                                the productive capacity (if the technology is                                #
        #                         non-transportable; 2) change the strategy if the verdict was                        #
        #                         to change it; 3) spend the available money into imitation,                          #
        #                        innovation or productive capacity; 4) check if the TP reached                        #
        #                            the threshold of innovation/imitation and change the                             #
        #                           Technology dictionary if it reached that; 5) update the                           #
        #                           global TECHNOLOGY dictionary; and lastly 6) do the self                           #
        #                                  NPV entry for the AGENTS dictionary later.                                 #
        #                                  8) Oh, it must also the capping process                                    #
        #                                                                                                             #
        ###############################################################################################################

        """ 1)  we have to get the base-CAPEX and adjust it to the productive capacity of the TP (only if its
         technology is not transportable) """
        if env.now == 0:
            """ if we are in the first period, then we have to get the starting CAPEX and tell the TP that this
             is his base capex, because there is no base_capex already """
            i = Technology['CAPEX']
            Technology.update({"base_CAPEX": i})

        if Technology['transport'] is False:
            """ if the technology is not transportable, then the productive capacity impacts on the CAPEX """
            base_capex = Technology['base_CAPEX']
            """ we have to produce the actual CAPEX, with is the base_CAPEX multiplied by euler's number to the
             power of the ratio of how many times the base capex is greater than the capacity itself multiplied
              by the threshold of capacity"""
            new_capex = max(min(
                base_capex, base_capex ** capacity_threshold / capacity
            ), base_capex/(2 ** (env.now // 120)))

            """if base_capex != new_capex != Technology['CAPEX']:
                print(name, env.now, new_capex, base_capex, base_capex ** capacity_threshold, capacity)"""

            Technology.update({
                'CAPEX': new_capex
            })
        else:
            """ the technology is transportable (e.g. solar panels)"""
            Technology.update({Technology['CAPEX'] : Technology["base_CAPEX"]})

        """3) now, if the TP has money, it will spend it on either capacity, imitation or innovation"""
        # ic(name, wallet, _strategy, value, env.now)
        if wallet > 0 and value > 0:
            wallet -= wallet * value
            """ having money, the TP will spend a portion (given by the value) of its wallet on something """
            if _strategy == 'innovation':
                """ if the strategy is to do R&D, then it will do R&D"""
                RandD += wallet * value
            elif _strategy == 'capacity':
                """ if not, then it will spend on productive capacity"""
                capacity += wallet * value
                prod_cap_pct[0] += wallet * value
                # print(name, 'got', wallet * value, 'more capacity to its roster, and now capacity is', capacity)

        """4) we have to check if the TP reached the threshold for innovation or imitation"""
        # a = 0
        # print("R&D and capacity", env.now, value,  name, RandD, capacity, RnD_threshold - RandD)
        RnD_threshold *= (1 - _discount)
        # innovation_index *= (1 - _discount)
        # print(env.now, name, "RandD", RandD, "RnD_threshold", RnD_threshold)
        if RandD > RnD_threshold and env.now > config.FUSS_PERIOD:  # and (strategy == 'innovation' or strategy == 'imitation'):
            """ then we get the 'a' which can either be a poisson + normal for innovation, or a simple binomial.
             Values above or equal (for the imitation) 1 indicate that innovation or imitation occured """
            # print('radical', name, radical)
            tech_age = starting_tech_age + radical # + 0.01 * marginal # + innovation_index * 0.01
            a = np.random.poisson(1 + 1/tech_age) + np.random.normal(0, 1)
            # a = np.random.poisson(1 / tech_age) + np.random.normal(0, 1 - 1/tech_age)
            # print(env.now, 1 / tech_age, name, a, 'radical', radical)

            if a > 1:
                # print('innovation was', env.now, name, a, RnD_threshold , RandD)  # , RandD > (RnD_threshold))
                innovation_index += a
                """ we are dealing with innovation """
                true_innovation_index += a
                RnD_threshold *= a  # ** (1 + rNd_INCREASE)
                # print('RnD_threshold', math.e ** ((1 - 1/tech_age) + rNd_INCREASE),  a)
                """ we have to check where did the innovation occur"""
                what_on = random.choice(['base_CAPEX', 'OPEX'])  # , 'MW'])
                """ if innovation ocurred then we multiply it"""
                if what_on == 'MW':
                    new_what_on = a * Technology[what_on]
                else:
                    new_what_on = (1 / a) * Technology[what_on]
                Technology.update({what_on: new_what_on})
                # print(name, ' got a new ', what_on, ' by ', a, ' at time ', env.now)

                if a > RADICAL_THRESHOLD:
                    """ if we reached over the radical innovation threshold we have to signal it """
                    radical_or_not = 'last_radical_innovation'
                    radical += 1
                else:
                    """ we did not reach over that threshold, so it was a marginal innovation """
                    radical_or_not = 'last_marginal_innovation'
                    marginal += 1
                Technology.update({
                    radical_or_not: env.now
                })

                """ if we reached the threshold, then we to set the bar of the RnD """
                RnD_threshold += random.uniform(0, RandD)

        """6) lastly, we have to get the self.NPV of its current technology"""
        if env.now > 0:
            i = TECHNOLOGIC[env.now - 1][name]
            price = weighting_FF(env.now - 1, 'price', 'MWh', MIX)
            self_NPV.update({
                'value': npv_generating_FF(r, i['lifetime'], 1, i['MW'], i['building_time'], i['CAPEX'], i['OPEX'],
                                           price[i['source']], i['CF'], AMMORT), 'MWh': i['MW']
            })

        """ 8) we must also check if the capping process is on"""

        Technology['min_price'] = (Technology['OPEX'] + (Technology['CAPEX'] / Technology['lifetime'])) / (Technology['MW'] * 24 * 30 * Technology['CF'])
        # print(Technology['min_price'])

        if len(cap_conditions) > 0:
            now = 0
            if len(MIX[env.now - 1]) > 0:
                # if there is no capping, we must first make sure that it has not started
                now = finding_FF(MIX[env.now - 1], 'MW', 'sum', {'EP': name})['value']
                if now > cap_conditions['cap']:
                    capped = True

                else:
                    capped = False

            if capped is True:
                # capping process is on, so we have to make sure that the capacity increased
                previous = finding_FF(MIX[env.now - 2], 'MW', 'sum', {'EP': name})['value']

                if now > previous:
                    # the capacity increased, so we have to apply the capping conditions
                    Technology.update({
                        cap_conditions['char']: Technology[cap_conditions['char']] * (1 + cap_conditions['increase'])
                    })

        """5) we also have to update the TECHNOLOGIC dictionary for the next period with the current technology
         (applying any changes if there were any whatsover)"""
        TECHNOLOGIC[env.now].update({name: Technology})

        min_price = Technology['min_price']

        ###############################################################################################################
        #                                                                                                             #
        #                                 And then, the TP will decide what to do next                                #
        #                                                                                                             #
        ###############################################################################################################
        if env.now > 0:
            decision_var = max(0, min(1, private_deciding_FF(name)))
            decisions = evaluating_FF(name)
            verdict = decisions['verdict']

        ###############################################################################################################
        #                                                                                                             #
        #                           Before leaving, the agent must update the outside world                           #
        #                                                                                                             #
        ###############################################################################################################

        update = {"name": name,
                  "genre": genre,
                  "wallet": wallet,
                  "capacity": capacity,
                  "Technology": Technology,
                  "profits": profits,
                  "RnD_threshold": RnD_threshold,
                  "RandD": RandD,
                  "capacity_threshold": capacity_threshold,
                  "decision_var": decision_var,
                  "cap_conditions": cap_conditions,
                  "verdict": verdict,
                  "self_NPV": self_NPV,
                  "capped": capped,
                  "impatience": impatience.copy(),
                  "past_weight": past_weight,
                  "LSS_thresh": LSS_thresh,
                  "memory": memory,
                  "discount": discount,
                  "strategy": strategy,
                  "innovation_index": innovation_index,
                  "shareholder_money": shareholder_money,
                  "true_innovation_index": true_innovation_index,
                  "LSS_tot": LSS_tot,
                  "prod_cap_pct": prod_cap_pct,
                  "strikables_dict": strikables_dict,
                  'PCT': prod_cap_pct[0]/prod_cap_pct[1],
                  "radical": radical,
                  "marginal": marginal,
                  "min_price": min_price
                  }

        if env.now > 1:
            update['impatience'][0] = max(1, update['impatience'][0] + decisions['impatience_increase'])
            update["LSS_weak"] = AGENTS[env.now - 1][name]["LSS_weak"] + abs(
                decision_var - AGENTS[env.now - 1][name]['decision_var'])
        else:
            update["LSS_weak"] = 0

        AGENTS[env.now][name] = update.copy()
        if env.now > 1:
            post_evaluating_FF(decisions['strikes'], verdict, name, strikables_dict)
            # print(env.now, name, LSS_tot, update["LSS_weak"], value)
            # profits_dedicting_FF(name)

        yield env.timeout(1)


class TPM(object):
    def __init__(self, env, wallet, boundness_cost, instrument, source, decision_var, LSS_thresh, impatience,
                 disclosed_thresh, past_weight, memory, discount, policies, rationale, periodicity):
        self.env = env
        self.genre = 'TPM'
        self.subgenre = 'TPM'
        self.name = 'TPM'
        self._wallet = wallet
        self.boundness_cost = boundness_cost
        self.instrument = instrument
        self.source = source
        self.decision_var = decision_var
        self.disclosed_var = decision_var
        self.verdict = 'keep'
        self.LSS_thresh = LSS_thresh
        self.impatience = impatience
        self.disclosed_thresh = disclosed_thresh
        self.past_weight = past_weight
        self.memory = memory
        self.discount = discount
        self.policies = policies
        self.rationale = rationale

        strikables_dict = {'impatience': impatience,
                           'LSS_thresh': LSS_thresh,
                           'memory': memory,
                           'discount': discount,
                           "source": source,
                           "disclosed_thresh": disclosed_thresh,
                           "past_weight": past_weight,
                           'instrument': instrument,
                           'rationale': rationale,
                           'periodicity': periodicity}

        self.strikables_dict = strikable_dicting(strikables_dict)
        
        self.periodicity = periodicity

        self.LSS_tot = 0

        self.action = env.process(run_TPM(self.genre,
                                          self.subgenre,
                                          self.name,
                                          self._wallet,
                                          self.boundness_cost,
                                          self.instrument,
                                          self.source,
                                          self.decision_var,
                                          self.disclosed_var,
                                          self.verdict,
                                          self.LSS_thresh,
                                          self.impatience,
                                          self.disclosed_thresh,
                                          self.past_weight,
                                          self.memory,
                                          self.discount,
                                          self.policies,
                                          self.rationale,
                                          self.strikables_dict,
                                          self.LSS_tot,
                                          self.periodicity))


def run_TPM(genre,
            subgenre,
            name,
            _wallet,
            boundness_cost,
            instrument,
            source,
            decision_var,
            disclosed_var,
            verdict,
            LSS_thresh,
            impatience,
            disclosed_thresh,
            past_weight,
            memory,
            discount,
            policies,
            rationale,
            strikables_dict,
            LSS_tot,
            periodicity):
    CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, POLICY_EXPIRATION_DATE, AMMORT, INSTRUMENT_TO_SOURCE_DICT, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.POLICY_EXPIRATION_DATE, config.AMMORT, config.INSTRUMENT_TO_SOURCE_DICT, config.env

    while True:

        ###############################################################################################################
        #                                                                                                             #
        #                                     First, the TPM checks if it got an                                      #
        #                                   strike, adds or changes its main policy                                   #
        #                                                                                                             #
        ###############################################################################################################

        _LSS_thresh = LSS_thresh[0] if env.now == 0 else AGENTS[env.now - 1][name]['LSS_thresh'][0]
        _past_weight = past_weight[0] if env.now == 0 else AGENTS[env.now - 1][name]['past_weight'][0]
        _source = list(source[0].keys())[0] if env.now == 0 else list(AGENTS[env.now - 1][name]['source'][0].keys())[0]
        _memory = memory[0] if env.now == 0 else AGENTS[env.now - 1][name]['memory'][0]
        _discount = discount[0] if env.now == 0 else AGENTS[env.now - 1][name]['discount'][0]
        _impatience = impatience[0] if env.now == 0 else AGENTS[env.now - 1][name]['impatience'][0]
        _rationale = rationale[0] if env.now == 0 else AGENTS[env.now - 1][name]['rationale'][0]
        _instrument = instrument[0] if env.now == 0 else AGENTS[env.now - 1][name]['instrument'][0]
        _periodicity = periodicity[0] if env.now == 0 else AGENTS[env.now - 1][name]['periodicity'][0]
        LSS_tot = LSS_tot if env.now == 0 else AGENTS[env.now - 1][name]['LSS_tot']
        value = disclosed_var
        # wallet = _wallet if not env.now > 0 and env.now % 12 == 0 else AGENTS[0][name]['wallet'] # the budget is monthly
        wallet = _wallet  # if not env.now % 12 == 0 else 0
        # wallet += AGENTS[0][name]['wallet'] if env.now > 0 and env.now % 12 == 0 else _wallet # the budget is monthly

        ###############################################################################################################
        #                                                                                                             #
        #                                    Now, the TPM gives out the incentives                                    #
        #                                                                                                             #
        ###############################################################################################################

        policy_pool = [{'instrument': _instrument,
                        'source': _source,
                        'budget': value * wallet}]

        if len(policies) > 0:
            policy_pool.append(policies)  # with this we a temporary list with first the current policy and afterwards
            # all the other policies

        if env.now >= 1:
            for entry in policy_pool:

                instrument = entry['instrument']
                chosen_source = entry['source']
                budget = entry['budget']
                value = disclosed_var

                if instrument == 'unbound':
                    bound3 = 'unbound'
                else:
                    bound3 = _rationale if 'rationale' not in entry else entry['rationale']

                firms = []
                for _ in AGENTS[env.now - 1]:
                    i = AGENTS[env.now - 1][_]
                    if i['genre'] == 'TP' and i['Technology']['source'] == chosen_source:
                        firms.append(_)
                if len(firms) > 0 and budget>0 and value>0:
                    """ we have to be certain that there are companies to be incentivised and now divides the possible 
                    incentive by the number of companies """
                    # print('incentivised_firms', incentivised_firms)
                    incentive = budget / len(
                        firms) if instrument == 'unbound' else budget * (1 - boundness_cost) / len(firms)
                    """ and now we give out the incentives"""
                    for TP in firms:
                        code = uuid.uuid4().int
                        CONTRACTS[env.now].update({
                            code: {
                                'sender': name,
                                'receiver': TP,
                                'status': 'payment',
                                'bound': bound3,
                                'value': incentive}})
                    wallet -= budget

            # value = disclosed_var

        # _wallet += wallet # all non-used budget is kept for next month

        if env.now > 1:
            add_source = source_reporting_FF(name, _past_weight)
            for entry in range(len(source) - 1):
                source[entry][list(source[entry].keys())[0]] *= (1 - _discount)
                source[entry][list(source[entry].keys())[0]] += add_source[list(source[entry].keys())[0]]

        ###############################################################################################################
        #                                                                                                             #
        #                           Now, the TPM analyses what happened to the system due to                          #
        #                                              its intervention                                               #
        #                                                                                                             #
        ###############################################################################################################
            
        if env.now > 2 and env.now % _periodicity == 0:
            add_source = source_reporting_FF(name, _past_weight)
            for entry in range(len(source) - 1):
                source[entry][list(source[entry].keys())[0]] *= (1 - _discount) ** _periodicity
                source[entry][list(source[entry].keys())[0]] += add_source[list(source[entry].keys())[0]]

        ###############################################################################################################
        #                                                                                                             #
        #                                And then, the TPM will decide what to do next                                #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 0:
            decision_var = max(0, min(1, public_deciding_FF(name)))  # if env.now % _periodicity == 0 else decision_var
            disclosed_var = thresholding_FF(_LSS_thresh, disclosed_var, decision_var)
            decisions = evaluating_FF(name) if env.now % _periodicity == 0 else {'verdict': 'keep', 'strikes': False, 'impatience_increase': 0}
            verdict = decisions['verdict']

        ###############################################################################################################
        #                                                                                                             #
        #                           Before leaving, the agent must uptade the outside world                           #
        #                                                                                                             #
        ###############################################################################################################

        update = {
            "genre": genre,
            "subgenre": subgenre,
            "name": name,
            "wallet": wallet,
            "instrument": instrument,
            "source": source,
            "decision_var": decision_var,
            "disclosed_var": disclosed_var,
            "verdict": verdict,
            "LSS_thresh": LSS_thresh,
            "impatience": impatience.copy(),
            "disclosed_thresh": disclosed_thresh,
            "past_weight": past_weight,
            "memory": memory,
            "discount": discount,
            "policies": policies,
            "rationale": rationale,
            "current_state": current_stating_FF(_rationale),
            "LSS_tot": LSS_tot,
            "strikables_dict": strikables_dict,
            "periodicity": periodicity
        }

        if env.now > 1:
            update['impatience'][0] = max(1, update['impatience'][0] + decisions['impatience_increase'])
            update["LSS_weak"] = AGENTS[env.now - 1][name]["LSS_weak"] + abs(
                decision_var - AGENTS[env.now - 1][name]['decision_var'])
        else:
            update["LSS_weak"] = 0

        AGENTS[env.now][name] = update.copy()
        if env.now > 1:
            post_evaluating_FF(decisions['strikes'], verdict, name, strikables_dict)
            # print(env.now, name, LSS_tot, update["LSS_weak"], value)

        yield env.timeout(1)


class EPM(object):
    def __init__(self, env, wallet, PPA_expiration, PPA_limit, COUNTDOWN, decision_var, auction_capacity, instrument, source, LSS_thresh, impatience, disclosed_thresh, past_weight, memory, discount, policies, rationale, periodicity):
        self.env = env
        self.genre = 'EPM'
        self.subgenre = 'EPM'
        self.name = 'EPM'
        self.wallet = wallet
        self.PPA_expiration = PPA_expiration
        self.PPA_limit = PPA_limit
        self.auction_countdown = 0
        self.auction_time = False
        self.COUNTDOWN = COUNTDOWN
        # self.dd_policy = dd_policy
        # self.dd_source = dd_source
        self.decision_var = decision_var
        self.disclosed_var = decision_var
        self.verdict = 'keep'
        # self.dd_kappas = dd_kappas
        # self.dd_qual_vars = dd_qual_vars
        # self.dd_backwardness = dd_backwardness
        # self.dd_avg_time = dd_avg_time
        # self.dd_discount = dd_discount
        # self.policies = policies
        # self.dd_index = dd_index
        self.index_per_source = {1: 0, 2: 0, 4: 0, 5: 0}
        # self.dd_eta = dd_eta
        # self.dd_ambition = dd_ambition
        # self.dd_target = dd_target
        # self.dd_rationale = dd_rationale
        self.auction_capacity = auction_capacity
        # self.dd_SorT = dd_SorT

        self.instrument = instrument
        self.source = source

        self.LSS_thresh = LSS_thresh
        self.impatience = impatience
        self.disclosed_thresh = disclosed_thresh
        self.past_weight = past_weight
        self.memory = memory
        self.discount = discount
        self.policies = policies
        self.rationale = rationale
        self.periodicity = periodicity

        self.LSS_tot = 0
        strikables_dict = {'impatience': impatience,
                           'LSS_thresh': LSS_thresh,
                           'source': source,
                           'past_weight': past_weight,
                           'memory': memory,
                           'discount': discount,
                           'disclosed_thresh': disclosed_thresh,
                           'rationale': rationale,
                           'instrument': instrument,
                           'periodicity': periodicity,
                           }

        self.strikables_dict = strikable_dicting(strikables_dict)

        self.action = env.process(run_EPM(self.genre,
                                          self.subgenre,
                                          self.name,
                                          self.wallet,
                                          self.PPA_expiration,
                                          self.PPA_limit,
                                          self.auction_countdown,
                                          self.auction_time,
                                          self.COUNTDOWN,
                                          self.decision_var,
                                          self.disclosed_var,
                                          self.verdict,
                                          self.index_per_source,
                                          self.auction_capacity,
                                          self.instrument,
                                          self.source,
                                          self.LSS_thresh,
                                          self.impatience,
                                          self.disclosed_thresh,
                                          self.past_weight,
                                          self.memory,
                                          self.discount,
                                          self.policies,
                                          self.rationale,
                                          self.LSS_tot,
                                          self.strikables_dict, 
                                          self.periodicity))


def run_EPM(genre,
            subgenre,
            name,
            wallet,
            PPA_expiration,
            PPA_limit,
            auction_countdown,
            auction_time,
            COUNTDOWN,
            decision_var,
            disclosed_var,
            verdict,
            index_per_source,
            auction_capacity,
            instrument,
            source,
            LSS_thresh,
            impatience,
            disclosed_thresh,
            past_weight,
            memory,
            discount,
            policies,
            rationale,
            LSS_tot,
            strikables_dict,
            periodicity):

    CONTRACTS, MIX, AGENTS, TECHNOLOGIC, DEMAND, AUCTION_WANTED_SOURCES, AMMORT, INSTRUMENT_TO_SOURCE_DICT, MARGIN, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.DEMAND, config.AUCTION_WANTED_SOURCES, config.AMMORT, config.INSTRUMENT_TO_SOURCE_DICT, config.MARGIN, config.env  # globals

    while True:

        ###############################################################################################################
        #                                                                                                             #
        #                                     First, the TPM checks if it got an                                      #
        #                                   strike, adds or changes its main policy                                   #
        #                                                                                                             #
        ###############################################################################################################

        _LSS_thresh = LSS_thresh[0] if env.now == 0 else AGENTS[env.now - 1][name]['LSS_thresh'][0]
        _impatience = impatience[0] if env.now == 0 else AGENTS[env.now - 1][name]['impatience'][0]
        _source = list(source[0].keys())[0] if env.now == 0 else list(AGENTS[env.now - 1][name]['source'][0].keys())[0]
        _past_weight = past_weight[0] if env.now == 0 else AGENTS[env.now - 1][name]['past_weight'][0]
        _memory = memory[0] if env.now == 0 else AGENTS[env.now - 1][name]['memory'][0]
        _discount = discount[0] if env.now == 0 else AGENTS[env.now - 1][name]['discount'][0]
        _disclosed_thresh = disclosed_thresh[0] if env.now == 0 else AGENTS[env.now - 1][name]['disclosed_thresh'][0]
        _rationale = rationale[0] if env.now == 0 else AGENTS[env.now - 1][name]['rationale'][0]
        _instrument = instrument[0] if env.now == 0 else AGENTS[env.now - 1][name]['instrument'][0]
        _periodicity = periodicity[0] if env.now == 0 else AGENTS[env.now - 1][name]['periodicity'][0]
        LSS_tot = LSS_tot if env.now == 0 else AGENTS[env.now - 1][name]['LSS_tot']
        value = disclosed_var

        ###############################################################################################################
        #                                                                                                             #
        #                                  First, the EPM adds or changes its policy                                  #
        #                                                                                                             #
        ###############################################################################################################

        policy_pool = [{'instrument': _instrument,
                        'source': _source,
                        'budget': value * wallet,
                        'entry_value': value}]

        if len(policies) > 0:
            policy_pool.append(policies)  # with this we create a temporary list with first the current policy and
            # afterwards all the other policies

        if env.now >= 1:
            for entry in policy_pool:

                entry_instrument = entry['instrument']
                entry_chosen_source = entry['source']
                budget = entry['budget']
                entry_value = entry['entry_value']

                if entry_instrument == 'carbon_tax':
                    """
                    CARBON-TAX
                    """

                    for _ in MIX[env.now]:
                        i = TECHNOLOGIC[0]['TP_thermal']
                        j = MIX[env.now][_]
                        """ for the carbon-tax, if the source is thermal (0) or natural gas (3), then, its OPEX gets
                         higher"""
                        if i['source'] == 0:
                            j['OPEX'] = i['OPEX'] * (1 + entry_value)
                            """ we can update in relation to the initial value, because there is no innovation for
                             non-renewables"""

                elif entry_instrument == 'FiT':
                    """
                    Feed-in Tariff
                    """

                    """ we are actually using Feed-in premium, since we are adding a payment to the market price"""
                    for _ in MIX[env.now]:
                        i = MIX[env.now][_]
                        if i['source'] == entry_chosen_source and (i['auction_contracted'] is False or
                                                                   'auction_contracted' not in i):
                            # print(i)
                            i['price'] = i['price'] * (1 + entry_value)

                elif entry_instrument == 'auction':
                    """
                    Auctions: these are a little bit more complicated
                    """

                    """ we do things with auction_countdown, which is a countdown for the auction period, which allows
                     the EPM to collect projects, and with auction_time, a boolean variable which signals if there is
                      an auction underway"""

                    if auction_countdown > 0 and auction_time is True:
                        """ if the countdown has yet to reach zero but an Auction is underway, then we cut down one
                         period from the auction_countdown """
                        auction_countdown -= 1
                    elif auction_time is False:
                        """ if we are doing an auction, but the auction_time is not signalling that an auction is 
                        underway, then we have to set it to true and trigger the countdown to the auction itself"""
                        auction_countdown = COUNTDOWN
                        auction_time = True
                        AUCTION_WANTED_SOURCES.append(entry_chosen_source)
                        # ic(AUCTION_WANTED_SOURCES)

                    elif auction_countdown == 0 and auction_time == True:
                        """ under these circumstances, it is time to do the auction and contract projects for the 
                        ppas"""
                        auction_time = False
                        possible_projects = []
                        contracted_projects = []

                        AUCTION_WANTED_SOURCES = []

                        """first we get the projects that were bidded"""
                        for i in range(env.now - COUNTDOWN, env.now - 1):
                            for _ in CONTRACTS[i]:
                                __ = CONTRACTS[i][_]
                                j = __.copy()
                                if j['receiver'] == name:
                                    """ here we are adding the whole contract"""
                                    possible_projects.append(j)
                        """ we have to sort in terms of 'OPEX' """
                        possible_projects = sorted(possible_projects, key=lambda x: x['price'])
                        # ic(possible_projects)
                        """ the capacity for auction is the value (in GW) multplied by 1000 (to become MW)"""

                        max_auction = 5000
                        min_auction = 1000

                        auction_size = max_auction - min_auction
                        auction_size *= entry_value
                        auction_size += min_auction

                        # remaining = AGENTS[env.now-1]['DD'].copy()['Remaining_demand']/(24 * 30)

                        """auction_MW = max(
                            min(
                                auction_size
                                + max(0, remaining),
                                max_auction),
                            min_auction)"""

                        auction_MW = max(
                            min(auction_size, max_auction), min_auction)

                        # ic(env.now, auction_MW, len(possible_projects))

                        remaining_capacity = auction_MW

                        """
                         The auction size is between 0,8 GW and 2 GW, being made of three parts:
                         
                         1- the auction capacity (in GW) times the decision_var
                         2- the remanining demand (surpluses reduce the auction size)
                         3- the expected increase in demand for the next 25 years
                         
                        """

                        uncontracted_projects = []
                        # _budget = budget
                        # print(_budget)
                        for project in possible_projects:
                            # print(12 * project['price'] * project['MWh'])
                            if remaining_capacity > 0:  # and _budget - 12 * project['price'] * project['MWh']>0:
                                """ if there is still capacity to contract, then the project is contracted"""
                                contracted_projects.append(project)
                                # _budget -= 12 * project['price'] * project['MWh']
                                # print('contracted', project['source'])
                            else:
                                uncontracted_projects.append(project)
                                # print('did not contract', project['source'])
                            remaining_capacity -= project['capacity']
                        """ then we cycle through the codes of the contracted projects in order to tell the proponent 
                        agents that their projects were contracted"""
                        for project in contracted_projects + uncontracted_projects:
                            """ first we update it to include attributes exclusive to PPAs, such as the auction price, 
                            the boolean auction_contracted and the date of expiration of the price"""
                            code = project['code']
                            project['receiver'] = project['sender']  # we have to switch them
                            """if math.isnan(price) is True:
                                print(project)
                                print(price, project['OPEX'], project['CAPEX'], project['lifetime'], MARGIN, project['MWh'])
                                print('check this out', math.isnan(project['OPEX']), math.isnan(project['CAPEX']), math.isnan(project['lifetime']), math.isnan(MARGIN), math.isnan(project['MWh']))"""
                            if project in contracted_projects:
                                project_update = {
                                    'status': 'project',
                                    'auction_contracted': True,
                                    'price_expiration': env.now + PPA_expiration + PPA_limit + 1,
                                    'limit': env.now + PPA_limit + 1,
                                    'sender': 'EPM',
                                    'auction_price': project.copy()['price']
                                }
                            else:
                                project_update = {
                                    'status': 'rejected',
                                    'auction_contracted': False,
                                    'sender': 'EPM'
                                }

                            project.update(project_update)

                            # ic(project)

                            CONTRACTS[env.now][code] = project

                        # TODO: check if this is necessary
                        """ lastly, we have to get rid of the contracted projects. This is relevant specially when  
                        there are multiple auctions happening"""
                        """ for i in range(env.now - 1 - COUNTDOWN, env.now - 1):
                            for _ in CONTRACTS[i]:
                                j = CONTRACTS[i][_]
                                if j['code'] in contracted_projects:
                                    j.update('bidded' == False)"""

        ###############################################################################################################
        #                                                                                                             #
        #                           Now, the EPM analyses what happened to the system due to                          #
        #                                              its intervention                                               #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 2 and env.now % _periodicity == 0:
            add_source = source_reporting_FF(name, _past_weight)
            for entry in range(len(source) - 1):
                source[entry][list(source[entry].keys())[0]] *= (1 - _discount) ** _periodicity
                source[entry][list(source[entry].keys())[0]] += add_source[list(source[entry].keys())[0]]

        ###############################################################################################################
        #                                                                                                             #
        #                                And then, the EPM will decide what to do next                                #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 0:
            decision_var = max(0, min(1, public_deciding_FF(name)))  # if env.now % _periodicity == 0 else decision_var
            disclosed_var = thresholding_FF(_LSS_thresh, disclosed_var, decision_var)
            decisions = evaluating_FF(name) if env.now % _periodicity == 0 else {'verdict': 'keep', 'strikes': False, 'impatience_increase': 0}
            verdict = decisions['verdict']

        ###############################################################################################################
        #                                                                                                             #
        #                           Before leaving, the agent must uptade the outside world                           #
        #                                                                                                             #
        ###############################################################################################################

        update = {
            "genre": genre,
            "subgenre": subgenre,
            "name": name,
            "wallet": wallet,
            "PPA_expiration": PPA_expiration,
            "PPA_limit": PPA_limit,
            "auction_countdown": auction_countdown,
            "auction_time": auction_time,
            "COUNTDOWN": COUNTDOWN,
            "decision_var": decision_var,
            "disclosed_var": disclosed_var,
            "verdict": verdict,
            "index_per_source": index_per_source,
            "auction_capacity": auction_capacity,
            "instrument": instrument,
            "source": source,
            "LSS_thresh": LSS_thresh,
            "impatience": impatience.copy(),
            "disclosed_thresh": disclosed_thresh,
            "past_weight": past_weight,
            "memory": memory,
            "discount": discount,
            "policies": policies,
            "rationale": rationale,
            "LSS_tot": LSS_tot,
            "strikables_dict": strikables_dict,
            "current_state": current_stating_FF(_rationale),
            "periodicity": periodicity
        }

        if env.now > 1:
            update['impatience'][0] = max(1, update['impatience'][0] + decisions['impatience_increase'])
            update["LSS_weak"] = AGENTS[env.now - 1][name]["LSS_weak"] + abs(
                decision_var - AGENTS[env.now - 1][name]['decision_var'])
        else:
            update["LSS_weak"] = 0

        AGENTS[env.now][name] = update.copy()
        if env.now > 1:
            post_evaluating_FF(decisions['strikes'], verdict, name, strikables_dict)

        yield env.timeout(1)


class DBB(object):
    def __init__(self, env, name, wallet, instrument, source, decision_var, LSS_thresh, past_weight,
                 memory, discount, policies, impatience, disclosed_thresh, rationale, periodicity):
        # Pre-Q:
        # self, env, wallet, dd_policy, dd_source, decision_var, dd_kappas, dd_qual_vars, dd_backwardness,
        #                  dd_avg_time, dd_discount, policies, dd_index, dd_eta, dd_ambition, dd_target, dd_rationale,
        #                  Portfolio,
        #                  accepted_sources, dd_SorT

        self.env = env
        self.NPV_THRESHOLD_DBB = config.NPV_THRESHOLD_DBB
        self.guaranteed_contracts = []
        self.genre = 'DBB'
        self.name = name
        self._wallet = wallet
        self.instrument = instrument
        self.source = source
        self.decision_var = decision_var
        self.disclosed_var = decision_var
        self.verdict = 'keep'
        self.LSS_thresh = LSS_thresh
        self.impatience = impatience
        self.disclosed_thresh = disclosed_thresh
        self.past_weight = past_weight
        self.memory = memory
        self.discount = discount
        self.policies = policies
        self.rationale = rationale
        self.financing_index = {0: 0, 1: 0, 2: 0}
        self.receivable = {0: 0, 1: 0, 2: 0}
        self.car_ratio = 0
        self.LSS_tot = 0
        self.accepted_tps = []
        # self.subgenre = 'DBB'
        # self.dd_qual_vars = dd_qual_vars
        # self.dd_policy = dd_policy
        # self.dd_index = dd_index
        # self.dd_eta = dd_eta
        # self.dd_ambition = dd_ambition
        # self.dd_target = dd_target
        # self.dd_SorT = dd_SorT

        strikables_dict = {'impatience': impatience,
                           'LSS_thresh': LSS_thresh,
                           'source': source,
                           'past_weight': past_weight,
                           'memory': memory,
                           'discount': discount,
                           'disclosed_thresh': disclosed_thresh,
                           'rationale': rationale,
                           'instrument': instrument,
                           'periodicity': periodicity
                           }

        self.strikables_dict = strikable_dicting(strikables_dict)
        self.added_mwh = 0
        self.periodicity = periodicity

        self.action = env.process(run_DBB(
            self.NPV_THRESHOLD_DBB,
            self.guaranteed_contracts,
            self.genre,
            self.name,
            self._wallet,
            self.instrument,
            self.source,
            self.decision_var,
            self.disclosed_var,
            self.verdict,
            self.LSS_thresh,
            self.impatience,
            self.disclosed_thresh,
            self.past_weight,
            self.memory,
            self.discount,
            self.policies,
            self.rationale,
            self.financing_index,
            self.receivable,
            self.car_ratio,
            self.strikables_dict,
            self.LSS_tot,
            self.accepted_tps,
            self.added_mwh,
            self.periodicity))


def run_DBB(NPV_THRESHOLD_DBB,
            guaranteed_contracts,
            genre,
            name,
            _wallet,
            instrument,
            source,
            decision_var,
            disclosed_var,
            verdict,
            LSS_thresh,
            impatience,
            disclosed_thresh,
            past_weight,
            memory,
            discount,
            policies,
            rationale,
            financing_index,
            receivable,
            car_ratio,
            strikables_dict,
            LSS_tot,
            accepted_tps,
            added_mwh,
            periodicity):

    global decisions
    CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, POLICY_EXPIRATION_DATE, INSTRUMENT_TO_SOURCE_DICT, RISKS, AGENTS_r, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.POLICY_EXPIRATION_DATE, config.INSTRUMENT_TO_SOURCE_DICT, config.RISKS, config.AGENTS_r, config.env  # globals

    while True:

        ###############################################################################################################
        #                                                                                                             #
        #                                     First, the DBB checks if it got an                                      #
        #                                   strike, adds or changes its main policy                                   #
        #                                                                                                             #
        ###############################################################################################################

        _LSS_thresh = LSS_thresh[0] if env.now == 0 else AGENTS[env.now - 1][name]['LSS_thresh'][0]
        _past_weight = past_weight[0] if env.now == 0 else AGENTS[env.now - 1][name]['past_weight'][0]
        _source = list(source[0].keys())[0] if env.now == 0 else list(AGENTS[env.now - 1][name]['source'][0].keys())[0]
        _memory = memory[0] if env.now == 0 else AGENTS[env.now - 1][name]['memory'][0]
        _discount = discount[0] if env.now == 0 else AGENTS[env.now - 1][name]['discount'][0]
        _impatience = impatience[0] if env.now == 0 else AGENTS[env.now - 1][name]['impatience'][0]
        _rationale = rationale[0] if env.now == 0 else AGENTS[env.now - 1][name]['rationale'][0]
        _instrument = instrument[0] if env.now == 0 else AGENTS[env.now - 1][name]['instrument'][0]
        _periodicity = periodicity[0] if env.now == 0 else AGENTS[env.now - 1][name]['periodicity'][0]
        LSS_tot = LSS_tot if env.now == 0 else AGENTS[env.now - 1][name]['LSS_tot']
        value = disclosed_var
        wallet = _wallet if not env.now % 12 == 0 else 0
        wallet += AGENTS[0][name]['wallet'] if env.now > 0 and env.now % 12 == 0 else _wallet  # the budget is monthly
        # print(wallet, 'wallet')

        ###############################################################################################################
        #                                                                                                             #
        #                           Before doing anything, the development bank must collect                          #
        #                             its  payments and guarantee contracts that need be                              #
        #                                                                                                             #
        ###############################################################################################################

        # first, we must update the risk dictionary

        """for source in [0, 1, 2, 3, 4, 5]:
            try:
                RISKS.update(
                    {source: finding_FF(MIX[env.now], 'risk', 'median', {'source': source})['value']})
                # we attempt to get the median risk for each source
            except:
                # if not possible (there are no banks financing that source or no banks at all) which translates to risky source
                RISKS.update({source: 1})"""

        if env.now > 0:

            for _ in CONTRACTS[env.now - 1]:
                i = CONTRACTS[env.now - 1][_]
                if i['receiver'] == 'DBB' and i['status'] == 'payment':
                    """ the contract is adressed to the development bank and is a payment"""
                    receivable.update({
                        i['source']: receivable[i['source']] - i['value']
                    })
                    wallet += i['value']
                elif i['receiver'] == 'DBB' and i['guarantee'] == True:
                    """ we are dealing with a guarantee"""
                    # wallet += i['value']

                    if i['status'] == 'default':
                        """ the company went into default and the guarantee is activated"""
                        j = i.copy()
                        j.update({'sender': 'DBB',
                                  'receiver': i['BB'],
                                  'status': 'payment'})
                        guaranteed_contracts.append(j)

                if len(guaranteed_contracts) > 0:
                    for i in guaranteed_contracts:
                        if i['ammortisation'] > env.now:
                            code = uuid.uuid4().int
                            j = i.copy()
                            CONTRACTS[env.now].update({code: j})
                            receivable.update({
                                i['source']: receivable[i['source']] - i['value']
                            })
                            wallet += i['value']
                        else:
                            guaranteed_contracts.remove(i)

        ###############################################################################################################
        #                                                                                                             #
        #                                  First, the DBB adds or changes its policy                                  #
        #                                                                                                             #
        ###############################################################################################################

        policy_pool = [{'instrument': _instrument,
                        'source': _source,
                        'budget': value * wallet}]

        if len(policies) > 0:
            policy_pool.append(policies)  # with this we create a temporary list with first the current policy and
            # afterwards all the other policies

        # ic(len(CONTRACTS[env.now - 1]), _instrument) if env.now>0 else None
        if env.now >= 1 and len(CONTRACTS[env.now - 1]) > 0:
            for entry in policy_pool:
                entry_instrument = entry['instrument']
                entry_chosen_source = entry['source']
                budget = entry['budget']
                entry_value = disclosed_var

                tps = []
                tp_value = []
                accepted_tps = [] if env.now % 24 == 0 else list(set(accepted_tps))
                for _ in AGENTS[env.now]:

                    tp = AGENTS[env.now][_]

                    # tps[tp['name']] = 0
                    if tp['genre'] == 'TP':
                        if _rationale == 'green':
                            tpupdate = tp['Technology']['avoided_emissions'] / tp['Technology']['MW']

                        elif _rationale == 'innovation':
                            tpupdate = tp['RandD']

                        else:
                            tpupdate = tp['capacity']

                        if entry_chosen_source == tp['Technology']['source']:
                            tps.append({tp['name']: tpupdate})
                            tp_value.append(tpupdate)

                tps.sort(key=lambda x: list(x.values())[0], reverse=True)
                try:
                    thresh = np.quantile(tp_value, (1-value)) # (env.now/config.SIM_TIME))
                except:
                    thresh = 10 ** 100

                for tp in tps:
                    if list(tp.values())[0] >= thresh:
                        accepted_tps.append(list(tp.keys())[0])

                # print('dbb', env.now, thresh, accepted_tps, env.now/config.SIM_TIME)
                # print(value)
                if entry_instrument == 'finance':
                    # print('DBB is trying to finance')

                    non_used_wallet = wallet - budget
                    financing = financing_FF(genre, name, budget, receivable, entry_value, financing_index,
                                             accepted_source=entry_chosen_source, accepted_tps=accepted_tps,
                                             added_mwh=added_mwh)

                    wallet = financing['wallet'] + non_used_wallet
                    receivable = financing['receivables']
                    added_mwh = financing['added_mwh']

                elif entry_instrument == 'guarantee':
                    """ We are dealing with guarantees """
                    financing = financing_FF(genre, name, wallet, receivable, entry_value, financing_index,
                                             guaranteeing=True, accepted_source=entry_chosen_source,
                                             accepted_tps=accepted_tps)

                    wallet = financing['wallet']
                    receivable = financing['receivables']
                    financing_index = financing['financing_index']


        # print('DBB wallet' , wallet)
        # _wallet += wallet # all non-used budget is kept for next month
        
        ###############################################################################################################
        #                                                                                                             #
        #                           Now, the DBB analyses what happened to the system due to                          #
        #                                              its intervention                                               #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 2 and env.now % _periodicity == 0:
            add_source = source_reporting_FF(name, _past_weight)
            for entry in range(len(source) - 1):
                source[entry][list(source[entry].keys())[0]] *= (1 - _discount) ** _periodicity
                source[entry][list(source[entry].keys())[0]] += add_source[list(source[entry].keys())[0]]

        ###############################################################################################################
        #                                                                                                             #
        #                                And then, the DBB will decide what to do next                                #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 0:
            decision_var = max(0, min(1, public_deciding_FF(name)))  # if env.now % _periodicity == 0 else decision_var
            disclosed_var = thresholding_FF(_LSS_thresh, disclosed_var, decision_var)
            decisions = evaluating_FF(name) if env.now % _periodicity == 0 else {'verdict': 'keep', 'strikes': False, 'impatience_increase': 0}
            verdict = decisions['verdict']
            # print('strikes are', decisions['strikes'])

        ###############################################################################################################
        #                                                                                                             #
        #                           Before leaving, the agent must uptade the outside world                           #
        #                                                                                                             #
        ###############################################################################################################

        update = {
            "NPV_THRESHOLD_DBB": NPV_THRESHOLD_DBB,
            "guaranteed_contracts": guaranteed_contracts,
            "genre": genre,
            "name": name,
            "wallet": wallet,
            "instrument": instrument,
            "source": source,
            "decision_var": decision_var,
            "disclosed_var": disclosed_var,
            "verdict": verdict,
            "LSS_thresh": LSS_thresh,
            "impatience": impatience.copy(),
            "disclosed_thresh": disclosed_thresh,
            "past_weight": past_weight,
            "memory": memory,
            "discount": discount,
            "policies": policies,
            "rationale": rationale,
            "financing_index": financing_index,
            "receivable": receivable,
            "car_ratio": car_ratio,
            "strikables_dict": strikables_dict,
            "current_state": current_stating_FF(_rationale),
            "LSS_tot": LSS_tot,
            "interest_rate": interesting_FF(disclosed_var),
            "accepted_tps": accepted_tps,
            "periodicity": periodicity
        }

        if env.now > 1:
            update['impatience'][0] = max(1, update['impatience'][0] + decisions['impatience_increase'])
            update["LSS_weak"] = AGENTS[env.now - 1][name]["LSS_weak"] + abs(
                decision_var - AGENTS[env.now - 1][name]['decision_var'])
        else:
            update["LSS_weak"] = 0

        AGENTS[env.now][name] = update.copy()
        if env.now > 1:
            post_evaluating_FF(decisions['strikes'], verdict, name, strikables_dict)

        yield env.timeout(1)


class BB(object):
    # TODO: all the BB...
    def __init__(self, env, Portfolio, accepted_sources, name, wallet, dd_source, decision_var, dd_kappas, dd_qual_vars,
                 dd_backwardness, dd_avg_time, dd_discount, dd_strategies, dd_index):
        self.env = env
        self.financing_index = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.Portfolio = Portfolio
        self.receivable = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.accepted_sources = accepted_sources
        self.car_ratio = 0
        self.name = name  # name of the agent, is a string, normally something like BB_01
        self.genre = 'BB'  # genre, we do not use type, because type is a dedicated command of python, is also a string
        self.subgenre = 'BB'
        self.wallet = wallet  # wallet or reserves, or savings, etc. How much the agent has? is a number
        self.profits = 0  # profits of the agent, is a number
        self.dd_profits = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0,
                           5: 0}  # same as profits, but as dict. Makes accounting faster and simpler
        self.dd_source = dd_source  # This, my ganzirosis, used to be the Tactics. It is the first of the ranked dictionaries. It goes a little sumthing like dis: dd = {'current' : 2, 'ranks' : {0: 3500, 1: 720, 2: 8000}}. With that we have the current decision for the variable or thing and on the ranks we have the score for
        self.decision_var = decision_var  # this is the value of the decision variable. Is a number between -1 and 1
        self.verdict = "keep"  # this is the verdict variable. It can be either 'keep', 'change' or 'add'
        """self.dd_kappas = dd_kappas  # this is the kappa, follows the current ranked dictionary
        self.dd_qual_vars = dd_qual_vars  # this tells the agent the qualitative variables in a form {0 : 'name of the zeroth variable', 1 : 'name of the first variable', 2 : 'name of the second variable'}
        self.dd_backwardness = dd_backwardness  # also a ranked dictionary, this one tells the backwardness of agents
        self.dd_avg_time = dd_avg_time  # also a ranked dictionary, this one tells the average time for deciding if change is necessary
        self.dd_discount = dd_discount  # discount factor. Is a ranked dictionary
        self.dd_strategies = dd_strategies  # initial strategy for the technology provider. Is a ranked dictionary
        self.dd_index = dd_index"""
        self.shareholder_money = 0

        self.action = env.process(run_BB(self.financing_index,
                                         self.Portfolio,
                                         self.receivable,
                                         self.accepted_sources,
                                         self.car_ratio,
                                         self.name,
                                         self.genre,
                                         self.subgenre,
                                         self.wallet,
                                         self.profits,
                                         self.dd_profits,
                                         self.dd_source,
                                         self.decision_var,
                                         self.verdict,
                                         self.shareholder_money
                                         ))


def run_BB(NPV_THRESHOLD_DBB,
           guaranteed_contracts,
           genre,
           name,
           wallet,
           policy,
           source,
           decision_var,
           disclosed_var,
           verdict,
           LSS_thresh,
           impatience,
           disclosed_thresh,
           past_weight,
           memory,
           discount,
           policies,
           index_per_source,
           rationale,
           financing_index,
           Portfolio,
           shareholder_money):
    CONTRACTS, MIX, AGENTS, AGENTS_r, TECHNOLOGIC, r, BASEL, AMMORT, TACTIC_DISCOUNT, NPV_THRESHOLD, RISKS, env = config.CONTRACTS, config.MIX, config.AGENTS, config.AGENTS_r, config.TECHNOLOGIC, config.r, config.BASEL, config.AMMORT, config.TACTIC_DISCOUNT, config.NPV_THRESHOLD, config.RISKS, config.env  # globals

    while True:

        ###############################################################################################################
        #                                                                                                             #
        #                            Before anything, we must the current values of each of                           #
        #                               the dictionaries that we use and other variables                              #
        #                                                                                                             #
        ###############################################################################################################

        list_of_strikables = [dd_kappas, dd_qual_vars, dd_backwardness, dd_avg_time, dd_discount, dd_strategies]

        _source = source[0]
        _LSS_thresh = LSS_thresh[0]
        _past_weight = past_weight[0]
        _memory = memory[0]
        _discount = dd_discount['current']
        _impatience = impatience[0]
        value = decision_var
        profits = 0  # in order to get the profits of this period alone

        ###############################################################################################################
        #                                                                                                             #
        #                                       First, the bank collect profits                                       #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 0:
            for _ in CONTRACTS[env.now - 1]:
                i = CONTRACTS[env.now - 1][_]
                if i['receiver'] == name and i['status'] == 'payment':
                    profits += i['value']
                    wallet += i['value']
                    receivable.update({
                        i['source']: receivable[i['source']] - i['value']
                    })
                    dd_profits.update({
                        i['source']: dd_profits[i['source']] - i['value']
                    })

        ###############################################################################################################
        #                                                                                                             #
        #                        If it is the end of the year, then the BB shares its profits                         #
        #                                            with its shareholders                                            #
        #                                                                                                             #
        ###############################################################################################################

        if env.now % 12 == 0 and env.now > 0 and wallet > 0:
            profits_to_shareholders = wallet * (1 - value)
            wallet -= profits_to_shareholders
            shareholder_money += profits_to_shareholders

        ###############################################################################################################
        #                                                                                                             #
        #                           Now, on to check if change is on and if there is a strike                         #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 0 and (verdict == 'add' or 'change'):
            striked = striking_FF(list_of_strikables, kappa)

            for entry in range(0, len(list_of_strikables)):
                list_of_strikables[entry] = striked[entry]
                if entry == 'source':
                    # we changed the source, so we have to update the accepted_sources dictionary
                    source_accepting_FF(accepted_sources, source)
            verdict = 'keep'  # we already changed, now back to business

        ###############################################################################################################
        #                                                                                                             #
        #                               Then, the bank decides which projects to accept                               #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 0 and len(CONTRACTS[env.now - 1]) > 0:
            financing = financing_FF(genre, name, wallet, receivable, value, financing_index)

            wallet = financing['wallet']
            receivable = financing['receivables']
            financing_index = financing['financing_index']

        ###############################################################################################################
        #                                                                                                             #
        #                           Now, the BB analyses what happened to the system due to                           #
        #                                              its intervention                                               #
        #                                                                                                             #
        ###############################################################################################################

        add_source = source_reporting_FF(name)

        for entry in dd_source['ranks']:
            dd_source['ranks'][entry] *= (1 - discount)
            dd_source['ranks'][entry] += add_source[entry]

        ###############################################################################################################
        #                                                                                                             #
        #                                 And then, the BB will decide what to do next                                #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 0:
            decision_var = max(0, min(1, private_deciding_FF(name)))
            decisions = evaluating_FF(name)

        ###############################################################################################################
        #                                                                                                             #
        #                           Before leaving, the agent must uptade the outside world                           #
        #                                                                                                             #
        ###############################################################################################################

        AGENTS[env.now].update({name: {
            "financing_index": financing_index,
            "Portfolio": Portfolio,
            "receivable": receivable,
            "accepted_sources": accepted_sources,
            "car_ratio": car_ratio,
            "name": name,
            "genre": genre,
            "subgenre": subgenre,
            "wallet": wallet,
            "profits": profits,
            "dd_profits": dd_profits,
            "dd_source": dd_source,
            "decision_var": decision_var,
            "verdict": verdict,
            "dd_kappas": dd_kappas,
            "dd_qual_vars": dd_qual_vars,
            "dd_backwardness": dd_backwardness,
            "dd_avg_time": dd_avg_time,
            "dd_discount": dd_discount,
            "dd_strategies": dd_strategies,
            "source": source,
            "kappa": kappa,
            "qual_vars": qual_vars,
            "backwardness": backwardness,
            "avg_time": avg_time,
            "discount": discount,
            "strategy": strategy,
            "index": index,
            "value": value,
            "interest_rate": 1 + r + decision_var
        }})

        profits_dedicting_FF(name)
        if env.now > 1:
            post_evaluating_FF(decisions['strikes'], verdict, name, strikables_dict)

        yield env.timeout(1)


class EP(object):
    def __init__(self, env, name, wallet, portfolio_of_plants, portfolio_of_projects,
                 periodicity, tolerance, last_acquisition_period, source, decision_var, LSS_thresh,
                 impatience, memory, discount, past_weight, current_weight):
        # Pre-Q variables:
        # self, env, accepted_sources, name, wallet, EorM, portfolio_of_plants, portfolio_of_projects,
        #                  periodicity, tolerance, last_acquisition_period, dd_source, decision_var, dd_kappas,
        #                  dd_qual_vars,
        #                  dd_backwardness, dd_avg_time, dd_discount, dd_strategies, dd_index

        self.env = env
        self.genre = 'EP'
        # self.accepted_sources = accepted_sources
        self.name = name
        self.wallet = wallet
        self.profits = 0
        self.impatience = impatience
        self.current_weight = current_weight
        # self.EorM = EorM
        # self.subgenre = EorM
        self.capacity = {0: 0, 1: 0, 2: 0}  # if self.EorM == 'E' else {3: 0, 4: 0, 5: 0}
        self.portfolio_of_plants = portfolio_of_plants
        self.portfolio_of_projects = portfolio_of_projects
        self.periodicity = periodicity
        self.subgenre_price = {0: 0, 1: 0, 2: 0}  # if self.EorM == 'E' else {3: 0, 4: 0, 5: 0}
        self.tolerance = tolerance
        self.last_acquisition_period = last_acquisition_period
        self.dd_profits = {0: 0, 1: 0, 2: 0}  # same as profits, but as dict. Makes accounting faster and simpler
        self.source = source
        self.decision_var = decision_var  # this is the value of the decision variable. Is a number between 0 and 1
        self.verdict = "keep"  # this is the verdict variable. It can be either 'keep', 'change' or 'add'
        self.LSS_thresh = LSS_thresh  # this is the kappa, follows the current ranked dictionary
        # self.dd_qual_vars = dd_qual_vars  # this tells the agent the qualitative variables in a form {0 : 'name of the zeroth variable', 1 : 'name of the first variable', 2 : 'name of the second variable'}
        self.past_weight = past_weight  # also a ranked dictionary, this one tells the backwardness of agents
        self.memory = memory  # also a ranked dictionary, this one tells the average time for deciding if change is necessary
        self.discount = discount  # discount factor. Is a ranked dictionary
        # self.dd_strategies = dd_strategies  # initial strategy for the technology provider. Is a ranked dictionary
        self.index = {0: 0, 1: 0, 2: 0}

        strikables_dict = {'impatience': impatience,
                           'LSS_thresh': LSS_thresh,
                           'tolerance': tolerance,
                           'past_weight': past_weight,
                           'memory': memory,
                           'discount': discount,
                           'source': source}
        # print(strikables_dict)

        self.strikables_dict = strikable_dicting(strikables_dict)

        self.strikables_dict = strikables_dict

        self.profits = 0

        self.LSS_tot = 0
        self.shareholder_money = 0
        self.reinvest = True

        # ic(name, wallet, portfolio_of_plants, portfolio_of_projects, periodicity, tolerance, last_acquisition_period, source, decision_var, LSS_thresh, impatience, memory, discount, past_weight, current_weight)

        self.action = env.process(run_EP(self.env,
                                         self.genre,
                                         self.name,
                                         self.wallet,
                                         self.portfolio_of_plants,
                                         self.portfolio_of_projects,
                                         self.periodicity,
                                         self.tolerance,
                                         self.last_acquisition_period,
                                         self.source,
                                         self.decision_var,
                                         self.LSS_thresh,
                                         self.impatience,
                                         self.memory,
                                         self.discount,
                                         self.past_weight,
                                         self.current_weight,
                                         self.index,
                                         self.strikables_dict,
                                         self.verdict,
                                         self.profits,
                                         self.dd_profits,
                                         self.LSS_tot,
                                         self.shareholder_money,
                                         self.reinvest))


def run_EP(env,
           genre,
           name,
           wallet,
           portfolio_of_plants,
           portfolio_of_projects,
           periodicity,
           tolerance,
           last_acquisition_period,
           source,
           decision_var,
           LSS_thresh,
           impatience,
           memory,
           discount,
           past_weight,
           current_weight,
           index,
           strikables_dict,
           verdict,
           profits,
           dd_profits,
           LSS_tot,
           shareholder_money,
           reinvest):
    CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, DEMAND, AMMORT, AUCTION_WANTED_SOURCES, AGENTS_r, EP_NUMBER, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.DEMAND, config.AMMORT, config.AUCTION_WANTED_SOURCES, config.AGENTS_r, config.EP_NUMBER, config.env

    while True:

        ###############################################################################################################
        #                                                                                                             #
        #                            Before anything, we must the current values of each of                           #
        #                               the dictionaries that we use and other variables                              #
        #                                                                                                             #
        ###############################################################################################################

        """if env.now == config.FUSS_PERIOD:
            strikables_dict['source'] = source
            # With this we ensure that sources are not changed during the fuss period"""

        # print(strikables_dict) if env.now > config.FUSS_PERIOD else None

        _source = list(source[0].keys())[0] if env.now == 0 else list(AGENTS[env.now - 1][name]['source'][0].keys())[0]
        _LSS_thresh = LSS_thresh[0] if env.now == 0 else AGENTS[env.now - 1][name]['LSS_thresh'][0]
        _past_weight = past_weight[0] if env.now == 0 else AGENTS[env.now - 1][name]['past_weight'][0]
        _memory = memory[0] if env.now == 0 else AGENTS[env.now - 1][name]['memory'][0]
        _discount = discount[0] if env.now == 0 else AGENTS[env.now - 1][name]['discount'][0]
        _impatience = impatience[0] if env.now == 0 else AGENTS[env.now - 1][name]['impatience'][0]
        _current_weight = current_weight[0] if env.now == 0 else AGENTS[env.now - 1][name]['current_weight'][0]
        _tolerance = tolerance[0] if env.now == 0 else AGENTS[env.now - 1][name]['tolerance'][0]
        index = {0: 0, 1: 0, 2: 0} if env.now == 0 else indexing_FF(name)
        LSS_tot = LSS_tot if env.now == 0 else AGENTS[env.now - 1][name]['LSS_tot']
        value = decision_var
        profits = 0  # in order to get the profits of this period alone
        dd_profits = {0: 0, 1: 0, 2: 0}

        ###############################################################################################################
        #                                                                                                             #
        #                                             Collecting profits                                              #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 0 and len(MIX[env.now - 1]) > 0:
            """ The EP only collects profits if it produces energy and that can only happen if we are not in the start of the simulation, and if the MIX is not empty"""
            """ Moreover, if the plant was activated, then the EP pays the OPEX of it """
            for _ in MIX[env.now - 1]:
                """ Then we check each plant in the mix """
                """ the _ is the code of the plant, whereas i is the dictionary of the plant itself """
                i = MIX[env.now - 1][_].copy()
                if i['EP'] == name:

                    addition = -i['OPEX']  # * 0.5

                    if i['status'] == 'contracted':
                        """ if the plant is mine and it is contracted, I'll collect profits """
                        addition = i['MWh'] * i['price']

                        """ we also have to put the profit as a contract in the CONTRACTS dictionary in order for the 
                        policy makers, other EPs and the demand to do some calculations """
                        code = uuid.uuid4().int
                        """ this is to get a unique and random number """
                        j = i.copy()
                        """ and we also have to create a copy of the i dictionary, if not things will update on that 
                        dictionary, and that's no good """
                        j.update({
                            'status': 'payment',
                            'sender': 'DD',
                            'source': j['source'],
                            'receiver': name,
                            'value': j['MWh'] * j['price']  # - i['OPEX'] * 0.5  # all plants have operating costs
                        })
                        """ and now we update"""
                        CONTRACTS[env.now - 1][code] = j
                    """else:
                        addition = -0.5 * i['OPEX']  # all plants have operating costs"""

                    wallet += addition
                    profits += addition
                    """if math.isnan(i['MWh'] * i['price'] - i['OPEX']) is True:
                        print(name, math.isnan(i['MWh']), math.isnan(i['price']), math.isnan(i['OPEX']))
                        print(i)"""
                    dd_profits[i['source']] += addition

        ###############################################################################################################
        #                                                                                                             #
        #                        If it is the end of the year, then the EP shares its profits                         #
        #                                            with its shareholders                                            #
        #                                                                                                             #
        ###############################################################################################################

        if env.now % 12 == 0 and env.now > 0 and wallet > 0:
            # print('BEFORE', wallet, value)
            profits_to_shareholders = wallet * (1 - value)
            wallet -= profits_to_shareholders
            shareholder_money += profits_to_shareholders
            # print('after', wallet)

        ###############################################################################################################
        #                                                                                                             #
        #                           Now the EP goes through its portfolio_of_plants in order                          #
        #                            to 1) pay the banks for the financing, 2) check if                               #
        #                              plants finished building, 3) retire old plants and                             #
        #                                        4) insert plants into the mix                                        #
        #                                                                                                             #
        ###############################################################################################################

        if len(portfolio_of_plants) > 0:
            for _ in portfolio_of_plants:
                i = portfolio_of_plants[_]
                Fee = (i['principal'] / (1 + AMMORT)) + i['debt'] * i['r'] if 'debt' in i else 0

                """  1) we pay for the ammortization of plants """

                if i['ammortisation'] > env.now and (
                        i['guarantee'] is not True or (i['guarantee'] is True and wallet >= Fee)
                ) and i['BB'] != 'reinvestment':
                    """ if the plant is not guaranteed or if it's guaranteed but the EP has enough money to cover it """
                    # print(env.now, name, Fee, wallet, i['r'], 'debt' in i, i['source'])
                    wallet -= Fee
                    i['debt'] -= Fee

                    code = uuid.uuid4().int
                    CONTRACTS[env.now].update({
                        code: {
                            'sender': name,
                            'receiver': i['BB'],
                            'status': 'payment',
                            'source': i['source'],
                            'value': Fee
                        }
                    })
                elif i['ammortisation'] > env.now and i['guarantee'] is True and wallet < Fee:
                    """ if not, then the development bank pays for that monthly fee"""
                    j = i.copy()
                    j.update({
                        'receiver': 'DBB',
                        'status': 'default'
                    })
                    code = uuid.uuid4().int
                    CONTRACTS[env.now][code] = j

                """ 2) now we retire old pants """

                if i['retirement'] <= env.now:
                    i['status'] = 'retired'
                    # print(_, 'was retired')

                """ 3) and we check if plants finished building """

                if i['completion'] <= env.now:
                    i['status'] = 'built'

                """ 4) now we insert built plants into the mix  """

                if i['status'] == 'built':
                    j = i.copy()

                    """if config.MARGIN < 1:
                        _margin = 1 + max(config.MARGIN, (1-value))
                    else:
                        _margin = 1 + config.MARGIN - value"""

                    _margin = config.MARGIN ** (1-value)

                    _price = (j['OPEX'] + (j['CAPEX'] / j['lifetime'])) * _margin / j['MWh']
                    j['price'] = 0 if 'auction_price' in j and i['auction_contracted'] is True else _price
                    if 'auction_contracted' not in j or j['auction_contracted'] == False:
                        j['auction_contracted'] = False

                        """if j['MWh'] * j['price'] < 0.5 * i['OPEX']:
                            j['status'] = 'not_available'"""

                    MIX[env.now][_] = j

        ###############################################################################################################
        #                                                                                                             #
        #                               Now, the EP has to check if projects that were                                #
        #                              auction contracted or had guarantees got financed.                             #
        #                                  If not, they need to be inserted into the                                  #
        #                               portfolio_of_projects dictionary in order to be                               #
        #                                retired each period until the TOLERANCE limit                                #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 0 and len(CONTRACTS[env.now - 1]) > 0:
            for _ in CONTRACTS[env.now - 1]:
                i = CONTRACTS[env.now - 1][_]
                # print('contract', _, i)
                if i['receiver'] == name and i['status'] == 'financed':
                    """ the project was financed """
                    last_acquisition_period = env.now
                    jj = i.copy()
                    jj.update({
                        'EP': name,
                        'BB': jj['sender'],
                        'status': 'building',
                        'principal': jj['CAPEX'] * (1 + r) ** jj['building_time'],
                        'completion': jj['building_time'] + 1 + env.now,
                        'retirement': jj['lifetime'] + jj['building_time'] + 1 + env.now
                    })
                    # print(name, 'found its project', jj, 'financed at time', env.now, 'by the bank', jj['BB'])

                    """ moreover, if it is a molecule project,
                    the price is pre-fixed """
                    """if i['EorM'] == 'M':
                        j.update({
                            'price': (i['OPEX'] + (i['CAPEX'] / i['lifetime'])) / i['MWh'] * (1 + value)
                        })"""
                    # j.pop('receiver')
                    # j.pop('sender')

                    portfolio_of_plants[_] = jj.copy()
                    # print('portfolio_of_plants.update({_ : jj})', _, jj)
                    # capacity[i['source']] = capacity[i['source']] + i['capacity']

                    """ if the financed project was in the pool of projects of the EP, we have to take it out """
                    if _ in portfolio_of_projects:
                        # print('popped ', _, 'from the portfolio of projects of ', name, ' being financed by', jj['sender'])
                        portfolio_of_projects.pop(_)

                    """ now we ready the contract that tells the TP that he got a new project"""
                    code = uuid.uuid4().int
                    CONTRACTS[env.now].update({
                        code: {'sender': name,
                               'receiver': jj['TP'],
                               'status': 'payment',
                               'source': jj['source'],
                               'value': jj['CAPEX'],
                               'MWh': jj['MWh']
                               }
                    })
                elif (
                        i['status'] in ['project', 'rejected'] and
                        (
                                (i['status'] == 'rejected') or
                                (i['guarantee'] == True or i['auction_contracted'] == True)
                        ) and name in [i['receiver'], i['sender']]
                ):
                    # if the project was not financed but it got a guarantee or whas a PPA, we have to prepare it to be
                    # inserted into the portfolio_of_projects dictionary
                    j = i.copy()
                    j.update({'code': _})
                    if 'limit' not in j:
                        # if the key 'limit' is not in j, then we insert it, as well as the list failed_attempts, in
                        # which we put the name of the banks that rejected the project
                        # j['limit'], j['failed_attempts'] = env.now + 1 + _tolerance, [i['receiver']]
                        j['limit'] = env.now + 1 + _tolerance
                    # print(_, 'added to the portfolio of projects')
                    portfolio_of_projects.update({_: j})

        ###############################################################################################################
        #                                                                                                             #
        #                            Then, the Energy producer decides how much to invest                             #
        #                                                                                                             #
        ###############################################################################################################
        if env.now > 0 and env.now % periodicity == 0:
            # cash_flow_RISK = config.RISKS[_source] if _source not in config.AUCTION_WANTED_SOURCES else 0
            cash_flow_RISK = config.RISKS[_source] if _source not in config.AUCTION_WANTED_SOURCES else 0
            if value > 0:
                remain = AGENTS[env.now - 1]['DD'].copy()['Remaining_demand'] / (
                            24 * 30 * EP_NUMBER ** math.e ** (1-value))
                self_expansion = DEMAND.copy()[env.now - 1] / EP_NUMBER ** math.e ** (1 - value)
            else:
                remain = 0
                self_expansion = 0

            remain *= value
            self_expansion *= value

            # mix_expansion = random.uniform(5, 100)

            mix_expansion = (remain + self_expansion * (1-cash_flow_RISK))  # / EP_NUMBER ** math.e ** (1-value)

            # print(env.now, '_source', _source, value, name, mix_expansion, "remain", remain, 'self_expansion', self_expansion, (1-cash_flow_RISK))

            # print(mix_expansion)
            # ic(mix_expansion,remain/ EP_NUMBER ** math.e ** (1-value), self_expansion/ EP_NUMBER ** math.e ** (1-value), value, _source) #  if env.now> config.FUSS_PERIOD else None
            # ic(AGENTS[env.now-1]['DD']['Remaining_demand'], value)
            """if AGENTS[env.now-1]['DD']['Remaining_demand'] > 0:
                condition = True
            else:
                condition = value > 0"""
            condition = True if value > 0 else False
        else:
            condition = False
            mix_expansion = 0

        if (condition == True) and (mix_expansion > 0):  # and wallet > 0:
            # print(name, 'is trying to increase its capacity')
            """if AGENTS[env.now - 1]['DD']['Remaining_demand'] > 0:
                mix_expansion = AGENTS[env.now - 1]['DD']['Remaining_demand'] / (24 * 30 * EP_NUMBER)  # the demand is
                # in mwh, but the expansion is in MW
            else:
                mix_expansion = config.INITIAL_DEMAND/EP_NUMBER
                mix_expansion *= value"""

            # print(mix_expansion)

            max_lump = {0: int(1500/config.THERMAL['MW']),
                        1: int(399/config.WIND['MW']),
                        2: int(329/config.SOLAR['MW'])}

            # print(EP_NUMBER)
            _TP = {'TP': 0,
                   'NPV': 0,
                   'Lumps': 0,
                   'CAPEX': 0,
                   'OPEX': 0
                  }
            tech = list(TECHNOLOGIC[env.now - 1].copy().values())
            random.shuffle(tech)

            for i in tech:
                if i['source'] == _source or ('BNDES' in AGENTS[env.now - 1] and i['name'] in AGENTS[env.now - 1]['BNDES']['accepted_tps']) or i['source'] in config.AUCTION_WANTED_SOURCES:

                    # print(source_price)
                    # Lumps = min(max(1, np.ceil(mix_expansion / i['MW'])), max_lump[_source])

                    Lumps = min(np.ceil(mix_expansion / i['MW']), max_lump[i['source']])

                    if 'BNDES' not in AGENTS[env.now - 1] and config.BB_NUMBER == 0:
                        # print('not')
                        Lumps = min(np.ceil(wallet * value / i['CAPEX']), Lumps)

                    if Lumps < 1 and not (('BNDES' in AGENTS[env.now - 1] and i['name'] in AGENTS[env.now - 1]['BNDES']['accepted_tps']) or i['source'] in config.AUCTION_WANTED_SOURCES):
                        # we want to contract less than a single lump of investment
                        # print('breakf stuff')
                        continue
                    elif Lumps < 1 and ('BNDES' in AGENTS[env.now - 1] and i['name'] in AGENTS[env.now - 1]['BNDES']['accepted_tps']) or i['source'] in config.AUCTION_WANTED_SOURCES:
                        Lumps = min(np.ceil(self_expansion / i['MW']), max_lump[i['source']])  # * (1-cash_flow_RISK)
                        # print('when the levee breaks')

                    # print('Lumps', Lumps)

                    if len(MIX[env.now-1]) == 0:
                        source_price = config.STARTING_PRICE
                    else:
                        if 'EPM' in AGENTS[env.now - 1] and i['source'] in config.AUCTION_WANTED_SOURCES:

                            source_price = max(AGENTS[env.now - 1]['DD']["Price"],
                                               (i['OPEX'] + (i['CAPEX'] / i['lifetime'])) * config.MARGIN ** (1-value) /
                                               (Lumps * i['MW'] * 24 * 30 * i['CF']))
                            """try:
                                source_price = max(
                                    AGENTS[env.now - 1]['DD']["Price"],
                                    weighting_FF(env.now - 1, 'price', 'MWh', MIX)[i['source']],
                                    weighting_FF(env.now - 1, 'auction_price', 'MWh', MIX)[i['source']])
                            except:
                                source_price = max(
                                    AGENTS[env.now - 1]['DD']["Price"],
                                    weighting_FF(env.now - 1, 'price', 'MWh', MIX)[i['source']])"""
                        else:
                            source_price = AGENTS[env.now - 1]['DD']["Price"]

                    price = source_price
                    # ic(config.RISKS)
                    # print(env.now, _source, cash_flow_RISK)
                    try:
                        """financing_RISK = 1 - (AGENTS[env.now - 1]['BNDES']['financing_index'][_source] /
                                              max(0.1 / 10 ** 25,
                                                  sum(AGENTS[env.now - 1]['BNDES']['financing_index'].values())
                                                  )
                                              )"""
                        # print(list(AGENTS[env.now - 1]['BNDES']['source'][0].keys())[0])
                        # ic(_source,list(AGENTS[env.now - 1]['BNDES']['source'][0].keys())[0], i['name'],AGENTS[env.now - 1]['BNDES']['accepted_tps'])
                        if i['name'] in AGENTS[env.now - 1]['BNDES']['accepted_tps']:
                            financing_RISK = AGENTS[env.now - 1]['BNDES']['disclosed_var']
                        else:
                            financing_RISK = True  # -1 * AGENTS[env.now - 1]['BNDES']['disclosed_var']

                        interest = r if financing_RISK is True else interesting_FF(financing_RISK)
                        NPV = npv_generating_FF(
                            interest, i['lifetime'], Lumps, i['MW'], i['building_time'], i['CAPEX'], i['OPEX'], price,
                            i['CF'], AMMORT, financing_RISK=financing_RISK)
                    except:
                        # financing_RISK = 0
                        NPV = npv_generating_FF(
                                r, i['lifetime'], Lumps, i['MW'], i['building_time'], i['CAPEX'], i['OPEX'], price,
                                i['CF'], AMMORT, cash_flow_RISK=cash_flow_RISK, reinvest=True)

                    if i['source'] != _source:
                        # that's not the main source, so we must check sumtings
                        cond = False
                        for ___ in source:
                            if list(___.keys())[0] == i['source']:
                                ratio = list(___.values())[0]/list(source[0].values())[0]
                                if ratio > random.uniform(0, 1):
                                    # If that's not the main source of the EP, then it must check the score of the
                                    # score: if it's higher than the score of its main source, then ok, if not, then
                                    # we must run a random uniform with the ratio as threshold
                                    cond = True
                                    # print('condi', list(___.values())[0], list(source[0].values())[0])
                                    # print('ratio', (ratio ** -1) ** _impatience)
                                    if (1 - _past_weight) > random.uniform(0, 1):
                                        # (max(ratio, 0.1 * 10 ** 25) ** -1) * (1 - _past_weight) < random.uniform(0, 1):
                                        for attempt in range(0, int(_impatience * (1 - _past_weight))):
                                            random.shuffle(source)
                                            if list(source[0].keys())[0] == i['source']:
                                                # print('redux')
                                                LSS_tot += 1
                                                break
                                    else:
                                        impatience[0] = impatience[0]+1  # if impatience[0]-1 > 0 else AGENTS[0][name][impatience[0]]

                    else:
                        cond = True

                    if cond is True and NPV >= 0 and (NPV > _TP['NPV'] or
                                                      random.uniform(0, 1) < config.INITIAL_RANDOMNESS):
                        # print('worked')
                        _TP.update({
                            'TP': i['name'],
                            'NPV': NPV,
                            'Lumps': Lumps,
                            'CAPEX': i['CAPEX'] * Lumps,
                            'OPEX': i['OPEX'] * Lumps,
                            'source_of_TP': i['source'],
                            'price': price
                        })

            # OPEX and CAPEX are in relation to one lump, so in the project we have to change them to account for the
            # whole project
            # ic(TP['TP'], name, _source) if TP['TP'] == 0 else None
            if _TP['TP'] != 0:
                project = TECHNOLOGIC[env.now - 1][_TP['TP']].copy()
                # we have to use .copy() here to avoid changing the TECHNOLOGIC dictionary entry
                project.update({
                    'sender': name,
                    'receiver': 'EPM' if _TP['source_of_TP'] in AUCTION_WANTED_SOURCES else bank_sending_FF(),
                    'TP': _TP['TP'],
                    'Lumps': _TP['Lumps'],
                    'old_CAPEX': TECHNOLOGIC[env.now - 1][_TP['TP']]['CAPEX'],
                    'old_OPEX': TECHNOLOGIC[env.now - 1][_TP['TP']]['OPEX'],
                    'CAPEX': _TP['CAPEX'],
                    'OPEX': _TP['OPEX'],
                    'status': 'bidded' if _TP['source_of_TP'] in AUCTION_WANTED_SOURCES else 'project',
                    'capacity': _TP['Lumps'] * project['MW'],
                    'MWh': _TP['Lumps'] * project['MW'] * 24 * 30 * project['CF'],
                    'avoided_emissions': TECHNOLOGIC[env.now - 1][_TP['TP']]['avoided_emissions'] * _TP['Lumps'],
                    'emissions': TECHNOLOGIC[env.now - 1][_TP['TP']]['emissions'] * _TP['Lumps'],
                    'guarantee': False,
                    'npv': _TP['NPV'],
                    'price': _TP['price']
                })
                if _TP['source_of_TP'] not in AUCTION_WANTED_SOURCES:
                    project['auction_contracted'] = False
                    """# print('sending to EPM for auction')

                    if config.MARGIN < 1:
                        _margin = 1 + max(config.MARGIN, (1-value))
                    else:
                        _margin = 1 + config.MARGIN - value

                    project.update(
                        {'status': 'bidded',
                         'receiver': 'EPM'})
                else:
                    project['status'] = 'project'"""
                    # project['auction_contracted'] = False
                # print(_TP['source_of_TP'], AUCTION_WANTED_SOURCES, _TP['source_of_TP'] in AUCTION_WANTED_SOURCES)
                code = uuid.uuid4().int
                project['code'] = code

                if config.BB_NUMBER > 0 or _TP['source_of_TP'] in AUCTION_WANTED_SOURCES:
                    # print('sent')
                    CONTRACTS[env.now][code] = project
                    # ic(name, project)
                else:
                    # If there are no banks and no current auctions, the project goes straight to the reinvestment
                    # possibility
                    portfolio_of_projects.update({code: project})
            else:
                None  # print(env.now, name, _TP, _source)
        # print(portfolio_of_projects, name)

        _to_pop = []
        for _ in portfolio_of_projects:
            """ now we have to resend the "projects in the portfolio_of_projects dictionary """
            i = portfolio_of_projects[_].copy()

            if 'limit' not in i:
                i['limit'] = env.now + 1 + _tolerance

            if i['limit'] == env.now and _ not in _to_pop:
                # print('project ', _, ' has reached its limit time...')
                _to_pop.append(_)
                continue

            if i['CAPEX'] > wallet*value:
                receiver = bank_sending_FF()
                project = i.copy()
                project.update({'sender': name,
                                'receiver': receiver,
                                'status': 'project'})
                CONTRACTS[env.now].update({
                    _: project
                })

            elif i['CAPEX'] <= wallet*value and reinvest is True:
                # print(i['CAPEX'], wallet*value, value, name)

                """if ('BNDES' in AGENTS[env.now - 1] and
                        i['source'] == list(AGENTS[env.now - 1]['BNDES']['source'][0].keys())[0]):
                    if random.uniform(0, 1) >= config.INITIAL_RANDOMNESS:
                        continue"""
                if i['source'] != _source:
                    break

                cash_flow_RISK = 0 if i['auction_contracted'] is True else config.RISKS[_source]

                if len(MIX[env.now - 1]) == 0:
                    source_price = config.STARTING_PRICE
                else:
                    if 'auction_contracted' in i and i['auction_contracted'] is True:
                        source_price = max(AGENTS[env.now - 1]['DD']["Price"], i['auction_price'])
                    else:
                        source_price = AGENTS[env.now - 1]['DD']["Price"]
                        """try:
                            source_price = max(
                                AGENTS[env.now - 1]['DD']["Price"],
                                weighting_FF(env.now - 1, 'price', 'MWh', MIX)[i['source']],
                                weighting_FF(env.now - 1, 'auction_price', 'MWh', MIX)[i['source']])
                        except:
                            source_price = max(
                                AGENTS[env.now - 1]['DD']["Price"] ** 0,
                                weighting_FF(env.now - 1, 'price', 'MWh', MIX)[i['source']])"""


                # financing_RISK = 0
                NPV = npv_generating_FF(
                    r, i['lifetime'], 1, i['MW'], i['building_time'], i['CAPEX'], i['OPEX'], source_price,
                    i['CF'], AMMORT, cash_flow_RISK=cash_flow_RISK, reinvest=True)

                max_npv = finding_FF(portfolio_of_plants, 'npv', how='highest')['value']
                min_npv = finding_FF(portfolio_of_plants, 'npv', how='lowest')['value']

                if NPV >= random.uniform(min_npv * (1 - value), max_npv * (1 - value)):
                    # print(name, 'has just reinvested in a plant with source ', i['source'], 'and capacity of', i['capacity'], value)
                    wallet -= i['CAPEX']
                    code = _  # uuid.uuid4().int
                    # print(_)
                    _to_pop.append(_)
                    project = i.copy()
                    project.update({
                        'EP': name,
                        'BB': 'reinvestment',
                        'sender': 'reinvestment',
                        'receiver': name,
                        'status': 'financed',
                        'principal': 0,
                        'completion': project['building_time'] + 1 + env.now,
                        'retirement': project['lifetime'] + project['building_time'] + 1 + env.now,
                        'ammortisation': env.now,
                        'npv': NPV
                    })
                    CONTRACTS[env.now][_] = project
                    last_acquisition_period = env.now
                    # code = uuid.uuid4().int
                    """CONTRACTS[env.now].update({
                        code:
                            {'sender': name,
                             'receiver': k['TP'],
                             'status': 'payment',
                             'source': k['source'],
                             'value': k['CAPEX'],
                             'MWh': k['MWh']
                             }
                    })"""

        if len(_to_pop) > 0:
            for code in _to_pop:
                portfolio_of_projects.pop(code)

        ###############################################################################################################
        #                                                                                                             #
        #                           Now, the EP analyses what happened to the system due to                           #
        #                                              its intervention                                               #
        #                                                                                                             #
        ###############################################################################################################

        if env.now > 0 and env.now % periodicity == 0:
            add_source = source_reporting_FF(name, _past_weight)  # , index)
            for entry in range(len(source) - 1):
                source[entry][list(source[entry].keys())[0]] *= (1 - _discount) ** periodicity

                source[entry][list(source[entry].keys())[0]] += add_source[list(source[entry].keys())[0]]

        ###############################################################################################################
        #                                                                                                             #
        #                                 And then, the EP will decide what to do next                                #
        #                                                                                                             #
        ###############################################################################################################
        if env.now > 0:
            decision_var = max(0, min(1, private_deciding_FF(name)))
            decisions = evaluating_FF(name)
            verdict = decisions['verdict']

        ###############################################################################################################
        #                                                                                                             #
        #                           Before leaving, the agent must update the outside world                           #
        #                                                                                                             #
        ###############################################################################################################
        update = {"name": name,
                  "genre": genre,
                  "wallet": wallet,
                  "portfolio_of_plants": portfolio_of_plants,
                  "portfolio_of_projects": portfolio_of_projects,
                  "periodicity": periodicity,
                  "tolerance": tolerance,
                  "last_acquisition_period": last_acquisition_period,
                  "source": source,
                  "decision_var": decision_var,
                  "LSS_thresh": LSS_thresh,
                  "impatience": impatience.copy(),
                  "memory": memory,
                  "discount": discount,
                  "past_weight": past_weight,
                  "current_weight": current_weight,
                  "index": index,
                  "strikables_dict": strikables_dict,
                  "verdict": verdict,
                  "profits": profits,
                  "dd_profits": dd_profits,
                  "LSS_tot": LSS_tot,
                  "shareholder_money": shareholder_money
                  }

        if env.now > 1:
            update['impatience'][0] = max(1, update['impatience'][0] + decisions['impatience_increase'])
            update["LSS_weak"] = AGENTS[env.now - 1][name]["LSS_weak"] + abs(
                decision_var - AGENTS[env.now - 1][name]['decision_var'])
        else:
            update["LSS_weak"] = 0

        AGENTS[env.now][name] = update.copy()

        if env.now > 1:
            post_evaluating_FF(decisions['strikes'], verdict, name, strikables_dict)
            profits_dedicting_FF(name)

        yield env.timeout(1)


class Demand(object):
    def __init__(self, env, initial_demand, when, increase):
        self.env = env
        self.genre = 'DD'
        self.name = 'DD'
        self.initial_demand = [initial_demand].copy()[0]
        self.when = when
        self.increase = increase
        self.Demand_by_source = {0: 0, 1:0, 2:0}
        self._Price = config.STARTING_PRICE
        self.action = env.process(run_DD(self.env,
                                         self.genre,
                                         self.name,
                                         self.initial_demand,
                                         self.when,
                                         self.increase,
                                         self.Demand_by_source,
                                         self._Price))


def run_DD(env,
           genre,
           name,
           initial_demand,
           when,
           increase,
           Demand_by_source,
           _Price
           ):
    CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, DEMAND, MARGIN,  EP_NUMBER, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.DEMAND, config.MARGIN, config.EP_NUMBER, config.env

    while True:

        # print(env.now)
        # from_time_to_agents_FF(AGENTS)
        # from_time_to_agents_FF(TECHNOLOGIC)

        if env.now < config.FUSS_PERIOD:
            config.RANDOMNESS = 1
        else:
            config.RANDOMNESS = config.INITIAL_RANDOMNESS

        ###############################################################################################################
        #                                                                                                             #
        #                                       Which plants will be contracted?                                      #
        #                                                                                                             #
        ###############################################################################################################

        if env.now == 0:
            DEMAND.update({env.now: initial_demand})
            printable = 'seed is ' + str(config.seed)
            # print(printable)

        elif env.now % when == 0 and env.now > 0:
            # print(env.now)
            """ first, we get how much green is E or M"""
            """greeness = {'E': 0, 'M': 0}
            total = 0
            for month in range(env.now - when, env.now - 1):
                if len(MIX[month]) > 1:
                    for _ in MIX[month]:
                        i = MIX[month][_]
                        if i['status'] == 'contracted':
                            if i['green'] is True:
                                greeness.update({i['EorM']: greeness[i['EorM']] + i['MWh']})
                            total += i['MWh']"""

            """green = {'E': greeness['E'] / total, 'M': greeness['M'] / total}

            expected_increase = (DEMAND[env.now - 1]['E'] + DEMAND[env.now - 1]['M']) * increase
            pendulum_demand = specificities['EorM'] * expected_increase
            prices = weighting_FF(env.now - 1, 'price', 'MWh', MIX, demand=True)
            a = green_awareness
            E_pendulum = a * (prices['E'] / sum(list(prices.values()))) + (1 - a) * green['E']
            E_pendulum *= pendulum_demand
            E_increase = DEMAND[env.now - 1]['E'] * specificities['increase'] + E_pendulum
            M_pendulum = a * (prices['M'] / sum(list(prices.values()))) + (1 - a) * green['M']
            M_pendulum *= pendulum_demand
            M_increase = DEMAND[env.now - 1]['M'] * specificities['increase'] + M_pendulum"""

            DEMAND.update({env.now: DEMAND.copy()[env.now - 1] * (1 + increase)})

        else:
            DEMAND.update({env.now: DEMAND.copy()[env.now - 1]})

        """ since the policy makers act after private agents, they are looking at the env.now, not the env.now-1 """
        demand = DEMAND.copy()[env.now] * 24 * 30
        price = config.STARTING_PRICE

        chosen = []
        installed_MWh = 0
        _Price_list = []
        _MWhs = []
        risk_dict = {0: 0, 1: 0, 2: 0}
        risk = 'price'

        if env.now > 0 and len(MIX[env.now]) > 0:

            """ 
            First, we contract and precify the electricity projects
            """

            possible_projects = list(MIX[env.now].values())
            possible_projects = sorted(possible_projects, key=lambda x: (~x['auction_contracted'],
                                                                         x['price']))

            # print(possible_projects) if env.now == 100 else None

            # possible_projects = sorted(possible_projects, reverse=True, key=lambda x: x['auction_contracted'])
            # print(possible_projects)
            # print('full demand is ', demand)
            for plant in possible_projects:
                """if plant['TP'] in AGENTS[env.now-1]['BNDES']['accepted_tps']:
                    print('plant', plant['BB'], plant['TP'])"""

                if plant['status'] == 'built' and (
                        ('auction_contracted' in plant and plant['auction_contracted'] is True) or demand >= 0):
                    """ if there is still demand to to be supplied, then, the power plant is contracted"""
                    """ if the plant is dispatchable it will always enter """
                    """ if the plant is auction contracted it will also always enter"""
                    MIX[env.now][plant['code']]['status'] = 'contracted'
                    chosen.append(plant)
                    Demand_by_source[plant['source']] += plant['MWh']
                    # print('demand decrease by', plant['MWh'])
                else:
                    """ if there is no more demand to be supplied, then the plant is not contracted"""
                    MIX[env.now][plant['code']].update({
                        'status': 'built',
                        'price': 0
                    })
                installed_MWh += plant['MWh']

                demand -= plant['MWh']
                # print(MIX[env.now][plant['code']], 'was not contracted')
            """ following the merit order, the system is precified in relation to its most costly unit"""

            chosen_wo_auctions = sorted(chosen, key=lambda x: x['price'])
            chosen_wo_auctions = [elem for elem in chosen_wo_auctions if elem['auction_contracted'] is not True]

            if len(chosen_wo_auctions) > 0:
                chosen_plant = chosen_wo_auctions[-1]
                price = chosen_plant['price']
                """if chosen_plant['auction_contracted'] is False:"""
                """price = (
                                chosen_plant['OPEX'] + (chosen_plant['CAPEX'] / chosen_plant['lifetime'])
                        ) * (1 + MARGIN) / chosen_plant['MWh']"""
                """else:
                    price = chosen_plant['price']"""

                # print(price) if env.now == 100 else None
            else:
                price = AGENTS[env.now-1]['DD']['Price']
                # print(env.now, 'zero price, no thermal contracted')

            _price = round(price, 3)
            for _ in MIX[env.now]:
                i = MIX[env.now][_]

                if i['status'] == 'contracted':
                    """ if the plant is auction contracted, then we do not mess with its price"""
                    if 'auction_contracted' in i and i['auction_contracted'] is True:
                        # _price = max(round(price, 3), i['auction_price'])
                        """MIX[env.now][i['code']].update({
                            'price': _price})"""
                        MIX[env.now][i['code']].update({
                            'price': i['auction_price']})
                        # print(i['auction_price'])
                    else:
                        MIX[env.now][i['code']].update({
                            'price': _price})
                        # print(price, ' normal_price')

                    _Price_list.append(_price)
                    _MWhs.append(MIX[env.now][i['code']]['MWh'])

            # _Price = weighting_FF(env.now, 'price', 'MWh', MIX) if len(MIX[env.now]) > 0 else config.STARTING_PRICE
            _Price = sum([_Price_list[k]*_MWhs[k] for k in range(len(_Price_list)-1)])/sum(_MWhs)
            # print(env.now, "_Price", _Price, "price", price)
            # ic(env.now, increase, demand, DEMAND[env.now], _Price)


            if risk == 'demand':
                total_MWh = installed_MWh  # sum(Demand_by_source.values())
                for source in list(risk_dict.keys()):
                    risk_dict[source] = Demand_by_source[source] / total_MWh
                max_percentage = max(list(risk_dict.values()))
                for source in list(risk_dict.keys()):
                    risk_dict[source] = max_percentage - risk_dict[source]
                    config.RISKS[source] = risk_dict[source]
                # config.RISKS[source] = risk_dict[source]
                # print(config.RISKS)

            elif risk == 'source':
                for source in risk_dict:
                    """print('RISK', finding_FF(
                        MIX[env.now], 'MWh', 'sum', {'status': 'contracted', 'source': source})['value'],  max(
                        finding_FF(MIX[env.now], 'MWh', 'sum', {'source': source})['value'], 0.1 / 10 ** 25))"""
                    risky = finding_FF(
                        MIX[env.now], 'MWh', 'sum', {'status': 'contracted', 'source': source})['value'] / max(
                        finding_FF(MIX[env.now], 'MWh', 'sum', {'source': source})['value'], 0.1 / 10 ** 25)
                    # print(source, risk_dict[source])
                    risk_dict[source] = 1 - risky
                    config.RISKS[source] = risk_dict[source]
                    # print(env.now, source, config.RISKS[source])

            elif risk == 'price':
                for source in risk_dict:
                    if source != 0:
                        min_price = finding_FF(TECHNOLOGIC[env.now - 1], 'min_price', 'lowest', {'source': source})[
                            'value']
                    else:
                        min_price = config.STARTING_PRICE/(1+config.MARGIN)
                    config.RISKS[source] = min_price/_price
                    # print(env.now, config.RISKS)
                    # print(_price)


            else:
                # print('BLAM')
                total_MWh = sum(Demand_by_source.values())
                for source in list(risk_dict.keys()):
                    risk_dict[source] = Demand_by_source[source] / total_MWh
                max_percentage = max(list(risk_dict.values()))
                for source in list(risk_dict.keys()):
                    risk_dict[source] = max_percentage - risk_dict[source]
                    risky = finding_FF(
                        MIX[env.now], 'MWh', 'sum', {'status': 'contracted', 'source': source})['value'] / max(
                        finding_FF(MIX[env.now], 'MWh', 'sum', {'source': source})['value'], 0.1 / 10 ** 25)
                    risk_dict[source] += 1 - risky
                    config.RISKS[source] = risk_dict[source]/2


            """if env.now == config.FUSS_PERIOD - 1:
                if demand > 0:
                    print('We failed to provide for the whole system on time by ', str(demand))
                elif env.now == config.FUSS_PERIOD and demand < 0:
                    print('We reached the whole system with excess of ', str(-demand))
                print('demand was', DEMAND[env.now])
                DEMAND[env.now] += -1 * demand/(24*30)
                print('now demand is', DEMAND[env.now])"""

        if env.now == config.SIM_TIME - 3:

            if len(_MWhs) == 0:
                _MWhs = [0]
            try:
                print('supply was', sum(_MWhs), "with excess of ", -demand, "at a price of", _Price, 'with a renewable to fossil ratio of', (Demand_by_source[1]+Demand_by_source[2])/Demand_by_source[0])
            except:
                print('supply was', sum(_MWhs), "with excess of ", -demand, "at a price of", _Price,
                      'without fossil generation')

        # print(env.now, _Price, price)

        AGENTS[env.now][name] = {
            'name': name,
            'genre': 'DD',
            'Demand': DEMAND.copy()[env.now],
            'Remaining_demand': demand,
            'Demand_by_source': Demand_by_source,
            'Price': price,
            'Avg_Price': _Price}

        yield env.timeout(1)


class Hetero(object):
    def __init__(self, env, threshold, heterogeneous, check):
        self.env = env
        self.genre = 'HH'
        self.name = 'HH'
        self.threshold = threshold
        self.randomly = True  # False if config.INITIAL_RANDOMNESS < 1 else True  # just to put more emphasis on the
        # randomness of the random runs
        self.heterogeneous = heterogeneous
        self.check = check
        self.action = env.process(run_HH(self.env,
                                         self.genre,
                                         self.name,
                                         self.threshold,
                                         self.randomly,
                                         self.heterogeneous,
                                         self.check))


def run_HH(env,
           genre,
           name,
           threshold,
           randomly,
           heterogeneous,
           check
           ):
    CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, DEMAND, MARGIN,  EP_NUMBER, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.DEMAND, config.MARGIN, config.EP_NUMBER, config.env

    while True:

        ###############################################################################################################
        #                                                                                                             #
        #                                       How many public agents are there?                                     #
        #                                                                                                             #
        ###############################################################################################################

        list_of_pm = []
        chchchanges = 0

        for _ in AGENTS[env.now]:
            agent = AGENTS[env.now][_]

            if agent['genre'] in ['DBB', 'EPM', 'TPM']:
                list_of_pm.append(agent['name'])

        # periodicity = np.median([AGENTS[env.now][x]['periodicity'][0] for x in list_of_pm])
        # print(periodicity)

        periodicity = 3

        if len(list_of_pm) > 1 and env.now > config.FUSS_PERIOD and env.now % periodicity == 0:

            # That means we have more than two policy makers

            ###########################################################################################################
            #                                                                                                         #
            #                                        Perform the heterogeneity check                                  #
            #                                                                                                         #
            ###########################################################################################################

            # We are saving the results of the homogeneity to the demand object, just to avoid creating a new entry
            # on the agents dictionary

            # above the threshold, so we have to homogenize things

            list_o_entries = []

            if check == 'policymaking' or check == 'all':
                list_o_entries += ['memory', 'LSS_thresh', 'past_weight', 'periodicity', 'disclosed_thresh']

            if check == 'policy' or check == 'all':
                list_o_entries += ['disclosed_var', 'source'] # , 'rationale', -> we can't change rationale, otherwise the satisficing score of policy makers become really bizarre
                """if heterogeneous is True:
                    list_o_entries += ['source']"""

            """list_o_entries = ['source',
                              'disclosed_var',
                              'LSS_thresh',
                              'disclosed_thresh']"""

            chosen_entries = set()
            agent = AGENTS[env.now]
            for entry in list_o_entries:

                for number in range(len(list_of_pm)-1):
                    if entry == 'disclosed_var':
                        if agent[list_of_pm[number]][entry] != agent[list_of_pm[number+1]][entry]:
                            # print(AGENTS[env.now]['BNDES'][entry], AGENTS[env.now]['EPM'][entry])
                            chosen_entries.add(entry)
                            # they are the same
                        elif round(agent[list_of_pm[number]][entry], 1) == round(agent[list_of_pm[number+1]][entry], 1):
                            if heterogeneous is True:
                                # they shouldn't be
                                chosen_entries.add(entry)

                    elif entry == 'source':
                        if list(agent[list_of_pm[number]][entry][0].keys()) != list(agent[list_of_pm[number+1]][entry][0].keys()):
                            # print(AGENTS[env.now]['BNDES'][entry], AGENTS[env.now]['EPM'][entry])
                            chosen_entries.add(entry)
                        else:
                            # they are the same
                            if heterogeneous is True:
                                # they shouldn't be
                                chosen_entries.add(entry)

                    else:
                        if agent[list_of_pm[number]][entry][0] != agent[list_of_pm[number+1]][entry][0]:
                            # print(AGENTS[env.now]['BNDES'][entry][0], AGENTS[env.now]['EPM'][entry][0])
                            chosen_entries.add(entry)
                        else:
                            # they are the same
                            if heterogeneous is True:
                                # they shouldn't be
                                chosen_entries.add(entry)

            chosen_entries = [elem for elem in chosen_entries]

            if len(chosen_entries) > 0 and len(chosen_entries)/len(list_o_entries) > threshold:
                # print('homo')
                # above the threshold and above the heterogeneity test, so we have to homogenize things
                random.shuffle(chosen_entries)  # it must be random which ones go first
                ratio = len(chosen_entries)/len(list_o_entries)
                for entry in chosen_entries:
                    current = {}
                    previous = {}
                    if env.now > 0 and ratio > threshold:
                        ratio -= 1/len(list_o_entries)
                        chchchanges += 1

                        for name in list_of_pm:
                            if entry == 'disclosed_var':
                                current[name] = agent[name][entry]
                                previous[name] = AGENTS[env.now - 1][name][entry]
                            elif entry == 'source':
                                current[name] = list(agent[name][entry][0].keys())[0]
                                previous[name] = list(AGENTS[env.now - 1][name][entry][0].keys())[0]
                            else:
                                current[name] = agent[name][entry][0]
                                previous[name] = AGENTS[env.now - 1][name][entry][0]

                    else:
                        for name in list_of_pm:
                            current[name] = random.uniform(0, 100)
                            previous[name] = random.uniform(0, 100)

                    who = []

                    if entry == 'source' and heterogeneous is True and len(list_of_pm)>2:
                        sources = []
                        for name in list_of_pm:
                            current_source = list(agent[name][entry][0].keys())[0]
                            sources.append(current_source)

                        if np.std(sources) == 0:
                            # std zero means that they are the same source
                            who = list_of_pm

                    else:
                        for name in list_of_pm:
                            if current[name] == previous[name]:
                                who.append(name)

                    # If nobody changed, then we really don't have to do nothing

                    if len(who) > 0:
                        # someone changed
                        who = list_of_pm
                        if randomly is True or len(who) > 1 or len(list_of_pm) == 3:
                            # If random is True, then it is randomized
                            # If more than one policy maker changed, then it really doesn't matter who follows who
                            # If we have more than one pm, then it's best to keep it randomized
                            """
                            It is best to just copy the whole list to avoid destroying entries
                            """

                            chosen_agent = random.choice(list_of_pm)
                            chosen = agent[chosen_agent][entry]
                            if entry == 'source':
                                chosen_source = list(chosen[0].keys())[0]
                                for name in list_of_pm:
                                    agent_source = list(agent[name][entry][0].keys())[0]
                                    if agent_source != chosen_source and name != chosen_agent:
                                        agent[name][entry] = agent[name][entry][::-1]
                                        # we only two renewable sources, so if it's different, by just inverting it, we
                                        # have what we want

                            else:
                                if entry == 'disclosed_var':
                                    # max
                                    """max_chosen = 0
                                    
                                    for name in 2*list_of_pm:
                                        max_chosen = max(max_chosen, agent[name][entry])
                                        agent[name][entry] = max_chosen"""
                                    # mean
                                    discloseds = []
                                    for name in list_of_pm:
                                        discloseds.append(agent[name][entry])
                                    if heterogeneous is False:
                                        # print('homo!')
                                        for name in who:
                                            # agent[name][entry] = np.mean(discloseds)
                                            _median = np.median(discloseds)
                                            agent[name][entry] = np.mean([agent[name][entry], _median])
                                            # print("median", _median, agent[name][entry], name)
                                        # print('homo', env.now, agent[name][entry])
                                    else:

                                        # print('hetero!')

                                        for name in who:
                                            # noinspection PyTypeChecker
                                            new_discloseds = []
                                            var = random.uniform(agent[name][entry] - agent[name]['disclosed_thresh'][0],
                                                                 agent[name][entry] + agent[name]['disclosed_thresh'][0])
                                            agent[name][entry] = min(max(0, var), 1)
                                            # print('median', agent[name][entry], name, env.now)
                                else:
                                    for name in list_of_pm:
                                        if heterogeneous is False:
                                            if name != chosen_agent:
                                                agent[name][entry] = chosen
                                        else:
                                            random.shuffle(agent[name][entry])

                        # If it's not random, then we have to change to the least updated one
                        else:
                            who = random.choice(who)
                            # if more than one changed, then we randomly select one that changed

                            for name in list_of_pm:
                                if entry != 'source' and name != who:
                                    agent[name][entry] = agent[who][entry]
                                elif (entry == 'source' and list(agent[who][entry][0].keys())[0] !=
                                      list(agent[name][entry][0].keys())[0]) and name != who:
                                    name[name][entry] = name[who][entry][::-1]

                """LSS_guys = []

                for name in list_of_pm:
                    if AGENTS[env.now][name]['LSS_tot'] != AGENTS[env.now - 1][name]['LSS_tot']:
                        LSS_guys.append(name)"""

                # bndes_lss_cond = AGENTS[env.now]['BNDES']['LSS_tot'] > AGENTS[env.now - 1]['BNDES']['LSS_tot']
                # epm_lss_cond = AGENTS[env.now]['EPM']['LSS_tot'] > AGENTS[env.now - 1]['EPM']['LSS_tot']
                """if len(LSS_guys) > 1:
                    # if both policy makers adapted, then it really doesn't matter who we decrease the LSS count
                    random.shuffle(list_of_pm) # we have to randomize who will decrease the LSS

                    for number in range(1, len(list_of_pm)):
                        # print(list_of_pm),
                        # print(number)
                        agent[list_of_pm[number]]['LSS_tot'] -= 1"""

                """elif bndes_lss_cond is True or epm_lss_cond is True:
                    # if just one adapted, then we decrease its LSS
                    chosen_agent = 'BNDES' if bndes_lss_cond is True else 'EPM'
                    AGENTS[env.now][chosen_agent]['LSS_tot'] -= 1"""

                # print(AGENTS[env.now][who_else[0]][entry], AGENTS[env.now][who[0]][entry], 'equal?')

                #AGENTS[env.now]['DD']['homo'] = chchchanges  # Just to say that things were changed
                # print(env.now, 'HOMO')
        AGENTS[env.now]['DD']['homo'] = chchchanges

        yield env.timeout(1)


#######################################################################################################################
#                                                                                                                     #
#                                             CREATION EQUATIONS                                                      #
#                                                                                                                     #
#######################################################################################################################

def make_ep(env,
            name=None,
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

    :param green_ep:
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
    if name is None:
        name = 'EP_' + str(uuid.uuid4().hex)

    if wallet is None:
        from __init__ import config_name
        if "NO_NO_NO" in config_name or "NO_YES_NO" in config_name or "NO_NO_YES" in config_name or "NO_YES_YES" in config_name:
            magnitude = random.triangular(4, 7, 8)
        else:
            magnitude = random.triangular(2, 4, 6)
        # print(magnitude)

        # wallet = random.normalvariate(config.THERMAL['CAPEX'] * int(1500/config.THERMAL['MW']), .1)
        wallet = random.normalvariate(0, 1) * 20 * 10 ** magnitude  # random.uniform(10, 30)*10**4
        # print(name, wallet)
    if decision_var is None:
        decision_var = random.uniform(0, 1)
    if current_weight is None:
        current_weight = [0.25, 0.5, .75, .1, .9]
        random.shuffle(current_weight)
    if past_weight is None:
        past_weight = [0.25, 0.5, .75, .1, .9]
        random.shuffle(past_weight)
    if memory is None:
        memory = [int(random.uniform(3, 24))]
    if impatience is None:
        impatience = [int(random.uniform(1, 5))]
    if LSS_thresh is None:
        LSS_thresh = [0.25, 0.5, .75, .1, .9]
        random.shuffle(LSS_thresh)
    if source is None:
        source = [{0: random.uniform(0, 1000)}, {1: random.uniform(0, 500)}, {2: random.uniform(0, 500)}]
        source.sort(key=lambda x: list(x.values())[0], reverse=True)

    if tolerance is None:
        tolerance = [int(random.uniform(6, 12))]
    if periodicity is None:
        periodicity = int(random.uniform(3, 12))
    if discount is None:
        discount = [random.uniform(0.00001, 0.0001)]

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


def make_tp(env, name=None, wallet=None, capacity=None, source=None, RnD_threshold=None, capacity_threshold=None, decision_var=None, cap_conditions=None, impatience=None, past_weight=None, LSS_thresh=None, memory=None, discount=None, strategy=None, starting_tech_age=None):

    if name is None:
        name = 'TP_' + str(uuid.uuid4().hex)
    if source is None:
        source = random.choice([1, 2])

    if wallet is None:
        wallet = random.normalvariate(0, 1)* 200*10**4 if source == 1 else random.normalvariate(0, 1)* 20*10**4

    if capacity is None:
        capacity = random.uniform(1, 2)*10**6 if source == 1 else random.uniform(0, 1)*10**6

    if RnD_threshold is None:
        RnD_threshold = random.uniform(2, 5)

    if capacity_threshold is None:
        capacity_threshold = random.uniform(2, 4)

    if decision_var is None:
        decision_var = random.uniform(0, 1)

    if cap_conditions is None:
        cap_conditions = {}

    if impatience is None:
        impatience = [int(random.uniform(1, 5))]

    if past_weight is None:
        past_weight = random.sample([.25, .5, .75, .1, .9], 5)

    if LSS_thresh is None:
        LSS_thresh = random.sample([.25, .5, .75, .1, .9], 5)

    if memory is None:
        memory = [int(random.uniform(12, 48))]

    if discount is None:
        discount = [random.uniform(0.00001, 0.0001)]

    if strategy is None:
        strategy = random.sample(['capacity', "innovation"], 2)

    if starting_tech_age is None:
        starting_tech_age = 2 if source == 1 else 1

    return TP(
        env=env,
        name=name,
        wallet=wallet,
        capacity=capacity,
        Technology={'name': name,
                    "green": True,
                    "source": source,
                    "source_name": 'wind' if source == 1 else 'solar',
                    "CAPEX": config.WIND['CAPEX'] if source == 1 else config.SOLAR['CAPEX'],
                    'OPEX': config.WIND['OPEX'] if source == 1 else config.SOLAR['OPEX'],
                    "dispatchable": False,
                    "transport": False if source == 1 else True,
                    "CF": config.WIND['CF'] if source == 1 else config.SOLAR['CF'],
                    "MW": config.WIND['MW'] if source == 1 else config.SOLAR['MW'],
                    "lifetime": config.WIND['lifetime'] if source == 1 else config.SOLAR['lifetime'],
                    'building_time': config.WIND['building_time'] if source == 1 else config.SOLAR['building_time'],
                    'last_radical_innovation': 0,
                    'last_marginal_innovation': 0,
                    'emissions': 0,
                    'avoided_emissions': config.THERMAL['emissions'] * (
                            config.WIND['MW'] * config.WIND['CF'] /
                            config.THERMAL['MW'] * config.THERMAL['CF']) if source == 1 else config.THERMAL['emissions'] * (
                            config.SOLAR['MW'] * config.SOLAR['CF'] /
                            config.THERMAL['MW'] * config.THERMAL['CF']
                    )},
        RnD_threshold=RnD_threshold,
        capacity_threshold=capacity_threshold,
        decision_var=decision_var,
        cap_conditions=cap_conditions,
        impatience=impatience,
        past_weight=past_weight,
        LSS_thresh=LSS_thresh,
        memory=memory,
        discount=discount,
        strategy=strategy,
        starting_tech_age=starting_tech_age)
