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
            self.prod_cap_pct
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
           prod_cap_pct):
    CONTRACTS, MIX, AGENTS, AGENTS_r, TECHNOLOGIC, TECHNOLOGIC_r, r, AMMORT, rNd_INCREASE, RADICAL_THRESHOLD, env = config.CONTRACTS, config.MIX, config.AGENTS, config.AGENTS_r, config.TECHNOLOGIC, config.TECHNOLOGIC_r, config.r, config.AMMORT, config.rNd_INCREASE, config.RADICAL_THRESHOLD, config.env  # globals

    while True:

        #################################################################
        #                                                               #
        #     Before anything, we must the current values of each of    #
        #        the dictionaries that we use and other variables       #
        #                                                               #
        #################################################################

        _LSS_thresh = LSS_thresh[0] if env.now == 0 else AGENTS[env.now - 1][name]['LSS_thresh'][0]
        _memory = memory[0] if env.now == 0 else AGENTS[env.now - 1][name]['memory'][0]
        _discount = discount[0] if env.now == 0 else AGENTS[env.now - 1][name]['discount'][0]
        _impatience = impatience[0] if env.now == 0 else AGENTS[env.now - 1][name]['impatience'][0]
        _past_weight = past_weight[0] if env.now == 0 else AGENTS[env.now - 1][name]['past_weight'][0]
        _strategy = strategy[0] if env.now == 0 else AGENTS[env.now - 1][name]['strategy'][0]
        value = decision_var
        profits = 0  # in order to get the profits of this period alone

        #################################################################
        #                                                               #
        #    First, the Technology provider closes any new deals and    #
        #                        collect profits                        #
        #                                                               #
        #################################################################

        dd_profits[Technology['source']] = 0

        if env.now > 0 and len(CONTRACTS[env.now - 1]) > 0:
            for _ in CONTRACTS[env.now - 1]:
                i = CONTRACTS[env.now - 1][_]
                if i['receiver'] == name and i['status'] == 'payment':
                    wallet += i['value']
                    profits += i['value']
                    dd_profits[Technology['source']] += i['value']
                    prod_cap_pct[1] += i['value']

        #################################################################
        #                                                               #
        #    Now, the Technology provider gets any incentive that is    #
        #                        available to it                        #
        #                                                               #
        #################################################################

        if env.now > 0:
            for _ in CONTRACTS[env.now - 1]:
                i = CONTRACTS[env.now - 1][_]
                if i['receiver'] == name and i['sender'] == 'TPM':
                    incentive = i['value']
                    if i['bound'] == 'capacity':
                        capacity += incentive
                    elif i['bound'] == 'innovation':
                        RandD += incentive
                    else:
                        wallet += incentive

        #################################################################
        #                                                               #
        # If it is the end of the year, then the TP shares its profits  #
        #                     with its shareholders                     #
        #                                                               #
        #################################################################

        if env.now % 12 == 0 and env.now > 0:
            profits_to_shareholders = wallet*(1-value)
            wallet -= profits_to_shareholders
            shareholder_money += profits_to_shareholders

        #################################################################
        #                                                               #
        #         The TP then has to 1) adjust the base capex to        #
        #         the productive capacity (if the technology is         #
        #  non-transportable; 2) change the strategy if the verdict was  #
        #  to change it; 3) spend the available money into imitation,   #
        # innovation or productive capacity; 4) check if the TP reached #
        #     the threshold of innovation/imitation and change the      #
        #    Technology dictionary if it reached that; 5) update the    #
        #    global TECHNOLOGY dictionary; and lastly 6) do the self    #
        #           NPV entry for the AGENTS dictionary later.          #
        #           8) Oh, it must also the capping process             #
        #                                                               #
        #################################################################

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
            new_capex = max(min(base_capex, capacity_threshold * base_capex * 10**9 / capacity), base_capex/2)
            Technology.update({
                'CAPEX': new_capex
            })
        else:
            """ the technology is transportable (e.g. solar panels)"""
            i = Technology['CAPEX']
            Technology.update({"base_CAPEX": i})

        """3) now, if the TP has money, it will spend it on either capacity, imitation or innovation"""
        # ic(name, wallet, value, env.now)
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
        a = 0
        if RandD > RnD_threshold: # and (strategy == 'innovation' or strategy == 'imitation'):
            """ then we get the 'a' which can either be a poisson + normal for innovation, or a simple binomial.
             Values above or equal (for the imitation) 1 indicate that innovation or imitation occured """
            tech_age = starting_tech_age + innovation_index
            a = (1/tech_age) * np.random.poisson(1) + (1 - 1/tech_age) * np.random.normal(0, 1)

            """ if we reached the threshold, then we to set the bar of the RnD """
            RnD_threshold += random.uniform(RandD, RandD*a)

            if a > 1:
                innovation_index += a
                """ we are dealing with innovation """
                true_innovation_index += a
                RnD_threshold *= rNd_INCREASE * a
                """ we have to check where did the innovation occur"""
                what_on = random.choice(['base_CAPEX', 'OPEX', 'MW'])
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
                else:
                    """ we did not reach over that threshold, so it was a marginal innovation """
                    radical_or_not = 'last_marginal_innovation'
                Technology.update({
                    radical_or_not: env.now
                })

        """6) lastly, we have to get the self.NPV of its current technology"""
        if env.now > 0:
            i = TECHNOLOGIC[env.now - 1][name]
            price = weighting_FF(env.now - 1, 'price', 'MWh', MIX)
            self_NPV.update({
                'value': npv_generating_FF(r, i['lifetime'], 1, i['MW'], i['building_time'], i['CAPEX'], i['OPEX'],
                                           price[i['source']], i['CF'], AMMORT), 'MWh': i['MW']
            })

        """ 8) we must also check if the capping process is on"""

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

        #################################################################
        #                                                               #
        #          And then, the TP will decide what to do next         #
        #                                                               #
        #################################################################
        if env.now > 0:
            decision_var = max(0, min(1, private_deciding_FF(name)))
            decisions = evaluating_FF(name)
            verdict = decisions['verdict']

        #############################################################
        #                                                           #
        #  Before leaving, the agent must update the outside world  #
        #                                                           #
        #############################################################

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
                  "impatience": impatience,
                  "past_weight": past_weight,
                  "LSS_thresh": LSS_thresh,
                  "memory": memory,
                  "discount": discount,
                  "strategy": strategy,
                  "innovation_index": innovation_index,
                  "shareholder_money": shareholder_money,
                  "true_innovation_index": true_innovation_index,
                  "LSS_tot": LSS_tot,
                  "prod_cap_pct": prod_cap_pct
                  }

        if env.now > 1:
            update['impatience'][0] = max(1, update['impatience'][0] + decisions['impatience_increase'])
            update["LSS_weak"] = LSS_tot + decision_var - AGENTS[env.now - 1][name]['decision_var']
        else:
            update["LSS_weak"] = LSS_tot

        AGENTS[env.now][name] = update.copy()
        if env.now > 0:
            post_evaluating_FF(decisions['strikes'], verdict, name, strikables_dict)
            # profits_dedicting_FF(name)

        yield env.timeout(1)


class TPM(object):
    def __init__(self, env, wallet, boundness_cost, instrument, source, decision_var, LSS_thresh, impatience,
                 disclosed_thresh, past_weight, memory, discount, policies, rationale):
        self.env = env
        self.genre = 'TPM'
        self.subgenre = 'TPM'
        self.name = 'TPM'
        self.wallet = wallet
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
                           "past_weight": past_weight} # ,
                           # 'instrument': instrument
                           # }

        self.strikables_dict = strikable_dicting(strikables_dict)

        self.LSS_tot = 0

        self.action = env.process(run_TPM(self.genre,
                                          self.subgenre,
                                          self.name,
                                          self.wallet,
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
                                          self.LSS_tot))


def run_TPM(genre,
            subgenre,
            name,
            wallet,
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
            LSS_tot):
    CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, POLICY_EXPIRATION_DATE, AMMORT, INSTRUMENT_TO_SOURCE_DICT, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.POLICY_EXPIRATION_DATE, config.AMMORT, config.INSTRUMENT_TO_SOURCE_DICT, config.env

    while True:

        #################################################################
        #                                                               #
        #              First, the TPM checks if it got an               #
        #            strike, adds or changes its main policy            #
        #                                                               #
        #################################################################

        _LSS_thresh = LSS_thresh[0] if env.now == 0 else AGENTS[env.now - 1][name]['LSS_thresh'][0]
        _past_weight = past_weight[0] if env.now == 0 else AGENTS[env.now - 1][name]['past_weight'][0]
        _source = list(source[0].keys())[0] if env.now == 0 else list(AGENTS[env.now - 1][name]['source'][0].keys())[0]
        _memory = memory[0] if env.now == 0 else AGENTS[env.now - 1][name]['memory'][0]
        _discount = discount[0] if env.now == 0 else AGENTS[env.now - 1][name]['discount'][0]
        _impatience = impatience[0] if env.now == 0 else AGENTS[env.now - 1][name]['impatience'][0]
        _rationale = rationale[0] if env.now == 0 else AGENTS[env.now - 1][name]['rationale'][0]
        _instrument = instrument[0] if env.now == 0 else AGENTS[env.now - 1][name]['instrument'][0]
        value = disclosed_var

        #################################################################
        #                                                               #
        #             Now, the TPM gives out the incentives             #
        #                                                               #
        #################################################################

        policy_pool = [{'instrument': _instrument,
                        'source': _source,
                        'budget': value * wallet}]

        if len(policies) > 0:
            policy_pool.append(policies)  # with this we a temporary list with first the current policy and afterwards
            # all the other policies

        if env.now >= 2:
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

        if env.now > 2:
            add_source = source_reporting_FF(name, _past_weight)
            for entry in range(len(source) - 1):
                source[entry][list(source[entry].keys())[0]] *= (1 - _discount)
                source[entry][list(source[entry].keys())[0]] += add_source[list(source[entry].keys())[0]]

        #################################################################
        #                                                               #
        #         And then, the TPM will decide what to do next         #
        #                                                               #
        #################################################################

        if env.now > 0:
            decision_var = max(0, min(1, public_deciding_FF(name)))
            disclosed_var = thresholding_FF(_LSS_thresh, disclosed_var, decision_var)
            decisions = evaluating_FF(name)
            verdict = decisions['verdict']

        #################################################################
        #                                                               #
        #    Before leaving, the agent must uptade the outside world    #
        #                                                               #
        #################################################################

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
            "impatience": impatience,
            "disclosed_thresh": disclosed_thresh,
            "past_weight": past_weight,
            "memory": memory,
            "discount": discount,
            "policies": policies,
            "rationale": rationale,
            "current_state": current_stating_FF(_rationale),
            "LSS_tot": LSS_tot
        }

        if env.now > 1:
            update['impatience'][0] = max(1, update['impatience'][0] + decisions['impatience_increase'])
            update["LSS_weak"] = LSS_tot + decision_var - AGENTS[env.now - 1][name]['decision_var']
        else:
            update["LSS_weak"] = LSS_tot

        AGENTS[env.now][name] = update.copy()
        if env.now > 0:
            post_evaluating_FF(decisions['strikes'], verdict, name, strikables_dict)

        yield env.timeout(1)


class EPM(object):
    def __init__(self, env, wallet, PPA_expiration, PPA_limit, COUNTDOWN, decision_var, auction_capacity, instrument, source, LSS_thresh, impatience, disclosed_thresh, past_weight, memory, discount, policies, rationale):
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

        self.LSS_tot = 0
        strikables_dict = {'impatience': impatience,
                           'LSS_thresh': LSS_thresh,
                           'source': source,
                           'past_weight': past_weight,
                           'memory': memory,
                           'discount': discount,
                           'disclosed_thresh': disclosed_thresh,
                           'rationale': rationale,
                           'instrument': instrument
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
                                          self.strikables_dict))


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
            strikables_dict):

    CONTRACTS, MIX, AGENTS, TECHNOLOGIC, DEMAND, AUCTION_WANTED_SOURCES, AMMORT, INSTRUMENT_TO_SOURCE_DICT, MARGIN, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.DEMAND, config.AUCTION_WANTED_SOURCES, config.AMMORT, config.INSTRUMENT_TO_SOURCE_DICT, config.MARGIN, config.env  # globals

    while True:

        #################################################################
        #                                                               #
        #              First, the TPM checks if it got an               #
        #            strike, adds or changes its main policy            #
        #                                                               #
        #################################################################

        _LSS_thresh = LSS_thresh[0] if env.now == 0 else AGENTS[env.now - 1][name]['LSS_thresh'][0]
        _impatience = impatience[0] if env.now == 0 else AGENTS[env.now - 1][name]['impatience'][0]
        _source = list(source[0].keys())[0] if env.now == 0 else list(AGENTS[env.now - 1][name]['source'][0].keys())[0]
        _past_weight = past_weight[0] if env.now == 0 else AGENTS[env.now - 1][name]['past_weight'][0]
        _memory = memory[0] if env.now == 0 else AGENTS[env.now - 1][name]['memory'][0]
        _discount = discount[0] if env.now == 0 else AGENTS[env.now - 1][name]['discount'][0]
        _disclosed_thresh = disclosed_thresh[0] if env.now == 0 else AGENTS[env.now - 1][name]['disclosed_thresh'][0]
        _rationale = rationale[0] if env.now == 0 else AGENTS[env.now - 1][name]['rationale'][0]
        _instrument = instrument[0] if env.now == 0 else AGENTS[env.now - 1][name]['instrument'][0]
        value = disclosed_var

        #################################################################
        #                                                               #
        #           First, the EPM adds or changes its policy           #
        #                                                               #
        #################################################################

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

                        max_auction = 2000
                        min_auction = 800

                        remaining_capacity = max(
                            min(auction_capacity * entry_value * 1000 + AGENTS[env.now-1]['DD']['Remaining_demand']/24*30 + 25 * 12 * (DEMAND[env.now-1]-DEMAND[env.now-2]),
                                max_auction),
                            min_auction)

                        """
                         The auction size is between 0,8 GW and 2 GW, being made of three parts:
                         
                         1- the auction capacity (in GW) times the decision_var
                         2- the remanining demand (surpluses reduce the auction size)
                         3- the expected increase in demand for the next 25 years
                         
                        """

                        for project in possible_projects:
                            if remaining_capacity > 0:
                                """ if there is still capacity to contract, then the project is contracted"""
                                contracted_projects.append(project)
                            remaining_capacity -= project['capacity']
                        """ then we cycle through the codes of the contracted projects in order to tell the proponent 
                        agents that their projects were contracted"""
                        for project in contracted_projects:
                            """ first we update it to include attributes exclusive to PPAs, such as the auction price, 
                            the boolean auction_contracted and the date of expiration of the price"""
                            code = project['code']
                            project['receiver'] = project['sender']  # we have to switch them
                            """if math.isnan(price) is True:
                                print(project)
                                print(price, project['OPEX'], project['CAPEX'], project['lifetime'], MARGIN, project['MWh'])
                                print('check this out', math.isnan(project['OPEX']), math.isnan(project['CAPEX']), math.isnan(project['lifetime']), math.isnan(MARGIN), math.isnan(project['MWh']))"""
                            project_update = {
                                'status': 'project',
                                'auction_contracted': True,
                                'price_expiration': env.now + PPA_expiration + PPA_limit + 1,
                                'limit': env.now + PPA_limit + 1,
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

        #################################################################
        #                                                               #
        #    Now, the EPM analyses what happened to the system due to   #
        #                       its intervention                        #
        #                                                               #
        #################################################################

        if env.now > 2:
            add_source = source_reporting_FF(name, _past_weight)
            for entry in range(len(source) - 1):
                source[entry][list(source[entry].keys())[0]] *= (1 - _discount)
                source[entry][list(source[entry].keys())[0]] += add_source[list(source[entry].keys())[0]]

        #################################################################
        #                                                               #
        #         And then, the EPM will decide what to do next         #
        #                                                               #
        #################################################################

        if env.now > 0:
            decision_var = max(0, min(1, public_deciding_FF(name)))
            disclosed_var = thresholding_FF(_LSS_thresh, disclosed_var, decision_var)
            decisions = evaluating_FF(name)
            verdict = decisions['verdict']

        #################################################################
        #                                                               #
        #    Before leaving, the agent must uptade the outside world    #
        #                                                               #
        #################################################################

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
            "impatience": impatience,
            "disclosed_thresh": disclosed_thresh,
            "past_weight": past_weight,
            "memory": memory,
            "discount": discount,
            "policies": policies,
            "rationale": rationale,
            "LSS_tot": LSS_tot,
            "strikables_dict": strikables_dict,
            "current_state": current_stating_FF(_rationale)
        }

        if env.now > 1:
            update['impatience'][0] = max(1, update['impatience'][0] + decisions['impatience_increase'])
            update["LSS_weak"] = LSS_tot + decision_var - AGENTS[env.now - 1][name]['decision_var']
        else:
            update["LSS_weak"] = LSS_tot

        AGENTS[env.now][name] = update.copy()
        if env.now > 0:
            post_evaluating_FF(decisions['strikes'], verdict, name, strikables_dict)

        yield env.timeout(1)


class DBB(object):
    def __init__(self, env, name, wallet, instrument, source, decision_var, LSS_thresh, past_weight,
                 memory, discount, policies, impatience, disclosed_thresh, rationale):
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
        self.wallet = wallet
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
                           'instrument': instrument
                           }

        self.strikables_dict = strikable_dicting(strikables_dict)

        self.action = env.process(run_DBB(
            self.NPV_THRESHOLD_DBB,
            self.guaranteed_contracts,
            self.genre,
            self.name,
            self.wallet,
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
            self.LSS_tot))


def run_DBB(NPV_THRESHOLD_DBB,
            guaranteed_contracts,
            genre,
            name,
            wallet,
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
            LSS_tot):

    global decisions
    CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, POLICY_EXPIRATION_DATE, INSTRUMENT_TO_SOURCE_DICT, RISKS, AGENTS_r, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.POLICY_EXPIRATION_DATE, config.INSTRUMENT_TO_SOURCE_DICT, config.RISKS, config.AGENTS_r, config.env  # globals

    while True:

        #################################################################
        #                                                               #
        #              First, the DBB checks if it got an               #
        #            strike, adds or changes its main policy            #
        #                                                               #
        #################################################################

        _LSS_thresh = LSS_thresh[0] if env.now == 0 else AGENTS[env.now - 1][name]['LSS_thresh'][0]
        _past_weight = past_weight[0] if env.now == 0 else AGENTS[env.now - 1][name]['past_weight'][0]
        _source = list(source[0].keys())[0] if env.now == 0 else list(AGENTS[env.now - 1][name]['source'][0].keys())[0]
        _memory = memory[0] if env.now == 0 else AGENTS[env.now - 1][name]['memory'][0]
        _discount = discount[0] if env.now == 0 else AGENTS[env.now - 1][name]['discount'][0]
        _impatience = impatience[0] if env.now == 0 else AGENTS[env.now - 1][name]['impatience'][0]
        _rationale = rationale[0] if env.now == 0 else AGENTS[env.now - 1][name]['rationale'][0]
        _instrument = instrument[0] if env.now == 0 else AGENTS[env.now - 1][name]['instrument'][0]
        LSS_tot = LSS_tot if env.now == 0 else AGENTS[env.now - 1][name]['LSS_tot']
        value = disclosed_var

        #################################################################
        #                                                               #
        #    Before doing anything, the development bank must collect   #
        #      its  payments and guarantee contracts that need be       #
        #                                                               #
        #################################################################

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

        #################################################################
        #                                                               #
        #           First, the DBB adds or changes its policy           #
        #                                                               #
        #################################################################

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

                if entry_instrument == 'finance':
                    # print('DBB is trying to finance')
                    financing = financing_FF(genre, name, wallet, receivable, entry_value, financing_index,
                                             accepted_source=entry_chosen_source)

                    wallet = financing['wallet']
                    receivable = financing['receivables']

                elif entry_instrument == 'guarantee':
                    """ We are dealing with guarantees """
                    financing = financing_FF(genre, name, wallet, receivable, entry_value, financing_index,
                                             guaranteeing=True, accepted_source=entry_chosen_source)

                    wallet = financing['wallet']
                    receivable = financing['receivables']
                    financing_index = financing['financing_index']

        #################################################################
        #                                                               #
        #    Now, the DBB analyses what happened to the system due to   #
        #                       its intervention                        #
        #                                                               #
        #################################################################

        if env.now > 2:
            add_source = source_reporting_FF(name, _past_weight)
            for entry in range(len(source) - 1):
                source[entry][list(source[entry].keys())[0]] *= (1 - _discount)
                source[entry][list(source[entry].keys())[0]] += add_source[list(source[entry].keys())[0]]

        #################################################################
        #                                                               #
        #         And then, the DBB will decide what to do next         #
        #                                                               #
        #################################################################

        if env.now > 0:
            decision_var = max(0, min(1, public_deciding_FF(name)))
            disclosed_var = thresholding_FF(_LSS_thresh, disclosed_var, decision_var)
            decisions = evaluating_FF(name)
            verdict = decisions['verdict']
            # print('strikes are', decisions['strikes'])

        #################################################################
        #                                                               #
        #    Before leaving, the agent must uptade the outside world    #
        #                                                               #
        #################################################################

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
            "impatience": impatience,
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
            "interest_rate": 1+r-disclosed_var
        }

        if env.now > 1:
            update['impatience'][0] = max(1, update['impatience'][0] + decisions['impatience_increase'])
            update["LSS_weak"] = LSS_tot + decision_var - AGENTS[env.now - 1][name]['decision_var']
        else:
            update["LSS_weak"] = LSS_tot

        AGENTS[env.now][name] = update.copy()
        if env.now > 0:
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

        #################################################################
        #                                                               #
        #     Before anything, we must the current values of each of    #
        #        the dictionaries that we use and other variables       #
        #                                                               #
        #################################################################

        list_of_strikables = [dd_kappas, dd_qual_vars, dd_backwardness, dd_avg_time, dd_discount, dd_strategies]

        _source = source[0]
        _LSS_thresh = LSS_thresh[0]
        _past_weight = past_weight[0]
        _memory = memory[0]
        _discount = dd_discount['current']
        _impatience = impatience[0]
        value = decision_var
        profits = 0  # in order to get the profits of this period alone

        #################################################################
        #                                                               #
        #                First, the bank collect profits                #
        #                                                               #
        #################################################################

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

        #################################################################
        #                                                               #
        # If it is the end of the year, then the BB shares its profits  #
        #                     with its shareholders                     #
        #                                                               #
        #################################################################

        if env.now % 12 == 0 and env.now > 0:
            profits_to_shareholders = wallet * (1 - value)
            wallet -= profits_to_shareholders
            shareholder_money += profits_to_shareholders

        #################################################################
        #                                                               #
        #    Now, on to check if change is on and if there is a strike  #
        #                                                               #
        #################################################################

        if env.now > 0 and (verdict == 'add' or 'change'):
            striked = striking_FF(list_of_strikables, kappa)

            for entry in range(0, len(list_of_strikables)):
                list_of_strikables[entry] = striked[entry]
                if entry == 'source':
                    # we changed the source, so we have to update the accepted_sources dictionary
                    source_accepting_FF(accepted_sources, source)
            verdict = 'keep'  # we already changed, now back to business

        #################################################################
        #                                                               #
        #        Then, the bank decides which projects to accept        #
        #                                                               #
        #################################################################

        if env.now > 0 and len(CONTRACTS[env.now - 1]) > 0:
            financing = financing_FF(genre, name, wallet, receivable, value, financing_index)

            wallet = financing['wallet']
            receivable = financing['receivables']
            financing_index = financing['financing_index']

        #################################################################
        #                                                               #
        #    Now, the BB analyses what happened to the system due to    #
        #                       its intervention                        #
        #                                                               #
        #################################################################

        add_source = source_reporting_FF(name)

        for entry in dd_source['ranks']:
            dd_source['ranks'][entry] *= (1 - discount)
            dd_source['ranks'][entry] += add_source[entry]

        #################################################################
        #                                                               #
        #          And then, the BB will decide what to do next         #
        #                                                               #
        #################################################################

        if env.now > 0:
            decision_var = max(0, min(1, private_deciding_FF(name)))
            decisions = evaluating_FF(name)

        #################################################################
        #                                                               #
        #    Before leaving, the agent must uptade the outside world    #
        #                                                               #
        #################################################################

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
        if env.now > 0:
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
                           'source': source,
                           'past_weight': past_weight,
                           'memory': memory,
                           'discount': discount}

        self.strikables_dict = strikable_dicting(strikables_dict)

        self.strikables_dict = strikables_dict

        self.profits = 0

        self.LSS_tot = 0
        self.shareholder_money = 0
        self.reinvest = True

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

        #################################################################
        #                                                               #
        #     Before anything, we must the current values of each of    #
        #        the dictionaries that we use and other variables       #
        #                                                               #
        #################################################################

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

        #############################################################
        #                                                           #
        #                    Collecting profits                     #
        #                                                           #
        #############################################################

        if env.now > 0 and len(MIX[env.now - 1]) > 0:
            """ The EP only collects profits if it produces energy and that can only happen if we are not in the start of the simulation, and if the MIX is not empty"""
            """ Moreover, if the plant was activated, then the EP pays the OPEX of it """
            for _ in MIX[env.now - 1]:
                """ Then we check each plant in the mix """
                i = MIX[env.now - 1][_].copy()
                """ the _ is the code of the plant, whereas i is the dictionary of the plant itself """
                if i['EP'] == name and i['status'] == 'contracted':
                    """ if the plant is mine and it is contracted, I'll collect profits """
                    addition = i['MWh'] * i['price'] - i['OPEX']
                    wallet += addition
                    profits += addition
                    """if math.isnan(i['MWh'] * i['price'] - i['OPEX']) is True:
                        print(name, math.isnan(i['MWh']), math.isnan(i['price']), math.isnan(i['OPEX']))
                        print(i)"""
                    dd_profits[i['source']] += addition

                    """ we also have to put the profit as a contract in the CONTRACTS dictionary in order for the 
                    policy makers, other EPs and the demand to do some calculations """
                    code = uuid.uuid4().int
                    """ this is to get a unique and random number """
                    j = i.copy()
                    """ and we also have to create a copy of the i dictionary, if not things will update on that 
                    dictionary, and that's no good """
                    j.update({
                        'status': 'payment',
                        'sender': 'D',
                        'source': j['source'],
                        'receiver': name,
                        'value': j['MWh'] * j['price']
                    })
                    """ and now we update"""
                    CONTRACTS[env.now - 1][code] = j

        #################################################################
        #                                                               #
        # If it is the end of the year, then the EP shares its profits  #
        #                     with its shareholders                     #
        #                                                               #
        #################################################################

        if env.now % 12 == 0 and env.now > 0:
            profits_to_shareholders = wallet * (1 - value)
            wallet -= profits_to_shareholders
            shareholder_money += profits_to_shareholders

        #############################################################
        #                                                           #
        #  Now the EP goes through its portfolio_of_plants in order #
        #   to 1) pay the banks for the financing, 2) check if      #
        #     plants finished building, 3) retire old plants and    #
        #               4) insert plants into the mix               #
        #                                                           #
        #############################################################

        if len(portfolio_of_plants) > 0:
            for _ in portfolio_of_plants:
                i = portfolio_of_plants[_]
                number = ((i['principal'] / (1 + AMMORT)) + i['principal'] * r)

                """  1) we pay for the ammortization of plants """

                if i['ammortisation'] > env.now and (
                        i['guarantee'] is not True or (i['guarantee'] is True and wallet >= number)
                ) and i['BB'] != 'reinvestment':
                    """ if the plant is not guaranteed or if it's guaranteed but the EP has enough money to cover it """
                    wallet -= number

                    code = uuid.uuid4().int
                    CONTRACTS[env.now].update({
                        code: {
                            'sender': name,
                            'receiver': i['BB'],
                            'status': 'payment',
                            'source': i['source'],
                            'value': number
                        }
                    })
                elif i['ammortisation'] > env.now and i['guarantee'] is True and wallet < number:
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

                """ 3) and we check if plants finished building """

                if i['completion'] <= env.now:
                    i['status'] = 'built'

                """ 4) now we insert built plants into the mix  """

                if i['status'] == 'built':
                    j = i.copy()
                    MIX[env.now][_] = j

        #############################################################
        #                                                           #
        #      Now, the EP has to check if projects that were       #
        #     auction contracted or had guarantees got financed.    #
        #         If not, they need to be inserted into the         #
        #      portfolio_of_projects dictionary in order to be      #
        #       retired each period until the TOLERANCE limit       #
        #                                                           #
        #############################################################

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
                                (i['CAPEX'] < wallet and i['status'] == 'rejected') or
                                ('guarantee' in i or 'auction_contracted' in i)
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

        #############################################################
        #                                                           #
        #   Then, the Energy producer decides how much to invest    #
        #                                                           #
        #############################################################
        if env.now > 0 and env.now % periodicity == 0:
            mix_expansion = ((
                                     config.INITIAL_DEMAND * value + (AGENTS[env.now - 1]['DD']['Remaining_demand'] / (24 * 30))
                             ) / EP_NUMBER)
            # ic(mix_expansion, value) if env.now> config.FUSS_PERIOD else None
            # ic(AGENTS[env.now-1]['DD']['Remaining_demand'], value)
            """if AGENTS[env.now-1]['DD']['Remaining_demand'] > 0:
                condition = True
            else:
                condition = value > 0"""
            condition = True
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
                  'NPV': False,
                  'Lumps': 0,
                  'CAPEX': 0,
                  'OPEX': 0
                  }
            for _ in TECHNOLOGIC[env.now - 1]:
                i = TECHNOLOGIC[env.now - 1][_]
                if i['source'] == _source:
                    source_price = weighting_FF(env.now - 1, 'price', 'MWh', MIX)
                    Lumps = min(max(1, np.ceil(mix_expansion / i['MW'])), max_lump[_source])
                    price = source_price[i['source']]
                    NPV = npv_generating_FF(
                        r, i['lifetime'], Lumps, Lumps * i['MW'], i['building_time'], i['CAPEX'], i['OPEX'], price,
                        i['CF'], AMMORT
                    )
                    if NPV > _TP['NPV'] or _TP['NPV'] is False:
                        _TP.update({
                            'TP': _,
                            'NPV': NPV,
                            'Lumps': Lumps,
                            'CAPEX': i['CAPEX'] * Lumps,
                            'OPEX': i['OPEX'] * Lumps,
                            'source_of_TP': i['source']
                        })

            # OPEX and CAPEX are in relation to one lump, so in the project we have to change them to account for the
            # whole project
            # ic(TP['TP'], name, _source) if TP['TP'] == 0 else None
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
                'status': 'project',
                'capacity': _TP['Lumps'] * project['MW'],
                'MWh': _TP['Lumps'] * project['MW'] * 24 * 30 * project['CF'],
                'avoided_emissions': TECHNOLOGIC[env.now - 1][_TP['TP']]['avoided_emissions'] * _TP['Lumps'],
                'emissions': TECHNOLOGIC[env.now - 1][_TP['TP']]['emissions'] * _TP['Lumps'],
                'guarantee': False
            })
            if _TP['source_of_TP'] in AUCTION_WANTED_SOURCES:
                # print('sending to EPM for auction')
                project.update(
                    {'status': 'bidded',
                     'price': (
                                      project['OPEX'] + (project['CAPEX'] / project['lifetime'])
                              ) * (2 + config.MARGIN - value) / project['MWh'],
                     'receiver': 'EPM'})
            else:
                project['status'] = 'project'
                project['auction_contracted'] = False
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

        # print(portfolio_of_projects, name)

        _to_pop = []
        for _ in portfolio_of_projects:
            """ now we have to resend the "projects in the portfolio_of_projects dictionary """
            i = portfolio_of_projects[_].copy()

            if 'limit' not in i:
                i['limit'] = env.now + 1 + _tolerance

            if i['limit'] == env.now:
                # print('project ', _, ' has reached its limit time...')
                _to_pop.append(_)

            elif i['CAPEX'] > wallet:
                receiver = bank_sending_FF()
                project = i.copy()
                project.update({'sender': name,
                                'receiver': receiver,
                                'status': 'project'})
                CONTRACTS[env.now].update({
                    _: project
                })
            elif i['CAPEX'] < wallet and reinvest is True:
                # print(name, 'has just reinvested, and this is the status', i['status'])
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
                    'ammortisation': env.now
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

        #################################################################
        #                                                               #
        #    Now, the EP analyses what happened to the system due to    #
        #                       its intervention                        #
        #                                                               #
        #################################################################
        if env.now > 0:
            add_source = source_reporting_FF(name, _past_weight, index)
            for entry in range(len(source) - 1):
                source[entry][list(source[entry].keys())[0]] *= (1 - _discount)

                source[entry][list(source[entry].keys())[0]] += add_source[list(source[entry].keys())[0]]

        #################################################################
        #                                                               #
        #          And then, the EP will decide what to do next         #
        #                                                               #
        #################################################################
        if env.now > 0:
            decision_var = max(0, min(1, private_deciding_FF(name)))
            decisions = evaluating_FF(name)
            verdict = decisions['verdict']

        #############################################################
        #                                                           #
        #  Before leaving, the agent must update the outside world  #
        #                                                           #
        #############################################################
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
                  "impatience": impatience,
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
            update["LSS_weak"] = LSS_tot + decision_var - AGENTS[env.now - 1][name]['decision_var']
        else:
            update["LSS_weak"] = LSS_tot

        AGENTS[env.now][name] = update.copy()

        if env.now > 0:
            post_evaluating_FF(decisions['strikes'], verdict, name, strikables_dict)
            profits_dedicting_FF(name)

        yield env.timeout(1)


class Demand(object):
    def __init__(self, env, initial_demand, when, increase):
        self.env = env
        self.genre = 'DD'
        self.name = 'DD'
        self.initial_demand = initial_demand
        self.when = when
        self.increase = increase
        self.Demand_by_source = {0: 0, 1:0, 2:0}
        self._Price = 0
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

        #################################################################
        #                                                               #
        #                Which plants will be contracted?               #
        #                                                               #
        #################################################################

        if env.now == 0:
            DEMAND.update({env.now: initial_demand})
            # printable = 'seed is ' + str(config.seed)
            # print(printable)

        elif env.now % when == 0:
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

            DEMAND.update({env.now: DEMAND[env.now - 1] + increase})

        else:
            DEMAND.update({env.now: DEMAND[env.now - 1]})

        """ since the policy makers act after private agents, they are looking at the env.now, not the env.now-1 """
        demand = DEMAND.copy()[env.now] * 24 * 30
        if env.now > 0 and len(MIX[env.now]) > 0:
            """ 
            First, we contract and precify the electricity projects
            """

            possible_projects = []
            for _ in MIX[env.now]:
                """ we build the list of possible projects, i.e., projects deployed or already built"""
                j = MIX[env.now][_]
                if j['status'] == 'built' or j['status'] == 'contracted':
                    possible_projects.append(j)
            """ now, we sort the list of dictionaries in terms of dispatchability and then OPEX (respecting the merit order)"""
            possible_projects = sorted(possible_projects, key=lambda x: (x['dispatchable'], x['OPEX']))
            chosen = []
            # print('full demand is ', demand)
            for plant in possible_projects:
                if demand < 0:
                    """ if there is no more demand to be supplied, then the plant is not contracted"""
                    MIX[env.now][plant['code']].update({
                        'status': 'built'
                    })
                    # print(MIX[env.now][plant['code']], 'was not contracted')
                else:
                    """ if there is still demand to to be supplied, then, the power plant is contracted"""
                    MIX[env.now][plant['code']].update({
                        'status': 'contracted'
                    })
                    chosen.append(plant)
                    demand -= plant['MWh']
                    Demand_by_source[plant['source']] += plant['MWh']
                    # print('demand decrease by', plant['MWh'])
            """ following the merit order, the system is precified in relation to its most costly unit"""
            chosen_plant = sorted(chosen, key=lambda x: x['OPEX'])[-1]
            price = (
                            chosen_plant['OPEX'] + (chosen_plant['CAPEX'] / chosen_plant['lifetime'])
                     ) * (1 + MARGIN) / chosen_plant['MWh']
            for i in possible_projects:
                """ if the plant is auction contracted, then we do not mess with its price"""
                if 'auction_contracted' not in i or i['auction_contracted'] is False:
                    MIX[env.now][i['code']].update({
                        'price': price})

            _Price = weighting_FF(env.now, 'price', 'MWh', MIX) if len(MIX[env.now]) > 0 else config.STARTING_PRICE

        AGENTS[env.now][name] = {
            'name': name,
            'genre': 'DD',
            'Demand': DEMAND[env.now],
            'Remaining_demand': demand,
            'Demand_by_source': Demand_by_source,
            'Price': _Price}

        yield env.timeout(1)


class Hetero(object):
    def __init__(self, env, hetero_threshold):
        self.env = env
        self.genre = 'HH'
        self.name = 'HH'
        self.hetero_threshold = hetero_threshold
        self.randomly = False if config.INITIAL_RANDOMNESS < 1 else True  # just to put more emphasis on the
        # randomness of the random runs
        self.action = env.process(run_HH(self.env,
                                         self.genre,
                                         self.name,
                                         self.hetero_threshold,
                                         self.randomly))


def run_HH(env,
           genre,
           name,
           hetero_threshold,
           randomly
           ):
    CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, DEMAND, MARGIN,  EP_NUMBER, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.DEMAND, config.MARGIN, config.EP_NUMBER, config.env

    while True:

        #################################################################
        #                                                               #
        #                How many public agents are there?              #
        #                                                               #
        #################################################################

        list_of_pm = []

        for _ in AGENTS[env.now]:
            agent = AGENTS[env.now][_]

            if agent['genre'] in ['DBB', 'EPM']:
                list_of_pm.append(agent)

        if len(list_of_pm) > 1:

            # That means we have a EPM and a DBB

            #################################################################
            #                                                               #
            #                Perform the heterogeneity check                #
            #                                                               #
            #################################################################

            # We are saving the results of the homogeneity to the demand object, just to avoid creating a new entry
            # on the agents dictionary

            # above the threshold, so we have to homogenize things
            list_o_entries = ['source', 'disclosed_var', 'LSS_thresh', 'past_weight', 'memory', 'discount',
                              'disclosed_thresh']
            chosen_entries = []
            for entry in list_o_entries:
                if entry == 'disclosed_var':
                    if AGENTS[env.now]['BNDES'][entry] != AGENTS[env.now]['EPM'][entry]:
                        # print(AGENTS[env.now]['BNDES'][entry], AGENTS[env.now]['EPM'][entry])
                        chosen_entries.append(entry)
                else:
                    if AGENTS[env.now]['BNDES'][entry][0] != AGENTS[env.now]['EPM'][entry][0]:
                        # print(AGENTS[env.now]['BNDES'][entry][0], AGENTS[env.now]['EPM'][entry][0])
                        chosen_entries.append(entry)

            if len(chosen_entries) > 0 and len(chosen_entries)/len(list_o_entries) > hetero_threshold and random.uniform(0, 1) > hetero_threshold:
                # above the threshold and above the heterogeneity test, so we have to homogenize things
                for entry in chosen_entries:
                    if env.now > 0:
                        current_bndes_var =  AGENTS[env.now]['BNDES'][entry][0]     if entry != 'disclosed_var' else AGENTS[env.now]['BNDES'][entry]
                        previous_bndes_var = AGENTS[env.now - 1]['BNDES'][entry][0] if entry != 'disclosed_var' else AGENTS[env.now - 1]['BNDES'][entry]
                        current_epm_var =    AGENTS[env.now]['EPM'][entry][0]       if entry != 'disclosed_var' else AGENTS[env.now]['EPM'][entry]
                        previous_epm_var =   AGENTS[env.now - 1]['EPM'][entry][0]   if entry != 'disclosed_var' else AGENTS[env.now - 1]['EPM'][entry]
                    else:
                        current_bndes_var = previous_bndes_var = current_epm_var = previous_epm_var = 0

                    if randomly is False:
                        who = []
                        if env.now > 0 and current_bndes_var != previous_bndes_var:
                            who.append('BNDES')
                        elif env.now > 0 and current_epm_var != previous_epm_var:
                            who.append('EPM')
                    else:
                        who = []  # just to avoid errors

                    if randomly is True or env.now == 0 or len(who) > 1:
                        # If random is True, then it is randomized
                        # If it's the first period, then it must be random
                        # If both policy makers changed, then it really doesn't matter who follows who

                        """
                        It is best to just copy the whole list to avoid destroying entries
                        """

                        chosen_agent = random.choice(['BNDES', 'EPM'])

                        chosen = AGENTS[env.now][chosen_agent][entry]

                        AGENTS[env.now]['BNDES'][entry] = chosen
                        AGENTS[env.now]['EPM'][entry] = chosen

                        """var = AGENTS[env.now]['BNDES'][entry][0] if 

                        AGENTS[env.now][chosen_agent]['LSS_tot'] -= 1 if env.now > 0 and  else 0
                        AGENTS[env.now][chosen_agent]['LSS_weak'] -= 1 if env.now > 0 and  else 0"""

                    else:

                        # If it's not random, then we have to change to the most updated one

                        who_else = ['EPM', 'BNDES']
                        who_else.remove(who[0])

                        AGENTS[env.now][who_else[0]][entry] = AGENTS[env.now][who[0]][entry]

                        # print(AGENTS[env.now][who_else[0]][entry], AGENTS[env.now][who[0]][entry], 'equal?')

                AGENTS[env.now]['DD']['homo'] = True  # Just to say that things were changed
                # print(env.now, 'HOMO')
            else:
                # Nothing happened
                AGENTS[env.now]['DD']['homo'] = False

        yield env.timeout(1)
