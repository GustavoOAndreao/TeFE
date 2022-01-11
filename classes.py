import random
import feather
import simpy
# !pip install simpy #on colab it must be this pip install thing, dunno why
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


class TP(object):
    def __init__(self, env, name, wallet, capacity, Technology, RnD_threshold, capacity_threshold, dd_source,
                 decision_var, dd_kappas, dd_qual_vars, dd_backwardness, dd_avg_time, dd_discount, cap_conditions,
                 strategy, dd_strategies):
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
        self.subgenre = Technology['source']  # subgenre or source, we used to use subgenre a lot, now it's kind of a
        # legacy. Is a number, 1 is wind, 2 is solar, 4 is biomass and 5 is hydrogen. 0 and 3 are not used because they
        # are the fossil options
        self.wallet = wallet
        self.profits = 0  # profits of the agent at that certain period, is a number
        self.capacity = capacity
        self.RandD = 0  # how much money was put into R&D? Is a number
        self.EorM = Technology['EorM']  # does the agent use electricity or molecules? is a string (either 'E' or 'M')
        self.innovation_index = 0  # index of innovation. Kind of a legacy, was used to analyze innovation
        self.self_NPV = {}  # The NPV of a unit of investment. Is a dictionary:
        # e.g. self_NPV={'value' : 2000, 'MWh' : 30}
        self.RnD_threshold = RnD_threshold
        self.capacity_threshold = capacity_threshold
        self.dd_profits = {0: 0, 1: 0, 2: 0} if Technology['EorM'] == 'E' else {3: 0, 4: 0, 5: 0}  # same as profits,
        # but as dict. Makes accounting faster and simpler
        self.dd_source = dd_source
        self.decision_var = decision_var
        self.action = "keep"  # this is the action variable. It can be either 'keep', 'change' or 'add'
        self.dd_kappas = dd_kappas
        self.dd_qual_vars = dd_qual_vars
        self.dd_backwardness = dd_backwardness
        self.dd_avg_time = dd_avg_time
        self.dd_discount = dd_discount
        self.cap_conditions = cap_conditions
        self.capped = False  # boolean variable to make the capping easier
        self.strategy = strategy
        self.dd_strategies = dd_strategies

        self.action = env.process(self.run_TP(
            self.name,
            self.genre,
            self.subgenre,
            self.wallet,
            self.profits,
            self.capacity,
            self.RandD,
            self.EorM,
            self.innovation_index,
            self.Technology,
            self.self_NPV,
            self.RnD_threshold,
            self.capacity_threshold,
            self.dd_profits,
            self.dd_source,
            self.decision_var,
            self.action,
            self.dd_kappas,
            self.dd_qual_vars,
            self.dd_backwardness,
            self.dd_avg_time,
            self.dd_discount,
            self.cap_conditions,
            self.capped,
            self.dd_strategies))

    def run_TP(self,
               name,
               genre,
               subgenre,
               wallet,
               profits,
               capacity,
               RandD,
               EorM,
               innovation_index,
               Technology,
               self_NPV,
               RnD_threshold,
               capacity_threshold,
               dd_profits,
               dd_source,
               decision_var,
               action,
               dd_kappas,
               dd_qual_vars,
               dd_backwardness,
               dd_avg_time,
               dd_discount,
               cap_conditions,
               capped,
               dd_strategies):

        CONTRACTS, MIX, AGENTS, AGENTS_r, TECHNOLOGIC, TECHNOLOGIC_r, r, TACTIC_DISCOUNT, AMMORT, rNd_INCREASE, RADICAL_THRESHOLD, env = config.CONTRACTS, config.MIX, config.AGENTS, config.AGENTS_r, config.TECHNOLOGIC, config.TECHNOLOGIC_r, config.r, config.TACTIC_DISCOUNT, config.AMMORT, config.rNd_INCREASE, config.RADICAL_THRESHOLD, config.env  # globals

        while True:

            #################################################################
            #                                                               #
            #     Before anything, we must the current values of each of    #
            #        the dictionaries that we use and other variables       #
            #                                                               #
            #################################################################

            list_of_strikeables = [dd_kappas, dd_qual_vars, dd_backwardness, dd_avg_time, dd_discount, dd_strategies]

            source = dd_source['current']
            kappa = dd_kappas['current']
            qual_vars = dd_qual_vars['current']
            backwardness = dd_backwardness['current']
            avg_time = dd_avg_time['current']
            discount = dd_discount['current']
            strategy = dd_strategies['current']

            value = decision_var
            profits = 0  # in order to get the profits of this period alone

            #################################################################
            #                                                               #
            #    First, the Technology provider closes any new deals and    #
            #                        collect profits                        #
            #                                                               #
            #################################################################

            if env.now > 0:
                for _ in CONTRACTS[env.now - 1]:
                    i = CONTRACTS[env.now - 1][_]
                    if i['receiver'] == name and i['status'] == 'payment':
                        wallet += i['value']
                        profits += i['value']
                        """ we also have to update the sales_MWh entry, to indicate to the policy makers how much MWh
                         of each source is there  """
                        j = dd_profits['ranks']
                        j.update({source: j[source] + i['value']})

            #################################################################
            #                                                               #
            #    Now, on to check if change is on and if there is a strike  #
            #                                                               #
            #################################################################

            if env.now > 0 and (action == 'add' or 'change'):
                striked = striking_FF(list_of_strikeables, kappa)

                for entry in range(0, len(list_of_strikeables)):
                    list_of_strikeables[entry] = striked[entry]
                action = 'keep'  # we already changed, now back to business

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
                        wallet += i['value']

            #################################################################
            #                                                               #
            #         The TP then has to 1) adjust the base capex to        #
            #         the productive capacity (if the technology is         #
            #  non-transportable; 2) change the strategy if the action was  #
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
            if Technology['transport'] == False:
                """ if the technology is not transportable, then the productive capacity impacts on the CAPEX """
                if env.now == 0:
                    """ if we are in the first period, then we have to get the starting CAPEX and tell the TP that this
                     is his base capex, because there is no base_capex already """
                    i = Technology['CAPEX']
                    Technology.update({"base_CAPEX": i})
                j = Technology['base_CAPEX']
                """ we have to produce the actual CAPEX, with is the base_CAPEX multiplied by euler's number to the
                 power of the ratio of how many times the base capex is greater than the capacity itself multiplied
                  by the threshold of capacity"""
                new_capex = min(j, (capacity_threshold / capacity) * j)
                Technology.update({
                    'CAPEX': j * new_capex
                })
            else:
                """ the technology is transportable (e.g. solar panels)"""
                i = Technology['CAPEX']
                Technology.update({"base_CAPEX": i})

            """3) now, if the TP has money, it will spend it on either capacity, imitation or innovation"""
            if wallet > 0 and value > 0:
                wallet -= wallet * value
                """ having money, the TP will spend a portion (given by the value) of its wallet on something """
                if strategy == 'innovation':
                    """ if the strategy is to do R&D, then it will do R&D"""
                    RandD += wallet * value
                    innovation_index += wallet * value
                    # print(name, 'got', wallet * value, 'more innovation to its roster, and now innovation is', innovation_index)
                else:
                    """ if not, then it will spend on productive capacity"""
                    capacity += wallet * value
                    # print(name, 'got', wallet * value, 'more capacity to its roster, and now capacity is', capacity)

            """4) we have to check if the TP reached the threshold for innovation or imitation"""
            if RandD > RnD_threshold and (strategy == 'innovation' or strategy == 'imitation'):
                """ if we reached the threshold, then we to set the bar of the RnD """
                RnD_threshold += RandD
                """ then we get the 'a' which can either be a poisson + normal for innovation, or a simple binomial.
                 Values above or equal (for the imitation) 1 indicate that innovation or imitation occured """
                a = np.random.poisson(1) + np.random.normal(0, 1)

                if a >= 1:
                    """ we are dealing with innovation """
                    RnD_threshold *= rNd_INCREASE * a
                    """ we have to check where did the innovation occur"""
                    what_on = random.choice(['base_CAPEX', 'OPEX', 'MW'])
                    """ if innovation ocurred then we multiply it"""
                    if what_on == 'MW':
                        new_what_on = a * Technology[what_on]
                    else:
                        new_what_on = (1 / a) * Technology[what_on]
                    Technology.update({what_on: new_what_on})

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
                price = weighting_FF(env.now - 1, 'price', 'MWh', MIX, EorM=EorM)
                interest_r = weighting_FF(env.now - 1, 'interest_r', 'MWh', MIX, EorM=EorM, discount=discount)
                self_NPV.update({
                    'value': npv_generating_FF(r, i['lifetime'], 1, i['MW'], i['building_time'], i['CAPEX'], i['OPEX'],
                                               price[i['source']], i['CF'], AMMORT), 'MWh': i['MW']
                })

            """ 8) we must also check if the capping process is on"""

            now = 0
            if len(MIX[env.now - 1]) > 0:
                # if there is no capping, we must first make sure that it has not started
                now = finding_FF(MIX[env.now - 1], 'MW', 'sum', {'EP': name})['value']
                if now > cap_conditions['cap']:
                    capped = True

                else:
                    capped = False

            if now > 0:
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
            #  And finally, the Technology provider decides what to do in   #
            #                        the next period                        #
            #                                                               #
            #################################################################

            """ TPs don't compare sources, so there is no reporting_FF here"""
            if env.now > 0:
                decision_var = private_deciding_FF(name)

                decisions = evaluating_FF(name)

                action = decisions['action']

            #################################################################
            #                                                               #
            #    Before leaving, the agent must uptade the outside world    #
            #                                                               #
            #################################################################

            AGENTS[env.now].update({name: {
                "name": name,
                "genre": genre,
                "subgenre": subgenre,
                "wallet": wallet,
                "profits": profits,
                "capacity": capacity,
                "RandD": RandD,
                "EorM": EorM,
                "innovation_index": innovation_index,
                "Technology": Technology,
                "self_NPV": self_NPV,
                "RnD_threshold": RnD_threshold,
                "capacity_threshold": capacity_threshold,
                "dd_profits": dd_profits,
                "dd_source": dd_source,
                "decision_var": decision_var,
                "action": action,
                "dd_kappas": dd_kappas,
                "dd_qual_vars": dd_qual_vars,
                "dd_backwardness": dd_backwardness,
                "dd_avg_time": dd_avg_time,
                "dd_discount": dd_discount,
                "cap_conditions": cap_conditions,
                "capped": capped,
            }})

            profits_dedicting_FF(name)
            if env.now > 0:
                post_evaluating_FF(decisions['strikes'], name)

            yield env.timeout(1)


class TPM(object):
    def __init__(self, env, wallet, dd_source, decision_var, dd_kappas, dd_qual_vars, dd_backwardness, dd_avg_time,
                 dd_discount, dd_policy, policies, dd_index, dd_eta, dd_ambition, dd_target, dd_rationale, dd_SorT):
        self.env = env
        self.genre = 'TPM'
        self.subgenre = 'TPM'
        self.name = 'TPM'
        self.wallet = wallet
        self.dd_policy = dd_policy
        self.dd_source = dd_source
        self.decision_var = decision_var
        self.disclosed_var = decision_var
        self.action = 'keep'
        self.dd_kappas = dd_kappas
        self.dd_qual_vars = dd_qual_vars
        self.dd_backwardness = dd_backwardness
        self.dd_avg_time = dd_avg_time
        self.dd_discount = dd_discount
        self.policies = policies
        self.dd_index = dd_index
        self.index_per_source = {1: 0, 2: 0, 4: 0, 5: 0}
        self.dd_eta = dd_eta
        self.dd_ambition = dd_ambition
        self.dd_target = dd_target
        self.dd_rationale = dd_rationale
        self.dd_SorT = dd_SorT

        self.action = env.process(self.run_TPM(self.genre,
                                               self.subgenre,
                                               self.name,
                                               self.wallet,
                                               self.dd_policy,
                                               self.dd_source,
                                               self.decision_var,
                                               self.disclosed_var,
                                               self.action,
                                               self.dd_kappas,
                                               self.dd_qual_vars,
                                               self.dd_backwardness,
                                               self.dd_avg_time,
                                               self.dd_discount,
                                               self.dd_policy,
                                               self.policies,
                                               self.dd_index,
                                               self.index_per_source,
                                               self.dd_eta,
                                               self.dd_ambition,
                                               self.dd_target,
                                               self.dd_rationale,
                                               self.dd_SorT))

    def run_TPM(self,
                genre,
                subgenre,
                name,
                wallet,
                dd_policy,
                dd_source,
                decision_var,
                disclosed_var,
                action,
                dd_kappas,
                dd_qual_vars,
                dd_backwardness,
                dd_avg_time,
                dd_discount,
                policies,
                dd_index,
                index_per_source,
                dd_eta,
                dd_ambition,
                dd_target,
                dd_rationale,
                dd_SorT):

        CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, POLICY_EXPIRATION_DATE, AMMORT, TACTIC_DISCOUNT, INSTRUMENT_TO_SOURCE_DICT, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.POLICY_EXPIRATION_DATE, config.AMMORT, config.TACTIC_DISCOUNT, config.INSTRUMENT_TO_SOURCE_DICT, config.env

        while True:

            #################################################################
            #                                                               #
            #              First, the TPM checks if it got an               #
            #            strike, adds or changes its main policy            #
            #                                                               #
            #################################################################

            list_of_strikeables = [dd_policy, dd_source, dd_kappas, dd_qual_vars, dd_backwardness, dd_avg_time,
                                   dd_discount, dd_policy, dd_index, dd_eta, dd_ambition, dd_target, dd_rationale]

            policy = dd_policy['current']
            source = dd_source['current']
            kappa = dd_kappas['current']
            backwardness = dd_backwardness['current']
            avg_time = dd_avg_time['current']
            discount = dd_discount['current']
            index = indexing_FF('TPM') if env.now > 0 else dd_index['current']
            eta_acc = dd_eta['current']
            ambition = dd_ambition['current']
            rationale = dd_rationale['current']
            value = disclosed_var

            if env.now > 0 and (action == 'add' or 'change'):
                striked = striking_FF(list_of_strikeables, kappa)  # with this we have a different list of strikeables

                for entry in range(0, len(list_of_strikeables)):

                    if list_of_strikeables[entry]['current'] == striked[entry]:
                        # that dictionary was not the changed one, so we can just update it
                        list_of_strikeables[entry] = striked[entry]

                    else:
                        # alright, that was the one that changed

                        policies = policymaking_FF(striked[entry], policies,
                                                   disclosed_var) if action == 'change' else policymaking_FF(
                            striked[entry], policies, disclosed_var, add=True)

                action = 'keep'  # we already changed, now back to business

            #################################################################
            #                                                               #
            #             Now, the TPM gives out the incentives             #
            #                                                               #
            #################################################################

            policy_pool = []
            policy_pool.append(policy)
            policy_pool.append(
                policies)  # with this we a temporary list with first the current policy and afterwars all the other policies

            if env.now >= 2:
                for entry in policy_pool:
                    instrument = entry['instrument']
                    source = entry['source']
                    budget = entry['budget'] if 'budget' in entry else value * wallet
                    value = disclosed_var if 'value' not in entry else entry['value']

                    if instrument == 'supply':

                        firms = []
                        for _ in AGENTS[env.now - 1]:
                            i = AGENTS[env.now - 1][_]
                            if i['genre'] == 'TP' and (i['source'] in INSTRUMENT_TO_SOURCE_DICT[source]):
                                firms.append(_)

                        if len(firms) > 0:
                            """ we have to be certain that there are companies to be inbcentivised and now divides the possible incentive by the number of companies """
                            # print('incentivised_firms', incentivised_firms)
                            incentive = budget / len(firms)

                            """ and now we give out the incentives"""
                            for TP in firms:
                                code = uuid.uuid4().int
                                CONTRACTS[env.now].update({
                                    code: {
                                        'sender': name,
                                        'receiver': TP,
                                        'status': 'payment',
                                        'value': incentive}})
                            wallet -= budget
                    else:
                        """
                        demmand-side incentives
                        """
                        print('TBD')

                """ and now back to the actual variables for the current policy"""
                instrument = policy_pool[0]['instrument']
                source = policy_pool[0]['source']
                value = disclosed_var

            #################################################################
            #                                                               #
            #    Now, the TPM analyses what happened to the system due to   #
            #                       its intervention                        #
            #                                                               #
            #################################################################

            add_source = source_reporting_FF(name)

            for entry in dd_source['ranks']:
                dd_source['ranks'][entry] *= (1 - discount)
                dd_source['ranks'][entry] += add_source[entry]

            #################################################################
            #                                                               #
            #         And then, the TPM will decide what to do next         #
            #                                                               #
            #################################################################

            if env.now > 0:
                decision_var = max(0, min(1, public_deciding_FF(name)))
                disclosed_var = thresholding_FF(kappa, disclosed_var, decision_var)

                decisions = evaluating_FF(name)

            #################################################################
            #                                                               #
            #    Before leaving, the agent must uptade the outside world    #
            #                                                               #
            #################################################################

            AGENTS[env.now].update({
                name: {
                    "genre": genre,
                    "subgenre": subgenre,
                    "name": name,
                    "wallet": wallet,
                    "dd_policy": dd_policy,
                    "dd_source": dd_source,
                    "decision_var": decision_var,
                    "disclosed_var": disclosed_var,
                    "action": action,
                    "dd_kappas": dd_kappas,
                    "dd_qual_vars": dd_qual_vars,
                    "dd_backwardness": dd_backwardness,
                    "dd_avg_time": dd_avg_time,
                    "dd_discount": dd_discount,
                    "policies": policies,
                    "dd_index": dd_index,
                    "index_per_source": index_per_source,
                    "dd_eta": dd_eta,
                    "dd_ambition": dd_ambition,
                    "dd_target": dd_target,
                    "dd_rationale": dd_rationale,
                    "policy": policy,
                    "source": source,
                    "kappa": kappa,
                    "backwardness": backwardness,
                    "avg_time": avg_time,
                    "discount": discount,
                    "index": index,
                    "eta_acc": eta_acc,
                    "ambition": ambition,
                    "rationale": rationale,
                    "value": value,
                }})
            if env.now > 0:
                post_evaluating_FF(decisions['strikes'], name)

            yield env.timeout(1)


class EPM(object):
    def __init__(self, env, wallet, PPA_expiration, PPA_limit, COUNTDOWN, dd_policy, dd_source, decision_var, dd_kappas,
                 dd_qual_vars, dd_backwardness, dd_avg_time, dd_discount, policies, dd_index, dd_eta, dd_ambition,
                 dd_target, dd_rationale, auction_capacity, dd_SorT):
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
        self.dd_policy = dd_policy
        self.dd_source = dd_source
        self.decision_var = decision_var
        self.disclosed_var = decision_var
        self.action = 'keep'
        self.dd_kappas = dd_kappas
        self.dd_qual_vars = dd_qual_vars
        self.dd_backwardness = dd_backwardness
        self.dd_avg_time = dd_avg_time
        self.dd_discount = dd_discount
        self.policies = policies
        self.dd_index = dd_index
        self.index_per_source = {1: 0, 2: 0, 4: 0, 5: 0}
        self.dd_eta = dd_eta
        self.dd_ambition = dd_ambition
        self.dd_target = dd_target
        self.dd_rationale = dd_rationale
        self.auction_capacity = auction_capacity
        self.dd_SorT = dd_SorT

        self.action = env.process(self.run_EPM(self.genre,
                                               self.subgenre,
                                               self.name,
                                               self.wallet,
                                               self.PPA_expiration,
                                               self.PPA_limit,
                                               self.auction_countdown,
                                               self.auction_time,
                                               self.COUNTDOWN,
                                               self.dd_policy,
                                               self.dd_source,
                                               self.decision_var,
                                               self.disclosed_var,
                                               self.action,
                                               self.dd_kappas,
                                               self.dd_qual_vars,
                                               self.dd_backwardness,
                                               self.dd_avg_time,
                                               self.dd_discount,
                                               self.dd_policy,
                                               self.dd_index,
                                               self.index_per_source,
                                               self.dd_eta,
                                               self.dd_ambition,
                                               self.dd_target,
                                               self.dd_rationale,
                                               self.auction_capacity,
                                               self.dd_SorT))

    def run_EPM(self,
                genre,
                subgenre,
                name,
                wallet,
                PPA_expiration,
                PPA_limit,
                auction_countdown,
                auction_time,
                COUNTDOWN,
                dd_policy,
                dd_source,
                decision_var,
                disclosed_var,
                action,
                dd_kappas,
                dd_qual_vars,
                dd_backwardness,
                dd_avg_time,
                dd_discount,
                policies,
                dd_index,
                index_per_source,
                dd_eta,
                dd_ambition,
                dd_target,
                dd_rationale,
                auction_capacity,
                dd_SorT):

        CONTRACTS, MIX, AGENTS, TECHNOLOGIC, DEMAND, M_CONTRACT_LIMIT, AUCTION_WANTED_SOURCES, AMMORT, TACTIC_DISCOUNT, INSTRUMENT_TO_SOURCE_DICT, MARGIN, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.DEMAND, config.M_CONTRACT_LIMIT, config.AUCTION_WANTED_SOURCES, config.AMMORT, config.TACTIC_DISCOUNT, config.INSTRUMENT_TO_SOURCE_DICT, config.MARGIN, config.env  # globals

        while True:

            #################################################################
            #                                                               #
            #              First, the TPM checks if it got an               #
            #            strike, adds or changes its main policy            #
            #                                                               #
            #################################################################

            list_of_strikeables = [dd_policy, dd_source, dd_kappas, dd_qual_vars, dd_backwardness, dd_avg_time,
                                   dd_discount, dd_policy, dd_index, dd_eta, dd_ambition, dd_target, dd_rationale]

            policy = dd_policy['current']
            source = dd_source['current']
            kappa = dd_kappas['current']
            backwardness = dd_backwardness['current']
            avg_time = dd_avg_time['current']
            discount = dd_discount['current']
            index = indexing_FF('EPM') if env.now > 0 else dd_index['current']
            eta_acc = dd_eta['current']
            ambition = dd_ambition['current']
            rationale = dd_rationale['current']
            value = disclosed_var

            if env.now > 0 and (action == 'add' or 'change'):
                striked = striking_FF(list_of_strikeables, kappa)  # with this we have a different list of strikeables

                for entry in range(0, len(list_of_strikeables)):

                    if list_of_strikeables[entry]['current'] == striked[entry]:
                        # that dictionary was not the changed one, so we can just update it
                        list_of_strikeables[entry] = striked[entry]

                    else:
                        # alright, that was the one that changed

                        policies = policymaking_FF(striked[entry], policies,
                                                   disclosed_var) if action == 'change' else policymaking_FF(
                            striked[entry], policies, disclosed_var, add=True)

                action = 'keep'  # we already changed, now back to business

            #################################################################
            #                                                               #
            #       Before doing policy, the policy maker must decide       #
            #                       which plants enter                      #
            #                                                               #
            #################################################################

            """ since the policy makers act after private agents, they are looking at the env.now, not the env.now-1 """
            if env.now > 0 and len(MIX[env.now]) > 0:
                """ 
                First, we contract and precify the electricity projects
                """

                possible_projects = []
                for _ in MIX[env.now]:
                    """ we build the list of possible projects, i.e., projects deployed or already built"""
                    j = MIX[env.now][_]
                    if ((j['status'] == 'built' or j['status'] == 'contracted') and j['EorM'] == 'E'):
                        possible_projects.append(j)
                """ now, we sort the list of dictionaries in terms of dispatchability and then OPEX (respecting the merit order)"""
                possible_projects = sorted(possible_projects, key=lambda x: (x['dispatchable'], x['OPEX']))
                chosen = []
                demand = DEMAND.copy()[env.now]['E']
                for plant in possible_projects:
                    if demand < 0:
                        """ if there is no more demand to be supplied, then the plant is not contracted"""
                        MIX[env.now][plant['code']].update({
                            'status': 'built'
                        })
                    else:
                        """ if there is still demand to to be supplied, then, the power plant is contracted"""
                        MIX[env.now][plant['code']].update({
                            'status': 'contracted'
                        })
                        chosen.append(plant)
                        demand -= plant['MWh']
                """ following the merit order, the system is precified in relation to its most costly unit"""
                chosen = sorted(chosen, key=lambda x: x['OPEX'])[-1]
                price = (chosen['OPEX'] + (chosen['CAPEX'] / chosen['lifetime'])) * (1 + MARGIN) / chosen['MWh']
                for i in possible_projects:
                    """ if the plant is auction contracted, then we do not mess with its price"""
                    if i['auction_contracted'] != True:
                        MIX[env.now][i['code']].update({
                            'price': price})
                """ 
                now, we contract molecule projects
                """

                possible_projects = []
                demand = DEMAND.copy()[env.now]['M']
                """ since molecule projects are take of pay, if they are not contracted, then demand make contract them"""
                for _ in MIX[env.now]:
                    j = MIX[env.now][_]
                    if j['status'] != 'contracted' and j['EorM'] == 'M' and j['M_limit'] != env.now:
                        possible_projects.append(j)
                    elif j['status'] == 'contracted' and j['EorM'] == 'M' and j['M_limit'] == env.now:
                        j.update({'status': 'uncontracted'})
                        demand -= j['MWh']

                """ this time, we sort by price, as the prices are defined by the agents """
                possible_projects = sorted(possible_projects, key=lambda x: (x['price']))
                for plant in possible_projects:
                    if demand < 0:
                        MIX[env.now][plant['code']].update({
                            'status': 'uncontracted'
                        })
                    else:
                        MIX[env.now][plant['code']].update({
                            'status': 'contracted',
                            'M_limit': env.now + 1 + M_CONTRACT_LIMIT
                        })
                    demand -= plant['MWh']

            #################################################################
            #                                                               #
            #           First, the EPM adds or changes its policy           #
            #                                                               #
            ################################################################# 

            policy_pool = []
            policy_pool.append(policy)
            policy_pool.append(
                policies)  # with this we a temporary list with first the current policy and afterwars all the other policies

            if env.now >= 2:
                for entry in policy_pool:
                    instrument = entry['instrument']
                    source = entry['source']
                    budget = entry['budget'] if 'budget' in entry else value * wallet
                    value = disclosed_var if 'value' not in entry else entry['value']

                    if instrument == 'carbon_tax':
                        """
                        CARBON-TAX
                        """

                        for _ in TECHNOLOGIC[0]:
                            i = TECHNOLOGIC[0][_]
                            j = TECHNOLOGIC[env.now + 1][_]
                            """ for the carbon-tax, if the source is thermal (0) or natural gas (3), then, its OPEX gets higher"""
                            if i['source'] in [0, 3]:
                                j.update({
                                    'OPEX': i['OPEX'] * (1 + value)
                                })
                                """ we can update in relation to the initial value, because there is no innovation for non-renewables"""

                    elif instrument == 'FiT':
                        """
                        Feed-in Tariff
                        """

                        """ we are actually using Feed-in premium, since we are adding a payment to the market price"""
                        for _ in MIX[env.now - 1]:
                            i = MIX[env.now - 1][_]
                            if i['source'] in INSTRUMENT_TO_SOURCE_DICT(source) and i['auction_contracted'] != True:
                                i.update({
                                    'price': i['price'] * (1 + value)
                                })

                    else:
                        """
                        Auctions: these are a little bit more complicated
                        """

                        """ we do things with auction_countdown, which is a countdown for the auction period, which allows the EPM to collect projhects, and with auction_time, a boolean variable which signals if there is an auction underway"""
                        if auction_countdown > 0 and auction_time == True:
                            """ if the coundtdown has yet to reach zero but an Auction is underway, then we cut down one period from the auction_countdown """
                            auction_countdown -= 1
                        elif auction_time == False:
                            """ if we are doing an auction, but the auction_time is not signalling that an auction is underway, then we have to set it to true and trigger the countdown to the auction itself"""
                            auction_countdown = COUNTDOWN
                            auction_time = True
                            for j in range(env.now, env.now + COUNTDOWN):
                                """ as we starting the auction, we need to signal energy producers that their next projects of the target-source will be put inside the auction, for that we set up bids for each energy producer for each period from now until the deadline. This process creates bids for new agents also"""
                                for source in INSTRUMENT_TO_SOURCE_DICT[source]:
                                    AUCTION_WANTED_SOURCES.append(INSTRUMENT_TO_SOURCE_DICT[source])
                        elif auction_countdown == 0 and auction_time == True:
                            """ under these circumnstances, it is time to do the auction and contract projects for the ppas"""
                            auction_time = False
                            possible_projects = []
                            contracted_projects = []
                            for source in INSTRUMENT_TO_SOURCE_DICT[source]:
                                AUCTION_WANTED_SOURCES.remove(INSTRUMENT_TO_SOURCE_DICT[source])
                            """first we get the projects that were bidded"""
                            for i in range(env.now - COUNTDOWN, env.now - 1):
                                for _ in CONTRACTS[i]:
                                    __ = CONTRACTS[i][_]
                                    j = __.copy()
                                    if j['bidded'] == True:
                                        """ here we are adding the whole contract"""
                                        possible_projects.append(j)
                            """ we have to sort in terms of 'OPEX' """
                            possible_projects = sorted(possible_projects, key=lambda x: x['OPEX'])
                            """ the capacity for auction is the value (in GW) multplied by 1000 (to become MW)"""
                            remaining_capacity = auction_capacity * value * 1000
                            for i in possible_projects:
                                if remaining_capacity > 0:
                                    """ if there is still capacity to contract, then the project is contracted"""
                                    contracted_projects.append(i['code'])
                                remaining_capacity -= i['capacity']
                            """ then we cycle throught the codes of the contracted projects in order to tell the proponent agents that their projects were contracted"""
                            for code in contracted_projects:
                                """ first we update it to include attributes exclusive to PPAs, such as the auction price, the boolean auction_contracted and the date of expiration of the price"""
                                CONTRACTS[env.now].update({code: {
                                    'status': 'project',
                                    'price': (possible_projects[i]['OPEX'] + (
                                                possible_projects[i]['CAPEX'] / possible_projects[i]['lifetime'])) * (
                                                         1 + MARGIN) / possible_projects[i]['MWh'],
                                    'auction_contracted': True,
                                    'price_expiration': env.now + PPA_expiration + PPA_limit + 1,
                                    'limit': env.now + PPA_limit + 1
                                }})

                            """ lastly, we have to get rid of the contracted projects. This is relevant specially when  there are multiple auctions happening"""
                            for i in range(env.now - 1 - COUNTDOWN, env.now - 1):
                                for _ in CONTRACTS[i]:
                                    j = CONTRACTS[i][_]
                                    if j['code'] in contracted_projects:
                                        j.update('bidded' == False)

                """ and now back to the actual variables for the current policy"""

                instrument = policies[0]['instrument']
                source = policies[0]['source']
                value = disclosed_var

            #################################################################
            #                                                               #
            #    Now, the EPM analyses what happened to the system due to   #
            #                       its intervention                        #
            #                                                               #
            #################################################################

            add_source = source_reporting_FF(name)

            for entry in dd_source['ranks']:
                dd_source['ranks'][entry] *= (1 - discount)
                dd_source['ranks'][entry] += add_source[entry]

            #################################################################
            #                                                               #
            #         And then, the EPM will decide what to do next         #
            #                                                               #
            #################################################################

            if env.now > 0:
                decision_var = max(0, min(1, public_deciding_FF(name)))
                disclosed_var = thresholding_FF(kappa, disclosed_var, decision_var)
                decisions = evaluating_FF(name)

                action = decisions['action']

            #################################################################
            #                                                               #
            #    Before leaving, the agent must uptade the outside world    #
            #                                                               #
            #################################################################

            AGENTS[env.now].update({name: {
                "genre": genre,
                "subgenre": subgenre,
                "name": name,
                "wallet": wallet,
                "PPA_expiration": PPA_expiration,
                "PPA_limit": PPA_limit,
                "auction_countdown": auction_countdown,
                "auction_time": auction_time,
                "COUNTDOWN": COUNTDOWN,
                "dd_policy": dd_policy,
                "dd_source": dd_source,
                "decision_var": decision_var,
                "disclosed_var": disclosed_var,
                "action": action,
                "dd_kappas": dd_kappas,
                "dd_qual_vars": dd_qual_vars,
                "dd_backwardness": dd_backwardness,
                "dd_avg_time": dd_avg_time,
                "dd_discount": dd_discount,
                "policies": policies,
                "dd_index": dd_index,
                "index_per_source": index_per_source,
                "dd_eta": dd_eta,
                "dd_ambition": dd_ambition,
                "dd_target": dd_target,
                "dd_rationale": dd_rationale,
                "auction_capacity": auction_capacity,
                "instrument": instrument,
                "source": source,
            }})
            if env.now > 0:
                post_evaluating_FF(decisions['strikes'], name)

            yield env.timeout(1)


class DBB(object):
    def __init__(self, env, wallet, dd_policy, dd_source, decision_var, dd_kappas, dd_qual_vars, dd_backwardness,
                 dd_avg_time, dd_discount, policies, dd_index, dd_eta, dd_ambition, dd_target, dd_rationale, Portfolio,
                 accepted_sources, dd_SorT):
        self.env = env
        self.NPV_THRESHOLD_DBB = config.NPV_THRESHOLD_DBB
        self.guaranteed_contracts = []
        self.genre = 'DBB'
        self.subgenre = 'DBB'
        self.name = 'DBB'
        self.wallet = wallet
        self.dd_policy = dd_policy
        self.dd_source = dd_source
        self.decision_var = decision_var
        self.disclosed_var = decision_var
        self.action = 'keep'
        self.dd_kappas = dd_kappas
        self.dd_qual_vars = dd_qual_vars
        self.dd_backwardness = dd_backwardness
        self.dd_avg_time = dd_avg_time
        self.dd_discount = dd_discount
        self.dd_policy = dd_policy
        self.policies = policies
        self.dd_index = dd_index
        self.index_per_source = {1: 0, 2: 0, 4: 0, 5: 0}
        self.dd_eta = dd_eta
        self.dd_ambition = dd_ambition
        self.dd_target = dd_target
        self.dd_rationale = dd_rationale
        self.financing_index = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.Portfolio = Portfolio
        self.receivable = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.accepted_sources = accepted_sources
        self.car_ratio = 0
        self.dd_SorT = dd_SorT

        self.action = env.process(self.run_DBB(self.NPV_THRESHOLD_DBB,
                                               self.guaranteed_contracts,
                                               self.genre,
                                               self.subgenre,
                                               self.name,
                                               self.wallet,
                                               self.dd_policy,
                                               self.dd_source,
                                               self.decision_var,
                                               self.disclosed_var,
                                               self.action,
                                               self.dd_kappas,
                                               self.dd_qual_vars,
                                               self.dd_backwardness,
                                               self.dd_avg_time,
                                               self.dd_discount,
                                               self.policies,
                                               self.dd_index,
                                               self.index_per_source,
                                               self.dd_eta,
                                               self.dd_ambition,
                                               self.dd_target,
                                               self.dd_rationale,
                                               self.financing_index,
                                               self.Portfolio,
                                               self.receivable,
                                               self.accepted_sources,
                                               self.car_ratio,
                                               self.dd_SorT))

    def run_DBB(self,
                NPV_THRESHOLD_DBB,
                guaranteed_contracts,
                genre,
                subgenre,
                name,
                wallet,
                dd_policy,
                dd_source,
                decision_var,
                disclosed_var,
                action,
                dd_kappas,
                dd_qual_vars,
                dd_backwardness,
                dd_avg_time,
                dd_discount,
                policies,
                dd_index,
                index_per_source,
                dd_eta,
                dd_ambition,
                dd_target,
                dd_rationale,
                financing_index,
                Portfolio,
                receivable,
                accepted_sources,
                car_ratio,
                dd_SorT):

        CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, POLICY_EXPIRATION_DATE, INSTRUMENT_TO_SOURCE_DICT, RISKS, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.POLICY_EXPIRATION_DATE, config.INSTRUMENT_TO_SOURCE_DICT, config.RISKS, config.env  # globals

        while True:

            #################################################################
            #                                                               #
            #              First, the DBB checks if it got an               #
            #            strike, adds or changes its main policy            #
            #                                                               #
            #################################################################

            list_of_strikeables = [dd_policy, dd_source, dd_kappas, dd_qual_vars, dd_backwardness, dd_avg_time,
                                   dd_discount, dd_policy, dd_index, dd_eta, dd_ambition, dd_target, dd_rationale]

            policy = dd_policy['current']
            source = dd_source['current']
            kappa = dd_kappas['current']
            backwardness = dd_backwardness['current']
            avg_time = dd_avg_time['current']
            discount = dd_discount['current']
            index = indexing_FF('EPM') if env.now > 0 else dd_index['current']
            eta_acc = dd_eta['current']
            ambition = dd_ambition['current']
            rationale = dd_rationale['current']
            value = disclosed_var

            if env.now > 0 and (action == 'add' or 'change'):
                striked = striking_FF(list_of_strikeables, kappa)  # with this we have a different list of strikeables

                for entry in range(0, len(list_of_strikeables)):

                    if list_of_strikeables[entry]['current'] == striked[entry]:
                        # that dictionary was not the changed one, so we can just update it
                        if entry == 'source':
                            # we changed the source, so we have to update the accepted_sources dictionary
                            source_accepting_FF(accepted_sources, source)
                            source = dd_source['current']
                        list_of_strikeables[entry] = striked[entry]

                    else:
                        # alright, that was the one that changed

                        policies = policymaking_FF(striked[entry], policies,
                                                   disclosed_var) if action == 'change' else policymaking_FF(
                            striked[entry], policies, disclosed_var, add=True)

                action = 'keep'  # we already changed, now back to business

            #################################################################
            #                                                               #
            #    Before doing anything, the development bank must collect   #
            #      its  payments and guarantee contracts that need be       #
            #                                                               #
            #################################################################

            # first, we must update the risk dictionary

            for source in [0, 1, 2, 3, 4, 5]:
                try:
                    RISKS.update(
                        {source: finding_FF(MIX[env.now], 'risk', 'median', {'source': source})['value']})
                    # we attempt to get the median risk for each source
                except:
                    # if not possible (there are no banks financing that source or no banks at all) which translates to risky source
                    RISKS.update({source: 1})

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
                        wallet += i['value']

                        if i['status'] == 'default':
                            """ the company went into deafult and the guarantee is activated"""
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

            policy_pool = []
            policy_pool.append(policy)
            policy_pool.append(
                policies)  # with this we a temporary list with first the current policy and afterwars all the other policies

            if env.now >= 2:
                for entry in policy_pool:
                    instrument = entry['instrument']
                    source = entry['source']
                    budget = entry['budget'] if 'budget' in entry else value * wallet
                    value = disclosed_var if 'value' not in entry else entry['value']

                    if instrument == 'finance' and len(CONTRACTS[env.now - 1]) > 0:
                        financing = financing_FF(genre, source, name, wallet, receivable, value, financing_index)

                        wallet = financing['wallet']
                        receivable = financing['receivables']

                    else:
                        """ We are dealing with guarentees """
                        financing = financing_FF(genre, source, name, wallet, receivable, value, financing_index, True)

                        wallet = financing['wallet']
                        receivable = financing['receivables']
                        financing_index = financing['financing_index']

                    """ and now back to the actual variables for the current policy"""
                instrument = policies[0]['instrument']
                source = policies[0]['source']
                value = disclosed_var

            #################################################################
            #                                                               #
            #    Now, the EPM analyses what happened to the system due to   #
            #                       its intervention                        #
            #                                                               #
            #################################################################

            add_source = source_reporting_FF(name)

            for entry in dd_source['ranks']:
                dd_source['ranks'][entry] *= (1 - discount)
                dd_source['ranks'][entry] += add_source[entry]

            #################################################################
            #                                                               #
            #         And then, the EPM will decide what to do next         #
            #                                                               #
            #################################################################

            if env.now > 0:
                decision_var = max(0, min(1, public_deciding_FF(name)))
                disclosed_var = thresholding_FF(kappa, disclosed_var, decision_var)

            #################################################################
            #                                                               #
            #    Before leaving, the agent must uptade the outside world    #
            #                                                               #
            #################################################################

            AGENTS[env.now].update({
                name: {
                    "NPV_THRESHOLD_DBB": NPV_THRESHOLD_DBB,
                    "guaranteed_contracts": guaranteed_contracts,
                    "genre": genre,
                    "subgenre": subgenre,
                    "name": name,
                    "wallet": wallet,
                    "dd_policy": dd_policy,
                    "dd_source": dd_source,
                    "decision_var": decision_var,
                    "disclosed_var": disclosed_var,
                    "action": action,
                    "dd_kappas": dd_kappas,
                    "dd_qual_vars": dd_qual_vars,
                    "dd_backwardness": dd_backwardness,
                    "dd_avg_time": dd_avg_time,
                    "dd_discount": dd_discount,
                    "policies": policies,
                    "dd_index": dd_index,
                    "index_per_source": index_per_source,
                    "dd_eta": dd_eta,
                    "dd_ambition": dd_ambition,
                    "dd_target": dd_target,
                    "dd_rationale": dd_rationale,
                    "financing_index": financing_index,
                    "Portfolio": Portfolio,
                    "receivable": receivable,
                    "accepted_sources": accepted_sources,
                    "car_ratio": car_ratio,
                    "policy": policy,
                    "source": source,
                    "kappa": kappa,
                    "backwardness": backwardness,
                    "avg_time": avg_time,
                    "discount": discount,
                    "index": index,
                    "eta_acc": eta_acc,
                    "ambition": ambition,
                    "rationale": rationale,
                    "value": value
                }})

            yield env.timeout(1)


class BB(object):
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
        self.action = "keep"  # this is the action variable. It can be either 'keep', 'change' or 'add' 
        self.dd_kappas = dd_kappas  # this is the kappa, follows the current ranked dictionary
        self.dd_qual_vars = dd_qual_vars  # this tells the agent the qualitative variables in a form {0 : 'name of the zeroth variable', 1 : 'name of the first variable', 2 : 'name of the second variable'}
        self.dd_backwardness = dd_backwardness  # also a ranked dictionary, this one tells the backwardness of agents
        self.dd_avg_time = dd_avg_time  # also a ranked dictionary, this one tells the average time for deciding if change is necessary
        self.dd_discount = dd_discount  # discount factor. Is a ranked dictionary
        self.dd_strategies = dd_strategies  # initial strategy for the technology provider. Is a ranked dictionary
        self.dd_index = dd_index

        self.action = env.process(self.run_BB(self.financing_index,
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
                                              self.action,
                                              self.dd_kappas,
                                              self.dd_qual_vars,
                                              self.dd_backwardness,
                                              self.dd_avg_time,
                                              self.dd_discount,
                                              self.dd_strategies,
                                              self.dd_index
                                              ))

    def run_BB(self,
               financing_index,
               Portfolio,
               receivable,
               accepted_sources,
               car_ratio,
               name,
               genre,
               subgenre,
               wallet,
               profits,
               dd_profits,
               dd_source,
               decision_var,
               action,
               dd_kappas,
               dd_qual_vars,
               dd_backwardness,
               dd_avg_time,
               dd_discount,
               dd_strategies,
               dd_index):

        CONTRACTS, MIX, AGENTS, AGENTS_r, TECHNOLOGIC, r, BASEL, AMMORT, TACTIC_DISCOUNT, NPV_THRESHOLD, RISKS, env = config.CONTRACTS, config.MIX, config.AGENTS, config.AGENTS_r, config.TECHNOLOGIC, config.r, config.BASEL, config.AMMORT, config.TACTIC_DISCOUNT, config.NPV_THRESHOLD, config.RISKS, config.env  # globals

        while True:

            #################################################################
            #                                                               #
            #     Before anything, we must the current values of each of    #
            #        the dictionaries that we use and other variables       #
            #                                                               #
            #################################################################

            list_of_strikeables = [dd_kappas, dd_qual_vars, dd_backwardness, dd_avg_time, dd_discount, dd_strategies]

            source = dd_source['current']
            kappa = dd_kappas['current']
            qual_vars = dd_qual_vars['current']
            backwardness = dd_backwardness['current']
            avg_time = dd_avg_time['current']
            discount = dd_discount['current']
            strategy = dd_strategies['current']
            discount = dd_discount['current']
            index = indexing_FF('TPM') if env.now > 0 else dd_index['current']
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
            #    Now, on to check if change is on and if there is a strike  #
            #                                                               #
            #################################################################

            if env.now > 0 and (action == 'add' or 'change'):
                striked = striking_FF(list_of_strikeables, kappa)

                for entry in range(0, len(list_of_strikeables)):
                    list_of_strikeables[entry] = striked[entry]
                    if entry == 'source':
                        # we changed the source, so we have to update the accepted_sources dictionary
                        source_accepting_FF(accepted_sources, source)
                action = 'keep'  # we already changed, now back to business

            #################################################################
            #                                                               #
            #        Then, the bank decides which projects to accept        #
            #                                                               #
            #################################################################

            if env.now > 0 and len(CONTRACTS[env.now - 1]) > 0:
                financing = financing_FF(genre, accepted_sources, name, wallet, receivable, value, financing_index)

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
                "action": action,
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
            }})

            profits_dedicting_FF(name)
            if env.now > 0:
                post_evaluating_FF(decisions['strikes'], name)

            yield env.timeout(1)


class EP(object):
    def __init__(self, env, accepted_sources, name, wallet, EorM, portfolio_of_plants, portfolio_of_projects,
                 periodicity, tolerance, last_acquisition_period, dd_source, decision_var, dd_kappas, dd_qual_vars,
                 dd_backwardness, dd_avg_time, dd_discount, dd_strategies, dd_index):
        self.env = env
        self.genre = 'EP'
        self.accepted_sources = accepted_sources
        self.name = name
        self.wallet = wallet
        self.profits = 0
        self.EorM = EorM
        self.subgenre = EorM
        self.capacity = {0: 0, 1: 0, 2: 0} if self.EorM == 'E' else {3: 0, 4: 0, 5: 0}
        self.portfolio_of_plants = portfolio_of_plants
        self.portfolio_of_projects = portfolio_of_projects
        self.periodicity = periodicity
        self.subgenre_price = {0: 0, 1: 0, 2: 0} if self.EorM == 'E' else {3: 0, 4: 0, 5: 0}
        self.tolerance = tolerance
        self.last_acquisition_period = last_acquisition_period
        self.dd_profits = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0,
                           5: 0}  # same as profits, but as dict. Makes accounting faster and simpler
        self.dd_source = dd_source  # This, my ganzirosis, used to be the Tactics. It is the first of the ranked dictionaries. It goes a little sumthing like dis: dd = {'current' : 2, 'ranks' : {0: 3500, 1: 720, 2: 8000}}. With that we have the current decision for the variable or thing and on the ranks we have the score for
        self.decision_var = decision_var  # this is the value of the decision variable. Is a number between -1 and 1
        self.action = "keep"  # this is the action variable. It can be either 'keep', 'change' or 'add'
        self.dd_kappas = dd_kappas  # this is the kappa, follows the current ranked dictionary
        self.dd_qual_vars = dd_qual_vars  # this tells the agent the qualitative variables in a form {0 : 'name of the zeroth variable', 1 : 'name of the first variable', 2 : 'name of the second variable'}
        self.dd_backwardness = dd_backwardness  # also a ranked dictionary, this one tells the backwardness of agents
        self.dd_avg_time = dd_avg_time  # also a ranked dictionary, this one tells the average time for deciding if change is necessary
        self.dd_discount = dd_discount  # discount factor. Is a ranked dictionary
        self.dd_strategies = dd_strategies  # initial strategy for the technology provider. Is a ranked dictionary
        self.dd_index = dd_index

        self.action = env.process(self.run_EP(self.genre,
                                              self.accepted_sources,
                                              self.name,
                                              self.wallet,
                                              self.profits,
                                              self.EorM,
                                              self.subgenre,
                                              self.capacity,
                                              self.portfolio_of_plants,
                                              self.portfolio_of_projects,
                                              self.periodicity,
                                              self.subgenre_price,
                                              self.tolerance,
                                              self.last_acquisition_period,
                                              self.dd_profits,
                                              self.dd_source,
                                              self.action,
                                              self.dd_kappas,
                                              self.dd_qual_vars,
                                              self.dd_backwardness,
                                              self.dd_avg_time,
                                              self.dd_discount,
                                              self.dd_strategies,
                                              self.dd_index))

    def run_EP(self,
               genre,
               accepted_sources,
               name,
               wallet,
               profits,
               EorM,
               subgenre,
               capacity,
               portfolio_of_plants,
               portfolio_of_projects,
               periodicity,
               subgenre_price,
               tolerance,
               last_acquisition_period,
               dd_profits,
               dd_source,
               action,
               dd_kappas,
               dd_qual_vars,
               dd_backwardness,
               dd_avg_time,
               dd_discount,
               dd_strategies,
               dd_index):

        CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, DEMAND, AMMORT, AUCTION_WANTED_SOURCES, BB_NAME_LIST, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.DEMAND, config.AMMORT, config.AUCTION_WANTED_SOURCES, config.BB_NAME_LIST, config.env

        while True:

            #################################################################
            #                                                               #
            #     Before anything, we must the current values of each of    #
            #        the dictionaries that we use and other variables       #
            #                                                               #
            #################################################################

            list_of_strikeables = [dd_profits, dd_source, dd_kappas, dd_qual_vars, dd_backwardness, dd_avg_time,
                                   dd_discount, dd_strategies]

            source = dd_source['current']
            kappa = dd_kappas['current']
            qual_vars = dd_qual_vars['current']
            backwardness = dd_backwardness['current']
            avg_time = dd_avg_time['current']
            discount = dd_discount['current']
            strategy = dd_strategies['current']
            discount = dd_discount['current']
            index = indexing_FF('TPM') if env.now > 0 else dd_index['current']
            value = decision_var
            profits = 0  # in order to get the profits of this period alone

            #################################################################
            #                                                               #
            #    Now, on to check if change is on and if there is a strike  #
            #                                                               #
            #################################################################

            if env.now > 0 and (action == 'add' or 'change'):
                striked = striking_FF(list_of_strikeables, kappa)

                for entry in range(0, len(list_of_strikeables)):
                    if entry == 'source':
                        # we changed the source, so we have to update the accepted_sources dictionary
                        source_accepting_FF(accepted_sources, source)
                    list_of_strikeables[entry] = striked[entry]
                action = 'keep'  # we already changed, now back to business

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
                    i = MIX[env.now - 1][_]
                    """ the _ is the code of the plant, whereas i is the dictionary of the plant itself """
                    if i['EP'] == name and i['status'] == 'contracted':
                        """ if the plant is mine and it is contracted, I'll collect profits """
                        wallet += i['MWh'] * i['price'] - i['OPEX']
                        profits += i['MWh'] * i['price'] - i['OPEX']

                        """ we also have to put the profit as a contract in the CONTRACTS dictionary in order for the policy makers, other EPs and the demmand to do some calculations """
                        code = uuid.uuid4().int
                        """ this is to get a unique and random number """
                        j = i.copy()
                        """ and we also have to create a copy of the i dictionary, if not things will update on that dictionary, and that's no good """
                        j.update({
                            'status': 'payment',
                            'sender': 'D',
                            'source': j['source'],
                            'receiver': name,
                            'value': j['MWh'] * j['price']
                        })
                        """ and now we update"""
                        CONTRACTS[env.now - 1].update({code: j})

            #############################################################
            #                                                           #
            #  Now the EP goes through its portfolio_of_plants in order #
            #   to 1) pay the banks for the financing, 2) check if      #
            #     plants finished building, 3) retire old plants and    #
            #               4) insert plants into the mix               #
            #                                                           #
            #############################################################

            """ before anything, we have to make sure that the MW_dict of this period has the same MW as the last period """

            if len(portfolio_of_plants) > 0:
                for _ in portfolio_of_plants:
                    i = portfolio_of_plants[_]
                    number = ((i['principal'] / (1 + AMMORT)) + i['principal'] * r)

                    """  1) we pay for the ammortization of plants """

                    if i['ammortisation'] > env.now and (
                            i['guarantee'] != True or (i['guarantee'] == True and wallet >= number)):
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
                    elif i['ammortisation'] > env.now and i['guarantee'] == True and wallet < number:
                        """ if not, then the development bank pays for that monthly fee"""
                        j = i.copy()
                        j.update({
                            'receiver': 'DBB',
                            'status': 'default'
                        })
                        code = uuid.uuid4().int
                        CONTRACTS[env.now].update({code: j})

                    """ 2) now we retire old pants """

                    if i['retirement'] <= env.now:
                        i.update({'status': 'retired'})

                    """ 3) and we check if plants finished building """

                    if i['completion'] <= env.now:
                        i.update({'status': 'built'})

                    """ 4) now we insert built plants into the mix  """

                    if i['status'] == 'built':
                        j = i.copy()
                        MIX[env.now].update({_: j})

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
                    if i['receiver'] == name and i['status'] == 'financed':
                        """ the project was financed """
                        last_acquisition_period = env.now
                        j = i.copy()
                        j.update({
                            'EP': name,
                            'BB': j['sender'],
                            'code': _,
                            'status': 'building',
                            'principal': j['CAPEX'] * (1 + r) ** j['building_time'],
                            'completion': j['building_time'] + 1 + env.now,
                            'retirement': j['lifetime'] + j['building_time'] + 1 + env.now
                        })

                        """ moreover, if it is a molecule project,
                        the price is pre-fixed """
                        if i['EorM'] == 'M':
                            j.update({
                                'price': (i['OPEX'] + (i['CAPEX'] / i['lifetime'])) / i['MWh'] * (1 + value)
                            })
                        # j.pop('receiver')
                        # j.pop('sender')

                        portfolio_of_plants.update({_: j})
                        # print('portfolio_of_plants.update({_ : j})', _, j)
                        capacity.update({i['source']: capacity[i['source']] + i['capacity']})

                        """ if the financed project was in the pool of projects of the EP, we have to take it out """
                        if _ in portfolio_of_projects:
                            portfolio_of_projects.pop(_)

                            """ now we ready the contract that tells the TP that he got a new project"""
                        code = uuid.uuid4().int
                        CONTRACTS[env.now].update({
                            code: {'sender': name,
                                   'receiver': j['TP'],
                                   'status': 'payment',
                                   'source': j['source'],
                                   'value': j['CAPEX'],
                                   'MWh': i['MWh']
                                   }
                        })
                    elif (i['guarantee'] == True or i['auction_contracted'] == True) and i['receiver'] == name and i[
                        'status'] == 'project' and 'failed_attempts' not in i:
                        """ if the project was not financed but it got a guarantee or whas a PPA, we have to prepare it to be be inserrted into the portfolio_of_projects dictionary """
                        j = i.copy()
                        j.update({'code': _})
                        if 'limit' not in j:
                            """ if the key 'limit' is not in j, then we insert it, as well as the list failed_attempts, in which we put the name of the banks that rejected the project"""
                            j.update({'limit': env.now + tolerance})
                            j.update({'failed_attempts': [i['receiver']]})
                        portfolio_of_projects.update({_: j})
                    elif _ in portfolio_of_projects and i['sender'] == name and i['status'] == 'rejected' and j[
                        'limit'] > env.now and j['CAPEX'] <= wallet:
                        """ the rejected project was in the portfolio_of_projects dicionary. So now we have to decide it will be prepared to be resent to another bank or if the EP will finance it through reinvestment"""
                        j = i.copy()
                        """ if no one accepts to finance and the EP has enough money, it can pay for the plant itself"""
                        wallet -= j['CAPEX']
                        portfolio_of_projects.pop(j['code'])
                        code = uuid.uuid4().int
                        last_acquisition_period = env.now
                        k = i.copy()
                        k.update({
                            'EP': name,
                            'BB': 'reinvestment',
                            'code': code,
                            'status': 'building',
                            'principal': 0,
                            'completion': k['building_time'] + 1 + env.now,
                            'retirement': k['lifetime'] + k['building_time'] + 1 + env.now
                        })
                        """ moreover, if it is a molecule project, the price is pre-fixed """
                        if i['EorM'] == 'M':
                            k.update({'price': (i['OPEX'] + (i['CAPEX'] / i['lifetime'])) * decision_var})
                        portfolio_of_plants.update({code: k})
                        capacity.update({k['source']: capacity[k['source']] + k['capacity']})

                        code = uuid.uuid4().int
                        CONTRACTS[env.now].update({
                            code:
                                {'sender': name,
                                 'receiver': k['TP'],
                                 'status': 'payment',
                                 'source': k['source'],
                                 'value': k['CAPEX'],
                                 'MWh': k['MWh']
                                 }
                        })
                    elif _ in portfolio_of_projects and i['sender'] == name and i[
                        'status'] == 'rejected' and j['limit'] > env.now and j['CAPEX'] > wallet:
                        """ if not, then we take that project from the pool"""
                        portfolio_of_projects.pop(j)
                    elif _ in portfolio_of_projects and i['sender'] == name and i['status'] == 'rejected' and j[
                        'limit'] < env.now:
                        portfolio_of_projects[i['code']].update({
                            'failed_attempts': j['failed_attempts'].append(
                                j['receiver']
                            )
                        })

            #############################################################
            #                                                           #
            #   Then, the Energy producer decides how much to invest    #
            #                                                           #
            #############################################################

            if env.now % periodicity == 0 and env.now > 1 + periodicity and decision_var > 0 and wallet > 0:
                TP = {'TP': 0,
                      'NPV': False,
                      'Lumps': 0,
                      'CAPEX': 0,
                      'OPEX': 0
                      }
                for _ in TECHNOLOGIC[env.now - 1]:
                    i = TECHNOLOGIC[env.now - 1][_]
                    if accepted_sources[i['source']] == True:
                        source_price = weighting_FF(env.now - 1, 'price', 'MWh', MIX)
                        Lumps = np.ceil((decision_var * DEMAND[EorM]) / i['MW'])
                        price = source_price[i['source']]
                        NPV = npv_generating_FF(r, i['lifetime'], Lumps, Lumps * i['MW'], i['building_time'],
                                                i['CAPEX'], i['OPEX'], price, i['CF'], AMMORT)
                        if NPV > TP['NPV'] or TP['NPV'] == False:
                            TP.update({
                                'TP': _,
                                'NPV': NPV,
                                'Lumps': Lumps,
                                'CAPEX': i['CAPEX'] * Lumps,
                                'OPEX': i['OPEX'] * Lumps,
                                'source_of_TP': i['source']
                            })

                if source in AUCTION_WANTED_SOURCES:
                    receiver = 'EPM'
                else:
                    """ if the source is not currently in an auction, the EP sends it directly to a bank"""
                    """ now we select which bank to try """
                    BB_ = []
                    for _ in AGENTS[env.now - 1]:
                        """ we must check at this period, because the EP goes after the bank """
                        agent = AGENTS[env.now - 1][_]
                        if agent['genre'] == 'BB' or agent['genre'] == 'DBB':
                            BB_.append([agent['name'], agent['financing_index'][source]])
                    number = np.random.poisson(1)
                    print(BB_)
                    number = number if number < len(BB_) else len(BB_) - 1
                    BB = sorted(BB_, key=lambda x: x[1], reverse=True)[number][0]
                    # first we sort the list of agents. With the .values() we have both keys and values. The [number] selects the random pick of which source to get. The ['name'] selects the name of the bank
                    receiver = BB

                # OPEX and CAPEX are in relation to one lump, so in the project we have to change them to account for the whole project
                project = TECHNOLOGIC[env.now - 1][TP['TP']].copy()
                # we have to use .copy() here to avoid changing the TECHNOLOGIC dictionary entry
                project.update({
                    'sender': name,
                    'receiver': receiver,
                    'TP': TP['TP'],
                    'Lumps': TP['Lumps'],
                    'old_CAPEX': TECHNOLOGIC[env.now - 1][TP['TP']]['CAPEX'],
                    'old_OPEX': TECHNOLOGIC[env.now - 1][TP['TP']]['OPEX'],
                    'CAPEX': TP['CAPEX'],
                    'OPEX': TP['OPEX'],
                    'status': 'project',
                    'capacity': TP['Lumps'] * project['MW'],
                    'MWh': TP['Lumps'] * project['MW'] * 24 * 30 * project['CF'],
                    'avoided_emissions': TECHNOLOGIC[env.now - 1][TP['TP']]['avoided_emissions'] * TP['Lumps'],
                    'emissions': TECHNOLOGIC[env.now - 1][TP['TP']]['emissions'] * TP['Lumps']})
                if TP['source_of_TP'] in AUCTION_WANTED_SOURCES:
                    project.update(
                        {'status': 'bidded',
                         'receiver': 'EPM'})
                else:
                    project.update({'status': 'project'})
                code = uuid.uuid4().int
                CONTRACTS[env.now].update({code: project})

            for _ in portfolio_of_projects:
                """ now we have to resend the "projects in the portfolio_of_projects dictionary """
                i = portfolio_of_projects[_].copy()
                number = np.random.poisson(1)
                BB_list = []
                for j in sorted(list(AGENTS[env.now - 1].values()), key=lambda x: x['financing_index'][i['source']],
                                reverse=True):
                    BB_list.append(j['name'])
                for bank in i['failed_attempts']:
                    BB_list.remove(bank)
                if len(BB_list) > 0 and number < len(BB_list):
                    BB = BB_list[number]
                else:
                    BB = BB_list[-1] if len(BB_list) > 0 else random.choice(
                        BB_NAME_LIST)  # if there are items in the list, it chooses the last one, if not, it simply chooses randomly from the possible banks
                project = i.copy()
                project.update({'sender': name,
                                'receiver': BB,
                                'status': 'project'})
                CONTRACTS[env.now].update({
                    project['code']: {project}
                })

            #################################################################
            #                                                               #
            #    Now, the EP analyses what happened to the system due to    #
            #                       its intervention                        #
            #                                                               #
            #################################################################
            add_source = source_reporting_FF(name)
            for entry in dd_source['ranks']:
                dd_source['ranks'][entry] *= (1 - discount)
                dd_source['ranks'][entry] += add_source[entry]

            #################################################################
            #                                                               #
            #          And then, the EP will decide what to do next         #
            #                                                               #
            #################################################################
            if env.now > 0:
                decision_var = max(0, min(1, private_deciding_FF(name)))
                decisions = evaluating_FF(name)

            #############################################################
            #                                                           #
            #  Before leaving, the agent must update the outside world  #
            #                                                           #
            #############################################################
            AGENTS[env.now].update({
                name:
                    {"genre": genre,
                     "accepted_sources": accepted_sources,
                     "name": name,
                     "wallet": wallet,
                     "profits": profits,
                     "EorM": EorM,
                     "subgenre": subgenre,
                     "capacity": capacity,
                     "portfolio_of_plants": portfolio_of_plants,
                     "portfolio_of_projects": portfolio_of_projects,
                     "periodicity": periodicity,
                     "subgenre_price": subgenre_price,
                     "tolerance": tolerance,
                     "last_acquisition_period": last_acquisition_period,
                     "dd_profits": dd_profits,
                     "dd_source": dd_source,
                     "action": action,
                     "dd_kappas": dd_kappas,
                     "dd_qual_vars": dd_qual_vars,
                     "dd_backwardness": dd_backwardness,
                     "dd_avg_time": dd_avg_time,
                     "dd_discount": dd_discount,
                     "dd_strategies": dd_strategies,
                     }})
            profits_dedicting_FF(name)
            if env.now > 0:
                post_evaluating_FF(decisions['strikes'], name)

            yield env.timeout(1)


class Demand(object):
    def __init__(self, env, initial_demand, specificities):
        self.env = env
        self.genre = 'D'
        self.name = 'D'
        self.initial_demand = initial_demand
        self.specificities = specificities
        self.action = env.process(self.run_DD(self.genre,
                                              self.name,
                                              self.initial_demand,
                                              self.specificities)),

    def run_DD(self,
               genre,
               name,
               initial_demand,
               specificities):

        CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, DEMAND, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.DEMAND, config.env

        while True:

            from_time_to_agents_FF(AGENTS)
            from_time_to_agents_FF(TECHNOLOGIC)

            if env.now % 10 == 0:
                print('time', env.now)

            when = specificities['when']
            increase = specificities['increase']
            green_awareness = specificities['green_awareness']

            if env.now == 0:
                DEMAND = {env.now:
                              {'E': initial_demand['E'],
                               'M': initial_demand['M']}
                          }
            else:
                DEMAND.update({env.now: {'E': DEMAND[env.now - 1]['E'],
                                         'M': DEMAND[env.now - 1]['M']}})

            if env.now % when == 0 and env.now != 0:

                """ first, we get how much green is E or M"""
                greeness = {'E': 0, 'M': 0}
                total = 0
                for month in range(env.now - when, env.now - 1):
                    if len(MIX[month]) > 1:
                        for _ in MIX[month]:
                            i = MIX[month][_]
                            if i['status'] == 'contracted':
                                if i['green'] is True:
                                    greeness.update({i['EorM']: greeness[i['EorM']] + i['MWh']})
                                total += i['MWh']

                green = {'E': greeness['E'] / total, 'M': greeness['M'] / total}

                expected_increase = (DEMAND[env.now - 1]['E'] + DEMAND[env.now - 1]['M']) * increase
                pendulum_demand = specificities['EorM'] * expected_increase
                prices = weighting_FF(env.now - 1, 'price', 'MWh', MIX, demand=True)
                a = green_awareness
                E_pendulum = a * (prices['E'] / sum(list(prices.values()))) + (1 - a) * green['E']
                E_pendulum *= pendulum_demand
                E_increase = DEMAND[env.now - 1]['E'] * specificities['increase'] + E_pendulum
                M_pendulum = a * (prices['M'] / sum(list(prices.values()))) + (1 - a) * green['M']
                M_pendulum *= pendulum_demand
                M_increase = DEMAND[env.now - 1]['M'] * specificities['increase'] + M_pendulum

                DEMAND.update({env.now:
                                   {'E': (24 * 30) * DEMAND[env.now - 1]['E'] + E_increase,
                                    'M': (24 * 30) * DEMAND[env.now - 1]['M'] + M_increase}})

            yield env.timeout(1)
