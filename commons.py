import random
# !pip install simpy #on colab it must be this pip install thing, dunno why
import numpy as np
from statistics import median
import config
from scipy.stats import halfnorm, norm
from icecream import ic
from config import seed
from scipy.stats import gamma
from matplotlib import pyplot as plt
from collections import OrderedDict


random.seed(seed)


def bla():
    """ example function, blem is a value inside the config file"""
    blem = config.blem  # gets the list
    print(blem)  # prints the list
    blem[0] += 1  # updates the list
    return


def interesting_FF(value=random.uniform(0,1), max_r=0.011, min_r=0.005, below_min=True, curve=0.5):

    interest = max_r - min_r
    interest *= (1 - value) ** curve
    interest += min_r * (1 - value) if below_min is True else min_r

    return interest


def normalize(_list):

    _list = np.array(_list)

    min_var = min(_list)
    if np.any(_list < 0):

        _list = _list + abs(min_var)
        min_var = 0
    # print(min_var)
    max_var = max(_list)

    denominator = (max_var - min_var) if max_var != min_var else 1
    # print(_list)
    _list = _list - min_var
    # print(_list)
    _list = _list / denominator
    # print(_list)

    # _list = [(i - min_var)/denominator for i in _list]

    return _list


def scheduling_FF(dictionary, time, changes):
    """    This function changes a dictionary entry of a certain time to a certain value. It is useful to change
     dictionaries, reload agents from dictionaries and continue the simulation

    :param dictionary:
    :param time:
    :param changes:
    :return:
    """
    to_change = dictionary
    old_entry = [to_change[time][changes[0]]]

    if len(changes) == 2:
        old_entry.append(to_change[time][changes[-2]])
        to_change[time][changes[-2]] = changes[-1]

    elif len(changes) == 3:
        old_entry.append(to_change[time][changes[-3]][changes[-2]])
        to_change[time][changes[-3]][changes[-2]] = changes[-1]

    elif len(changes) == 4:
        old_entry.append(to_change[time][changes[-4]][changes[-3]][changes[-2]])
        to_change[time][changes[-4]][changes[-3]][changes[-2]] = changes[-1]

    else:
        print('exceed len, change the code, my ganzirosis')

    return print('changed', old_entry[1], 'in', old_entry[0], 'to', changes[-1])


"""def striking_FF(list_o_entries, kappa):
    this is the striking function, it returns a dictionary the updated list of entries

    :param list_o_entries:
    :param kappa:
    :return:
    
    for entry in list_o_entries:
        if ('strikes' in list(entry.keys()) and entry.get('strikes') < 0) or ('0_th_var' in list(entry.keys())):
            # we are dealing with one of the qualitative variables and it is the one that got striked or we are dealing 
            with the zeroth

            change = min(np.random.poisson(1)), len(entry.get('ranks') - 1)  # with this we ensure that the chosen
            # thing is within range of the dictionary

            new = sorted(list(entry.get('ranks').items()), reverse=True)[change][0]
            entry.update({'current': new})

            # lastly we reset the strikes

            entry.update({
                'strikes': 10 * kappa
            })

    return list_o_entries"""

"""def new_targeting_FF(name):
    This function returns a dictionary containing the new_eta

    :param name:
    :return:
    
    AGENTS_r, env = config.AGENTS_r, config.env

    target = AGENTS_r[name][env.now - 1]["dd_target"]['current']
    current = AGENTS_r[name][env.now - 1]['current_state']
    ambition = AGENTS_r[name][env.now - 1]['dd_ambition']['current']
    eta_acc = AGENTS_r[name][0]['dd_eta'][
        'current']  # this way we get the first accepted ETA, which is the duration of the target

    new_target = min(target, current) * ambition

    return {'new_eta': env.now + eta_acc, 'new_target': new_target}"""


def finding_FF(complete_dictionary, what,
               how='highest', cond_dict=None):
    """ this function returns the highest, lowest or the median value ('what') for any dictionary entry (MIX, TECHNOLOGIC,
     AGENTS, CONTRACTS) in the form of condiction dict = {'condition' : 'state of the condition'}

    :param complete_dictionary:
    :param what:
    :param how:
    :param cond_dict:
    :return:
    """
    if cond_dict is None:
        cond_dict = {}
    whats, whos, completes = [], [], []

    if len(complete_dictionary) > 0:
        for _ in complete_dictionary:
            i = complete_dictionary.get(_)
            # first we must check if all conditions are good
            if cond_dict != {}:
                for cond_key in cond_dict:
                    condition = cond_dict.get(cond_key)
                    # we loop for each condition in order to check if they are true
                    if i.get(cond_key) == condition:
                        conds_satisfied = True  # this matters more for the first, but it must be here
                        continue  # this command returns to the loop if true
                    else:
                        conds_satisfied = False
                        break  # this gets us out of out loop
            else:
                conds_satisfied = True  # we have no conditions, all things must pass

            if not isinstance(what, dict) or isinstance(what, list):
                if what in list(i.keys()) and conds_satisfied == True:
                    whats.append(i.get(what))
                    whos.append(i.get('name'))
                    completes.append(i)
            else:
                if isinstance(what, dict) and conds_satisfied == True:
                    key = list(what.keys())[0]
                    #ic(i.get(key).get(what.get(key)), i.get('name'))
                    whats.append(i.get(key).get(what.get(key)))
                    whos.append(i.get('name'))
                    completes.append(i)

    if len(whats) == 0:
        # nothing was got, so we must put whats as zero and assign nothing
        whats = [0]
        whos = ['No one']
        completes = ['Nothing']
        idx = 0

    else:

        if how == 'highest':
            idx = whats.index(max(whats))

        elif how == 'lowest':
            idx = whats.index(min(whats))

        elif how == 'median':
            chosen = sorted(whats, reverse=True)[int(len(whats)//2)] if len(whats) % 2 else median(whats)
            #ic(chosen, whats, whos)
            idx = whats.index(chosen)  # if (isinstance(chosen, int) and chosen in whats) is True else whats.index(int(
            # chosen))

        elif how == 'sum':
            whats = [sum(whats)]
            whos = ['twas a sum']
            idx = 0

    return {'name': whos[idx],
            'value': whats[idx],
            'complete_dict_entry': completes}  # [idx]}


def weighting_FF(time, var, weight, dictionary,
                 Type='average', EorM=False, demand=False, public=False, discount=0):
    """ 20/8/21 this used to be price_generator, now it is the weighting_FF because it basically finds a weighted
    average, maximums or minimums of something in relation to other thing in one certain dictionary. Still may have
    some problems for multiple periods. We have to get the average price per source or per type (if we are dealing
    with demand). For that we have to get the price of each plant , multiply it by its MWh, and lastly divide it by
    the total MWh of that source (or type).

    :param time:
    :param var:
    :param weight:
    :param dictionary:
    :param Type:
    :param EorM:
    :param demand:
    :param public:
    :param discount:
    :return:
    """
    STARTING_PRICE = config.STARTING_PRICE
    """ if we are dealing with an EP or TP, then we have to get only the prices for the sources of the same EorM as they """
    if EorM == False:
        denominator = {0: 0, 1: 0, 2: 0}  # {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0} if public == False else
        # {1: 0, 2: 0, 4: 0, 5: 0}
    else:
        denominator = {0: 0, 1: 0, 2: 0}  # if EorM == 'E' else {3: 0, 4: 0, 5: 0}
    var_dict = denominator.copy()  # WE HAVE TO USE COPY, if not, we always divide by itself and get 1...

    """ If time is a number, we need to make sure that the time loop only catches it once"""
    if isinstance(time, list):
        time_loop = []
        for index in range(len(time)):
            time_loop.append(sorted(time, reverse=True)[index] * (1 - discount) ** index)
    else:
        time_loop = [time]

    for period in time_loop:
        if len(dictionary.get(period)) > 0:
            for _ in dictionary.get(period):
                i = dictionary.get(period).get(_)
                if Type == 'average':
                    if i.get('source') in list(denominator.keys()):
                        denominator.update(
                            {
                                i.get('source'): denominator.get(i.get('source')) + i.get(weight)
                            })
                        var_dict.update(
                            {i.get('source'): var_dict.get(i.get('source')) + i.get(var) * i.get(weight)
                             })

                elif Type == 'max':
                    if i.get('source') in list(denominator.keys()):
                        var_dict.update(
                            {i.get('source'): i.get(var) if i.get(var) > var_dict.get(i.get('source')) else None
                             })

                elif Type == 'min':
                    if i.get('source') in list(denominator.keys()):
                        var_dict.update(
                            {i.get('source'): i.get(var) if i.get(var) < var_dict.get(i.get('source')) else None
                             })
        else:
            # print('EXCEPTION')
            # exceptions if there is no entry in the dictionary
            if (var, weight == 'price', 'MWh'):
                if demand != True:
                    var_dict = {0: STARTING_PRICE, 1: STARTING_PRICE, 2: STARTING_PRICE, 3: STARTING_PRICE,
                                4: STARTING_PRICE, 5: STARTING_PRICE}
                else:
                    var_dict = {'E': STARTING_PRICE, 'M': STARTING_PRICE}

        """ if we are dealing with the demand turtle, then we need to sum the different sources into their categories """
    if demand == True:
        E_denominator = (denominator.get(0) + denominator.get(1) + denominator.get(2))
        M_denominator = (denominator.get(3) + denominator.get(4) + denominator.get(5))
        E_price = (var_dict.get(0) + var_dict.get(1) + var_dict.get(2))
        M_price = (var_dict.get(3) + var_dict.get(4) + var_dict.get(5))
        denominator = {'E': E_denominator,
                       'M': M_denominator}
        var_dict = {'E': E_price,
                    'M': M_price}
    """ If we are dealing with a public agent, then we need to get also the 12, 45 and 1245 entries in the 
    dictionary """
    if public == True:
        dem_12 = (denominator.get(1) + denominator.get(2))
        var_12 = (var_dict.get(1) + var_dict.get(2))
        dem_45 = (denominator.get(4) + denominator.get(5))
        var_45 = (var_dict.get(4) + var_dict.get(5))
        var_1245 = var_12 + var_45
        dem_1245 = dem_12 + dem_45

        denominator.update({12: dem_12, 45: dem_45, 1245: dem_1245})
        var_dict.update({12: var_12, 45: var_45, 1245: var_1245})

    for _ in list(denominator.keys()):
        i = denominator.get(_) if denominator.get(_) > 0 else 1
        j = var_dict.get(
            _)  # EDIT AUGUST 2021, we used to have this if to avoid zero pricing. Not necessary anymore, in fact we
        # will need it. if var_dict.get(_) else STARTING_PRICE this is to avoid zero pricing for sources not used yet
        var_dict.update({_: j / i})

    return var_dict


def private_reporting_FF(genre):
    #  EorM='No'):
    """ Internal function called by source_reporting_FF. this is the reporting_FF, it reports both the backwards and
     forwards for private agents. Technology producers don't report because they can't switch the source, but banks and
      energy providers yes. Backwards basically reports what happened

    :param genre:
    :param EorM:
    :return:
    """
    r, TECHNOLOGIC, AGENTS, DEMAND, MIX, env, AMMORT = config.r, config.TECHNOLOGIC, config.AGENTS, config.DEMAND, config.MIX, config.env, config.AMMORT

    backward_dict, forward_dict = {}, {}

    Sources = [0, 1, 2]  # if EorM == 'E' else [3, 4, 5]

    for source in Sources:
        forward_dict.update({source: 0})
        chosen_TP = forward_dict.copy()  # only used for the BB

    """
    On to the forward_dict
    """

    for TP in TECHNOLOGIC.get(env.now - 1):
        # we have the highest MW in order to compare things, the maximum price
        technology = TECHNOLOGIC[env.now - 1][TP].copy()
        demand_now = DEMAND.copy().get(env.now - 1)  # .get(technology.get('EorM'))
        source = technology['source']
        max_price = max(
            finding_FF(MIX.get(env.now - 1), 'price', 'highest', {'source': source})['value'],
            finding_FF(MIX.get(env.now - 1), 'price', 'highest')['value']
        )
        # max_price = finding_FF(TECHNOLOGIC.get(env.now - 1), 'MW', 'highest', {'EorM': EorM})[
        # 'value'] if max_price == 0 else max_price  # if there is no contracted capacity of the source, we attempt to
        # the get the highest price for the eletrcitity or molecule part.
        lumps = np.floor(demand_now / technology['MW'])
        try:
            if source == list(AGENTS[env.now - 1]['BNDES']['source'][0].keys())[0] and technology['name'] in AGENTS[env.now - 1]['BNDES']['accepted_tps']:
                financing_RISK = AGENTS[env.now - 1]['BNDES']['disclosed_var']
            else:
                financing_RISK = True  # -1 * AGENTS[env.now - 1]['BNDES']['disclosed_var']
            interest = r if financing_RISK is True else interesting_FF(financing_RISK)

            NPV = npv_generating_FF(interest, technology.get('lifetime'), lumps, technology.get('MW'),
                                    technology.get('building_time'), technology.get('CAPEX'),
                                    technology.get('OPEX'), max_price, technology.get('CF'), AMMORT)

        except:
            NPV = npv_generating_FF(
                r, technology.get('lifetime'), lumps, technology.get('MW'),
                technology.get('building_time'), technology.get('CAPEX'),
                technology.get('OPEX'), max_price, technology.get('CF'), AMMORT, reinvest=True)

        # print(env.now, source, max_price, lumps, NPV)

        if NPV > forward_dict[technology['source']]:
            forward_dict.update({technology['source']: NPV})
            chosen_TP.update({technology['source']: TP})

    if genre == 'BB':
        # if the agent is a bank, it is actually interested in the principal
        for source in Sources:
            technology = TECHNOLOGIC[env.now - 1][chosen_TP[source]]
            principal = (technology.get('CAPEX') * (1 + r)) ** technology.get('building_time')
            forward_dict.update({source: principal})

    """
    Now to the backward_dict
    """

    for source in Sources:
        specific_profits = 'profits' + str(source)  # we have to get the profits for that source of the
        current = finding_FF(AGENTS.get(env.now - 1), specific_profits, 'sum',
                             {'genre': genre})['value']  # we now have the current profits for that source in that agent
        # print(current)
        current += Sources[source]  # we now add to the current profits what we had
        backward_dict.update({source: current})  # and update the backward_dict dictionary
    # print(backward_dict, forward_dict)

    for source in config.RISKS:
        # backward_dict[source] *= (1 - config.RISKS[source])
        forward_dict[source] *= (1 - config.RISKS[source])
        # print(config.RISKS)

    return {"backward_dict": backward_dict, "forward_dict": forward_dict}


def public_reporting_FF(rationale):
    """
    Internal function called by source_reporting_FF. This is the reporting_FF, it reports both the backwards and
    forwards for public agents.
    :param rationale:
    :return:
    """
    r, TECHNOLOGIC, MIX, AGENTS, CONTRACTS, DEMAND, env = config.r, config.TECHNOLOGIC, config.MIX, config.AGENTS, config.CONTRACTS, config.DEMAND, config.env

    backward_dict = {}

    Sources = [0, 1, 2]

    for source in Sources:
        backward_dict[source] = 0
    forward_dict = backward_dict.copy()

    """
    On to the forward_dict
    """

    # if the rationale is innovation or internalization, the policy maker basically analyzes agents: on the
    # backward_dict for what happened and on the forward_dict for what could've happend. For the what could've it
    # basically analyzes what would be the effects if all profits were reinvested. For emissions it is a little more
    # complicated: it needs to get the avoided emissions for the backwards and get the whole demand for electricity or
    # molecules and look what would be the avoided emissions if the best technology in each source was used.

    if rationale == 'green':
        # the rationale is green, so we need to look at the avoided emissions
        for TP in TECHNOLOGIC[env.now - 1]:
            # we have the highest MW in order to compare things, the maximum price
            technology = TECHNOLOGIC[env.now - 1][TP]
            demand_now = DEMAND.copy().get[env.now - 1]
            lumps = np.floor(demand_now / technology['MW'])

            avoided_emissions = technology['avoided'] * lumps

            if technology['source'] in Sources and avoided_emissions > forward_dict[technology['source']]:
                forward_dict[technology['source']] = avoided_emissions

        if len(MIX[env.now - 1]) > 0:
            for code in MIX[env.now - 1]:
                plant = MIX[env.now - 1][code]
                if 'status' == 'contracted':
                    # the plant was contracted at that period
                    current = plant['avoided']  # we now have the current avoided emissions of that plant
                    # current += Sources[plant['source']]  # we now add to the current profits what we had
                    backward_dict[plant['source']] += current  # and update the backward_dict dictionary

    else:
        # the rationale is either internalization (capacity) or innovation (R&D), so we have to check the agents
        # dictionary instead
        for _ in AGENTS[env.now - 1]:
            agent = AGENTS[env.now - 1][_]

            if agent['genre'] == 'TP':
                profits = agent['profits']
                var = agent['RandD'] if rationale == 'innovation' else agent['capacity']  # if the rationale is
                # innovation we are looking into R&D, if not then the rationale isinternalization and we are looking at
                # the productive capacity we now sum the two, because it would mean what would happen if all profits
                # were reinvested into R&D
                backward_dict[agent['Technology']['source']] += var
                forward_dict[agent['Technology']['source']] += profits + var

        # print(backward_dict, forward_dict)

    return {"backward_dict": backward_dict, "forward_dict": forward_dict}


def sourcing_FF(backward_dict, forward_dict, backwardness, index_dict=None):
    """ Internal function called by source_reporting_FF. Transforms the results of the reporting function into the
     new_dd_source

    :param backward_dict:
    :param forward_dict:
    :param index_dict:
    :param backwardness:
    :return:
    """
    bwd = backwardness
    back, forw = backward_dict, forward_dict
    new_dd_source = {}

    # if index_dict is None:

    index_dict = {0: 1, 1: 1, 2: 1}

    back_std = np.std(list(backward_dict.values())) * config.RANDOMNESS
    forw_std = np.std(list(forward_dict.values())) * config.RANDOMNESS

    for source in back:

        source_back = np.random.normal(back[source], back_std)
        source_forw = np.random.normal(forw[source], forw_std)

        # ic(source, source_back, back[source], source_forw, forw[source])

        new_dd_source[source] = index_dict[source] * (bwd * source_back + (1 - bwd) * source_forw)
    """if config.env.now > config.FUSS_PERIOD:
        print(new_dd_source)"""
    return new_dd_source


def source_reporting_FF(name, backwardness, index_dict=None):
    """ This function produces the dd_source dict with the scores for sources according to the agent's characteristics.

    :param backwardness:
    :param index_dict:
    :param name:
    :return:
    """

    AGENTS, env = config.AGENTS, config.env
    agent = AGENTS[env.now-1][name]
    genre = agent['genre']

    if genre in ['EPM', 'TPM', 'DBB']:
        report = public_reporting_FF(agent['rationale'])

    else:
        report = private_reporting_FF(genre)

    new_dd_source = sourcing_FF(report['backward_dict'],
                                report['forward_dict'],
                                backwardness,
                                index_dict)

    return new_dd_source


def indexing_FF(name):
    AGENTS, env, CONTRACTS, INSTRUMENT_TO_SOURCE_DICT, MIX = config.AGENTS, config.env, config.CONTRACTS, config.INSTRUMENT_TO_SOURCE_DICT, config.MIX

    agent = AGENTS[env.now - 1][name]
    index = {0: 0, 1: 0, 2: 0}

    if len(MIX[env.now - 1]) > 0:
        for _ in MIX[env.now - 1]:
            plant = MIX[env.now - 1][_]
            if plant['EP'] == name:
                index[plant['source']] += plant['MWh'] * plant['price'] - plant['OPEX']

        if sum(index.values())>0:
            # check if the agent has any plants at all
            for source in list(index.keys()):
                index[source] = index[source] / sum(index.values())

    else:
        denominator = 1/len(index)
        index = {0: denominator, 1: denominator, 2: denominator}

    """for source in index:
        index[source] *= (1 - agent['dd_discount']['current'])

    if agent['genre'] == 'TPM':
        # we are dealing with the TPM, so it has to check the last round of incentives

        for contract in CONTRACTS[env.now - 1]:
            if 'sender' == 'TPM':
                # that contract was an incentive, so it goes into account

                index[contract['source']] += contract['value']

    elif agent['genre'] == 'EPM':

        # The EPM does something a little more complicated: carbon tax does not go into account; FiT are deploy-based 
        and auctions are deploy-based
        FiTs = {1: 0, 2: 0, 4: 0, 5: 0}
        for policy in agent['Policies']:
            Policy = agent['Policies'][policy]
            if Policy['instrument'] == 'FiT':
                source_list = INSTRUMENT_TO_SOURCE_DICT[Policy['source']]
                for source in source_list:
                    FiTs[source] += Policy['value']

        for plant in MIX[env.now - 1]:
            Plant = MIX[env.now - 1][plant]
            if 'auction_contracted' in Plant:
                # that was plant was auction_contracted
                index[Plant['source']] += Plant.get('MWh') * Plant.get('price')
            else:
                # that plant was not auction contracted, so we just have to get how much the EPM paid for
                index[Plant['source']] += (Plant.get('MWh') * Plant.get('price')) * FiTs[Plant['source']]

    elif agent['genre'] in ['BB', 'DBB']:

        for source in index:
            index += agent['finance_index'][source] - AGENTS[env.now - 2][name]['finance_index'][source]"""

    return index


def thresholding_FF(threshold, disclosed_var, decision_var):
    """ This function checks if the treshold for changing the disclosed value was reached. If it was not, the disclosed
    value remains the same, if it was, then the disclosed value is changed to the current decision value. It returns

    :param threshold:
    :param disclosed_var:
    :param decision_var:
    :return:
    """
    threshold_upper, threshold_lower = disclosed_var + threshold, disclosed_var - threshold

    if not threshold_lower < decision_var < threshold_upper:
        disclosed_var = decision_var

    # print(config.env.now,decision_var, disclosed_var)

    return disclosed_var


"""def from_time_to_agents_FF(dictionary):"""
"""     This function gets the dictionaries that are created by time and arrange copys as other dictionaries reversed,
     i.e., time by agents

    :param dictionary:
    :return:
    """
"""AGENTS, TECHNOLOGIC, AGENTS_r, TECHNOLOGIC_r, env = config.AGENTS, config.TECHNOLOGIC, config.AGENTS_r, config.TECHNOLOGIC_r, config.env

    for _ in dictionary.get(env.now):
        i = dictionary.get(env.now).get(_).copy()
        if dictionary == AGENTS:
            dictionary_r = AGENTS_r

        elif dictionary == TECHNOLOGIC:
            dictionary_r = TECHNOLOGIC_r

        else:
            print('not coded')

        dictionary_r[_] = {env.now: i}

    return"""


def evaluating_FF(name, add=None, change=None):
    """
    This function evaluates what happen to a certain agent at the previous period

    :param change:
    :param name:
    :return: dictionary as following {'action' : action, 'strikes' : strikes}
    """
    if change is None:
        change = {'EP': True,
                  'TP': True,
                  'BB': True,
                  'EPM': True,
                  'TPM': True,
                  'DBB': True}
        # this controls if one agent can add or not
    if add is None:
        add = {'EP': False,
               'TP': False,
               'BB': False,
               'EPM': False,
               'TPM': False,
               'DBB': False}
    env, AGENTS = config.env, config.AGENTS
    randomness = config.RANDOMNESS

    # first we have to get the information of the agent from the specififed dictionary

    memory = AGENTS[env.now - 1][name]['memory'][0]
    LSS_thresh = AGENTS[env.now - 1][name]['LSS_thresh'][0]
    discount = AGENTS[env.now - 1][name]['discount'][0]
    genre = AGENTS[env.now - 1][name]['genre']
    impatience = AGENTS[env.now - 1][name]['impatience'][0]

    # ic(name, env.now, impatience)

    verdict = 'keep'
    strikes = False
    impatience_increase = 0

    if random.uniform(0, 1) > randomness:

        # if there is more time than the memory, then the agents do things and we can look into things now
        present, hist = [], []  # first we start these two numbers, they'll be used for the ratio
        if genre in ['DBB', 'TPM', 'EPM']:
            # we are dealing with public agents
            # eta_acc = AGENTS_r[name][env.now - 1]['dd_eta']['current']
            # target = AGENTS_r[name][env.now - 1]['dd_target']['current']
            # present = AGENTS[env.now - 1][name]['current_state']
            # before = 0
            start = max(0, env.now - 1 - 2 * memory)
            end = env.now - 1
            for time in range(start, end):
                _append = AGENTS[time][name]['current_state'] * ((1 - discount) ** (env.now - time))
                _to_append = hist if time < env.now - 1 - memory else present
                _to_append.append(_append)
            # increase = (current - before) / before
            # eta_exp = target / ((1 + increase) * current)
        else:
            # we are dealing with private agents
            for time in range(max(0, env.now - 1 - memory), env.now - 1):
                for _ in AGENTS[time]:
                    i = AGENTS[time][_]

                    if i['genre'] == genre:
                        _append = i['profits'] * ((1 - discount) ** (env.now - time))
                        if i['name'] == name:
                            _to_append = present
                        else:
                            _to_append = hist

                        _to_append.append(_append)
            # present += AGENTS[env.now-1][name]['profits']

        # ic(present, np.mean(hist))
        upper_cond = np.mean(present) > (1 + LSS_thresh) * np.mean(hist)
        lower_cond = np.mean(present) < (1 - LSS_thresh) * np.mean(hist)

        if upper_cond is True or lower_cond is True:
            if upper_cond is True:
                # ic(name, present, (1 - LSS_thresh) * np.mean(hist), LSS_thresh, np.mean(hist))
                # current is better than hist, so now we run the distribution
                verdict = 'add' if add[genre] is True else 'keep'
                impatience_increase = -1
                ratio = np.mean(hist) / np.mean(present)
            elif lower_cond is True:
                # ic(name, present, (1 - LSS_thresh) * np.mean(hist), LSS_thresh, np.mean(hist))
                # current is better than hist, so now we run the distribution
                verdict = 'change' if change[genre] is True else 'keep'
                impatience_increase = 1
                ratio = np.mean(present) / np.mean(hist)
            else:
                print('deu ruim')

            # ic(name, env.now, ratio, np.mean(present), np.mean(hist), verdict)
            for attempt in range(1, impatience):
                dist = np.random.beta(1, 3)
                if dist > ratio:
                    # ic(name, env.now, dist, ratio, np.mean(present), np.mean(hist), verdict)
                    strikes = True
                    break

    else:
        choice_list = ['keep']
        if change[genre] is True:
            choice_list.append('change')

        if add[genre] is True:
            choice_list.append('add')

        verdict = random.choice(['keep', 'change', 'add'])
        if verdict == 'keep':
            impatience_increase = 0
        elif verdict == 'change':
            impatience_increase = 1
        elif verdict == 'add':
            impatience_increase = -1
        verdict = verdict if verdict in choice_list else 'keep'
        """dist = np.random.beta(1, 3)
        ratio = np.random.beta(1, 3)
        strikes = True if dist > ratio else False"""

        strikes = random.choice([True, False])

        """
        verdict = random.choice(choice_list)
        strikes = random.choice([True, False])
        impatience_increase = random.choice([-1, 0, 1])
        """

    # ic(name, verdict, strikes, env.now)

    return {'verdict': verdict, 'strikes': strikes, 'impatience_increase': impatience_increase}


def strikable_dicting(strikables_dict):
    removable = []
    # print('before', strikables_dict)

    for _ in strikables_dict:
        i = strikables_dict[_]
        if len(i) == 1:
            removable += [_]
    try:
        [strikables_dict.pop(k) for k in removable]

    except:
        strikables_dict = strikables_dict

    # ic('strikables_dict', strikables_dict)

    # print('after', strikables_dict)

    return strikables_dict


def post_evaluating_FF(strikes, verdict, name, strikables_dict):
    """

    :param strikes:
    :param verdict:
    :param name:
    :param strikables_dict: dict containing the variables that can be changed {'var 1': list of that variable}
    :return:
    """

    # ic(strikes, verdict, name)

    AGENTS, env = config.AGENTS, config.env

    # dd_qual_vars = AGENTS[env.now][name]['dd_qual_vars']

    if strikes is True:
        # there was another strike

        striked = random.choice(list(strikables_dict.keys()))

        options = strikables_dict[striked].copy()  # to avoid changing the actual dictionary
        previous = strikables_dict[striked].copy()
        if striked != 'source':
            chosen = int(random.uniform(0, len(options)))
            chosen_entry = strikables_dict[striked][chosen]
        else:

            """_list = [list(x.values())[0] for x in strikables_dict[striked]]
            # _list = normalize(_list)
            _sources = [list(x.keys())[0] for x in strikables_dict[striked]]

            chosen_entry = strikables_dict[striked][_list.index(max(_list))]"""

            chosen = min(int(
                np.random.poisson(.355)), len(options) - 1)  # this poisson has 70% chance of choosing the first

            # print(name, 'went from ', _sources[0], ' to ',  chosen_entry)

            chosen_entry = sorted(strikables_dict[striked], reverse=True, key=lambda x: list(x.values())[0])[chosen]

        # print('before striking', strikables_dict[striked]) if striked == "source" else None
        options.remove(chosen_entry)
        # print('after striking', strikables_dict[striked]) if striked == "source" else None

        # print('before rebuilding', strikables_dict[striked]) if striked == "source" else None
        options = [chosen_entry] + strikables_dict[striked]
        # print('after rebuilding', strikables_dict[striked]) if striked == "source" else None

        # print(striked)

        options = [options[i] for i in range(len(options)) if i == options.index(options[i])]

        AGENTS[env.now][name][striked] = options
        # print(AGENTS[env.now][name][striked])

        # print(name, 'changed', striked, 'from', AGENTS[env.now - 1][name][striked][0], 'to', chosen_entry, 'at', env.now, 'period') if striked == "source" else None
        # ic(AGENTS[env.now][name]["LSS_tot"])
        AGENTS[env.now][name]["LSS_tot"] += 1 if options != previous else 0  # If the process didn't actually change anything, then it doesn't count
        # print('changed') if options != previous else print('not changed')
        # print(options, previous)
        # ic(AGENTS[env.now][name]["LSS_tot"])

        _memory = AGENTS[env.now][name]['memory'][0]
        if verdict in ['add', 'change'] and env.now > _memory + 1:
            """
            If the agent added or changed something, it resets its impatience: impatience only builds up until something 
            happens
            """
            # print('Impatience of agent', name, 'was reset at period', env.now, 'from', AGENTS[env.now][name]['impatience'][0], ' to ', AGENTS[0][name]['impatience'][0])
            AGENTS[env.now][name]['impatience'][0] = AGENTS[_memory][name]['impatience'][0]

    # updated the list

    return


def private_deciding_FF(name):
    AGENTS, env = config.AGENTS, config.env

    randomness = config.RANDOMNESS
    agent = AGENTS[env.now - 1][name].copy()
    past_weight = agent['past_weight'][0]
    previous_var = agent['decision_var']

    """if env.now - 1 - memory < 0:
        # we cannot search in negative periods
        start = 0
        end = env.now + memory  # we have to search unsimulated periods to get zero results and keep the ratio
    else:
        start = env.now - 1 - memory
        end = env.now - 1"""

    if random.uniform(0, 1) > randomness:

        memory = agent['memory'][0]
        discount = agent['discount'][0]
        genre = AGENTS[env.now - 1][name]["genre"]

        start = max(0, env.now - 1 - memory)
        end = env.now - 1

        profits = []

        for period in range(start, end+1):
            for _ in AGENTS[period]:
                i = AGENTS[period][_]

                if i['genre'] == genre:
                    profits.append(i['profits'] * ((1 - discount) ** (end - period)))

                """ profits.append(finding_FF(AGENTS[period], 'profits', i, {'genre': genre})['value'] * (
                        (1 - discount) ** (end - 1 - period)))"""
                """medians.append(finding_FF(AGENTS[period], 'profits', 'median', {'genre': genre})['value'] * (
                        (1 - discount) ** (end - 1 - period)))"""
        # print(env.now, AGENTS[end][name]["profits"], AGENTS[end][name]["profits"] * ((1 - discount) ** (end - end)))
        # print(profits)
        profits.remove(AGENTS[end][name]["profits"])
        profits += [AGENTS[end][name]["profits"]]  # with this we ensure that our profits is the last one
        profits = normalize(profits)
        ratio = (np.mean(profits) - profits[-1])
        # ic(AGENTS[end][name]["profits"])
        # ic(np.mean(profits))

        # new_value = max(0, min(1, (1 - past_weight) * ratio + past_weight * previous_var, 1))
        addition = halfnorm.rvs(loc=0, scale=abs(
            ratio))  # random.triangular(0, abs(ratio), 1)  #abs(random.normalvariate(0, abs(ratio)))
        addition = addition / config.doug if ratio > 0 else -addition / config.doug
        new_value = max(0, min(1, previous_var + (1 - past_weight) * addition))
        # ic(env.now, name, previous_var, ratio, new_value)

    else:
        _random = random.normalvariate(previous_var, (1 - past_weight))
        new_value = max(0, min(1, _random))

    return new_value


def public_deciding_FF(name):
    """ This function tells the ratio of 'effort' for the public agents as well as the current state of the analyzed
     variable

    :param name:
    :return:
    """
    AGENTS, AGENTS, MIX, env = config.AGENTS, config.AGENTS, config.MIX, config.env

    randomness = config.RANDOMNESS

    now = AGENTS[env.now - 1][name]

    rationale = now['rationale'][0]
    memory = now['memory'][0]
    discount = now['discount'][0]
    # target = now['target'][0]
    # eta_acc = now['dd_eta']['current']
    past_weight = now['past_weight'][0]
    previous_var = now['decision_var']
    # SorT = now['dd_SorT']['current']

    # ratio = previous_var
    if random.uniform(0, 1) > randomness:
        """if rationale == 'green':
            # the policy maker wants less emissions
            dictionary = MIX
            restriction = {'status': 'contracted'}
            rationale = 'avoided'
        else:
            dictionary = AGENTS
            restriction = {'genre': 'TP'}
            rationale = 'RandD' if rationale == 'innovation' else 'capacity'"""
        results = []
        start = max(0, env.now - 1 - memory)
        end = env.now
        rationale_past = AGENTS[start][name]['rationale'][0]
        if rationale_past == rationale:
            for period in range(start, end):
                current = AGENTS[period][name]['current_state'] * ((1-discount) ** (env.now - period))
                results.append(current)

            results = normalize(results)
            ratio = (np.mean(results) - results[-1])
            addition = halfnorm.rvs(loc=0, scale=abs(ratio))
            addition = addition / config.doug if ratio > 0 else -addition / config.doug
            new_value = max(0, min(1, previous_var + (1 - past_weight) * addition))

        else:
            new_value = previous_var

        # new_value = max(0, min(1, (1 - past_weight) * ratio + past_weight * previous_var))
        # new_value = max(0, min(1, random.normalvariate(previous_var, (1 - past_weight)*ratio)))

        # new_value = max(0, min(1, previous_var + (1 - past_weight) * addition))

        """if env.now > config.FUSS_PERIOD:
            ic(results, env.now, name, previous_var, ratio, addition, addition*10, (1 - past_weight), new_value)"""

    else:
        _random = random.normalvariate(previous_var, (1 - past_weight))
        new_value = max(0, min(1, _random))
    # print(name, env.now, new_value)

    return new_value


def current_stating_FF(rationale):
    """ This function returns the current state of the observed variable according to the rationale. For policy makers
     only

    :param rationale:
    :return:
    """
    MIX, AGENTS, env = config.MIX, config.AGENTS, config.env
    if rationale == 'green':
        # the policy maker wants less emissions
        dictionary = MIX[env.now-1] if env.now > 1 else None
        restriction = ['status', 'contracted']
        rationale = 'avoided_emissions'

    else:
        dictionary = AGENTS[env.now-1] if env.now > 1 else None
        restriction = ['genre', 'TP']
        rationale = 'RandD' if rationale == 'innovation' else 'capacity'

    current_state = 0
    if dictionary is not None:
        for _ in dictionary:
            i = dictionary[_]
            # print(i)
            if i[restriction[0]] == restriction[1]:
                current_state += i[rationale]

    """try:
        current_state = finding_FF(dictionary[env.now], rationale, 'sum', restriction)['value']
    except:
        print("exception...")
        current_state = 0"""

    return current_state


def npv_generating_FF(interest, time, lumps, MW, building_t, capex, opex, p, capacity_factor, ammort_t,
                      cash_flow_RISK=0, financing_RISK=0, true_capex_and_opex=False, reinvest=False):
    """ this function produces the NPV for a certain investment. risks set to 0 mean that there exists no risk

    :param interest:
    :param time:
    :param lumps:
    :param MW:
    :param building_t:
    :param capex:
    :param opex:
    :param p:
    :param capacity_factor:
    :param ammort_t:
    :param cash_flow_RISK:
    :param financing_RISK:
    :param true_capex_and_opex:
    :return:
    """
    ### Credits to Alexandre Mejdalani from the gavetário ###

    financing_RISK = 0 # we have to deal with this later

    CashFlow = []
    if true_capex_and_opex == False:
        caPEX = capex * lumps
        oPEX = opex * lumps
    else:
        caPEX = capex
        oPEX = opex

    if reinvest is False:
        # Principal = caPEX * (1 + interest * (1 - financing_RISK)) ** building_t  # Present Value of the Principle
        Principal = caPEX * ((1 + interest) ** building_t)  # Present Value of the Principle
    else:
        Principal = 0
        CashFlow.append(caPEX)
        for t in range(building_t):
            CashFlow.append(0)
    Debt = Principal


    for t in range(time):

        inflow = ((p * MW * lumps * (24 * 30 * capacity_factor) - oPEX) * (1 - cash_flow_RISK))

        if Debt > 0:
            Debt *= (1 + interest)  # * (1 - financing_RISK))

            Fee = (Principal / (1 + ammort_t)) + Debt * interest  # * (1 - financing_RISK)

            Debt -= Fee

            inflow -= Fee

        CashFlow.append(inflow)

    # NPV = [CashFlow[t] / ((1 + interest * (1 - financing_RISK))) ** t for t in range(len(CashFlow))]
    NPV = [CashFlow[t] / ((1 + interest) ** t) for t in range(len(CashFlow))]

    return sum(NPV)


def policymaking_FF(dd, Policies, value,
                    add=False, rationale=False):
    """ This function adds a current policy to the pool of perenious policies as well as change the current policy.
     dd is the dictionary that changed, Policies is the list of current policies Now policies is the only output, which
      is the list of policies dictionary"""

    """ attention: not yet implemented: REPEAT DO NOT PUT POLICY MAKERS ADDING POLICIES

    :param dd:
    :param Policies:
    :param value:
    :param add:
    :return:
    """
    POLICY_EXPIRATION_DATE, AGENTS, env = config.POLICY_EXPIRATION_DATE, config.AGENTS, config.env

    """ first we retire policies that reached their time"""
    for policy in Policies:
        if policy['deadline'] == env.now:
            Policies.pop(policy)

    Policy = Policies[0]

    if add is True:
        # if we are adding a policy, we must first update its deadline and then add it to the
        Policy.update({
            'deadline': env.now + 1 + POLICY_EXPIRATION_DATE,
            'entry_value': value,
            'entry_instrument': instrument,
            'entry_chosen_source': source,
            'budget': wallet * value
        })
        if rationale != False:
            Policy['rationale'] = rationale
        Policies.append(Policy.copy())

    if 'auction_countdown' in Policy:

        Policy.update({'auction_time': False,
                       'auction_countdown': AGENTS[0]['EPM']['COUNTDOWN']})

    return Policies


def profits_dedicting_FF(name):
    """ this function gets the profits_dd and turns them into specific dictionary entries. Needed for the evaluation

    :param name:
    :return:
    """
    AGENTS, env = config.AGENTS, config.env

    agent = AGENTS[env.now][name]
    dd_profits = agent["dd_profits"]

    for source in dd_profits:
        agent.update({'profits' + str(source): dd_profits[source]})

    return


def capital_adequacy_rationing_FF(receivables_dict, wallet):
    """ This function produces the capital adequacy ratio according to the receivables, risk and the reserves (wallet) of a certain bank. car_ratio is a number

    :param receivables_dict:
    :param risk_dict:
    :param wallet:
    :return:
    """
    BASEL, RISKS, MIX, env = config.BASEL, config.RISKS, config.MIX, config.env

    risk_dict = RISKS.copy()# {0: 0, 1: 0, 2: 0}

    """if env.now > 1 and len(MIX[env.now-1]) > 0:
        for _ in MIX[env.now - 1]:
            plant = MIX[env.now - 1][_]
            risk_dict[plant['source']] += plant['MWh']

        total_MWh = sum(risk_dict.values())
        for source in list(risk_dict.keys()):
            risk_dict[source] = risk_dict[source] / total_MWh
        max_percentage = max(list(risk_dict.values()))
        for source in list(risk_dict.keys()):
            risk_dict[source] = max_percentage - risk_dict[source]
            RISKS[source] = risk_dict[source]"""

    recv_w_risk = {source: receivables_dict[source] * (1 + risk_dict[source]) for source in receivables_dict}
    denominator = sum(recv_w_risk[source] for source in recv_w_risk)
    car_ratio = wallet / denominator if denominator > 0 else 2*BASEL
    # ic(car_ratio)
    return car_ratio


"""def source_accepting_FF(accepted_sources, old):
    accepted_sources.update({
        old: True
    })

    return accepted_sources"""


def financing_FF(genre, name, my_wallet, my_receivables, value, financing_index,
                 guaranteeing=False, accepted_source=None, accepted_tps=None):
    CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, BASEL, AMMORT, NPV_THRESHOLD, NPV_THRESHOLD_DBB, INSTRUMENT_TO_SOURCE_DICT, RISKS, TP_THERMAL_PROD_CAP_PCT, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.BASEL, config.AMMORT, config.NPV_THRESHOLD, config.NPV_THRESHOLD_DBB, config.INSTRUMENT_TO_SOURCE_DICT, config.RISKS, config.TP_THERMAL_PROD_CAP_PCT, config.env

    global new_wallet

    new_wallet = my_wallet
    new_receivables = my_receivables.copy()

    """ first the bank collects all the projects adressed to it """

    adressed_projects = {}
    adressed_projects_NPVs = []
    source_price = weighting_FF(env.now - 1, 'price', 'MWh', MIX)
    max_price = max(source_price.values())
    for source in list(source_price.keys()):
        if source_price[source] == 0:
            source_price[source] = max_price

    # print('prices at time', env.now, source_price)

    time = env.now - 1  # - 1  # if genre == 'BB' else env.now -1  # The DBB looks at the present period

    if env.now > 0 and len(CONTRACTS.get(time)) > 0:  # and car_ratio >= BASEL:
        # print('There are contract here')
        for _ in CONTRACTS.get(time):
            i = CONTRACTS.get(time).get(_).copy()
            """ the bank only appends that project if its addressed to it, is a project and it would accept to finance 
            such source"""

            if i.get('receiver') == name and i.get('status') == 'project':

                if genre == 'BB':
                    interest_r = r * (1 + value)
                else:
                    if i['source'] == accepted_source and i['TP'] in accepted_tps:
                        interest_r = interesting_FF(value)
                    else:
                        interest_r = r
                # if the agent is a private bank it increases
                # the general interest rate by (1+risk), on the other hand, if the agent is the development bank it reduces the
                # general interest rate by (1-effort)

                if genre == 'BB':
                    financing_risk = 0 if i.get('guarantee') is True or genre == 'DBB' else value
                else:
                    financing_risk = 0
                cash_flow_risk = 0 if i['auction_contracted'] == True else config.RISKS[i['source']]
                if i['auction_contracted'] == True:
                    price = max(i['auction_price'], source_price[i['source']])
                else:
                    price = source_price[i['source']]
                NPV = npv_generating_FF(interest_r, i.get('lifetime'), i.get('Lumps'), i.get('capacity'),
                                        i.get('building_time'), i.get('CAPEX'), i.get('OPEX'), price, i.get('CF'),
                                        AMMORT, cash_flow_risk, financing_risk, true_capex_and_opex=True)
                # print('at', env.now, 'project from', i.get('receiver'), i.get('TP'), 'has this NPV', NPV)
                """ic(interest_r, i.get('lifetime'), i.get('Lumps'), i.get('capacity'),
                                        i.get('building_time'), i.get('CAPEX'), i.get('OPEX'), price, i.get('CF'),
                                        AMMORT, cash_flow_risk, financing_risk, True, NPV, i['source'])"""
                adressed_projects_NPVs.append({'code': _, 'NPV': NPV, 'capex': i['CAPEX']})
                adressed_projects.update({_: i.copy()})
        if genre == 'BB':
            adressed_projects_NPVs = sorted(adressed_projects_NPVs, key=lambda x: x['NPV'], reverse=True)
        else:
            adressed_projects_NPVs = sorted(adressed_projects_NPVs, key=lambda x: x['capex'])

    for project in adressed_projects_NPVs:
        i = project.get('code')
        j = adressed_projects.get(i)
        k = j.get('source')
        npv = project['NPV']

        new_new_wallet = new_wallet - j.get('CAPEX')
        receiv_value = new_receivables.get(k) + j.get('CAPEX') * (
                interest_r ** j.get('building_time')) if guaranteeing is False else new_receivables.get(k)
        new_new_receivables = new_receivables.copy()
        new_new_receivables.update({k: receiv_value})
        car_ratio = capital_adequacy_rationing_FF(new_new_receivables, new_new_wallet)

        # ic(env.now, name, RISKS[k], value, k, accepted_source, new_new_wallet, car_ratio)
        # ic(j)
        # tresHOLD = NPV_THRESHOLD if genre == 'BB' else NPV_THRESHOLD_DBB
        if genre == 'BB':
            acceptance_cond = RISKS[k] < value and npv > NPV_THRESHOLD and new_new_wallet >= 0
        else:
            if config.FUSS_PERIOD < env.now or AGENTS[env.now - 1]['DD']['Remaining_demand'] < 0:
                acceptance_cond = k == accepted_source and j['TP'] in accepted_tps and new_new_wallet >= 0  # and car_ratio >= BASEL
                print(env.now , 'FLAMENGO') if acceptance_cond == True else print(env.now , 'fogão because', k == accepted_source, j['TP'] in accepted_tps, new_new_wallet >= 0)
                # ic(k, accepted_source, j['TP'], accepted_tps, interest_r, new_new_wallet, acceptance_cond, car_ratio) if acceptance_cond == True else None
            else:
                # if there is no excess demand, BNDES doesn't finance capacity
                acceptance_cond = new_new_wallet >= 0  # and car_ratio >= BASEL
                print(env.now , 'vasquinho') if acceptance_cond == True else print(env.now , 'fluzão')
            # ic(k, accepted_source, j['TP'], accepted_tps, interest_r, new_new_wallet, acceptance_cond)

        """if AGENTS[env.now-1]['DD']['Remaining_demand']>0:
            acceptance_cond=True""" # This was previously here to ensure that some capacity would be financed

        # ic(k, accepted_source, prod_cap_pct, (1-value), npv, car_ratio, BASEL, name, acceptance_cond)
        # print(name, acceptance_cond, new_new_wallet >= 0, car_ratio >= BASEL)
        # print(acceptance_cond is True and new_new_wallet >= 0 and car_ratio >= BASEL)
        if acceptance_cond == True:  # and car_ratio >= BASEL:
            new_wallet = new_new_wallet
            new_receivables = new_new_receivables
            # print("new contract", j)
            new_contract = j.copy()
            # print('project', i, 'approved ', j.copy(), 'by ', name)
            if guaranteeing is True:
                new_contract.update({'sender': 'DBB',
                                     'receiver': j.get('sender'),
                                     'guarantee': True})
            else:
                financing_index.update({k: financing_index.get(k) + j.get('CAPEX')})
                new_contract.update({
                    'sender': name,
                    'receiver': j.get('sender'),
                    'status': 'financed',
                    'ammortisation': env.now + 1 + AMMORT,
                    'risk': RISKS[k],
                    'principal': j.get('CAPEX') * (interest_r ** j.get('building_time')),
                    'r': interest_r})

                new_contract['debt'] = new_contract['principal']

                """if 'guarantee' not in new_contract:
                    new_contract.update({'guarantee': False})"""

            # CONTRACTS.get(env.now).update({i: new_contract})
          #  name, 'financed', new_contract)
        else:
            # print(name, 'Rejected due to')
            # ic(acceptance_cond, new_new_wallet >= 0, car_ratio >= BASEL)
            new_contract = j.copy()

            new_contract.update({'status': 'rejected',
                                 'sender': name,
                                 'receiver': j.get('sender'),
                                 'reason': ['wallet after', new_new_wallet, 'car_ratio', car_ratio],
                                 })
            """if 'guarantee' not in new_contract:
                new_contract.update({'guarantee': False})"""
        CONTRACTS.get(env.now).update({i: new_contract})

    # print(env.now, my_wallet, new_wallet, 'wallets')
    my_wallet = new_wallet
    my_receivables = new_receivables
    return {'wallet': my_wallet, 'receivables': my_receivables, 'financing_index': financing_index}

def bank_sending_FF():
    """

    :return:
    """

    AGENTS, env = config.AGENTS, config.env

    BB_ = []
    for _ in AGENTS[env.now - 1]:
        """ we must check at this period, because the EP goes after the bank """
        agent = AGENTS[env.now - 1][_]
        if agent['genre'] == 'BB':
            BB_.append([agent['name'], agent['decision_var']])
        elif agent['genre'] == 'DBB':
            BB_.append([agent['name'], agent['disclosed_var']])
    number = np.random.poisson(1)
    # print(BB_)
    number = number if number < len(BB_) else len(BB_) - 1
    _BB = sorted(BB_, key=lambda x: x[1])[number][0] if len(BB_) > 0 else None
    # first we sort the list of agents. With the .values() we have both keys and values. The [number]
    # selects the random pick of which source to get. The ['name'] selects the name of the bank

    return _BB


def run_graphs(agents_dict, contracts_dict, mix_dict, technologic_dict, save=False,
               name='test', pathfile='analysis/Figures/', show=False, norm=False, weak=False):
    """
    Function for producing specific graphs for checking how things are going

    :param weak:
    :param show:
    :param save:
    :param name:
    :param pathfile:
    :param agents_dict:
    :param contracts_dict:
    :param mix_dict:
    :param technologic_dict:
    :return:
    """

    #################################################################
    #                                                               #
    #                         BEFORE GRAPHS:                        #
    #                                                               #
    #################################################################

    """
    First we have to produce the four (actually eight) time series that we use:

    1- private adaptation
    2- public adaptation
    3- private goal
    4- public goal

    And their respective speeds (5 through 8)
    """

    """
    1- private adaptation: priv_adaptation
    2- public adaptation: publ_adaptation
    3- private goal
    4- public goal
    """

    _priv_adaptation = {}
    _publ_adaptation = {}
    _priv_goal = {}
    _publ_goal = {}
    time = list(range(0, len(agents_dict)))
    number_of_priv = config.EP_NUMBER + config.TP_NUMBER + config.BB_NUMBER
    number_of_publ = config.PUB_NUMBER

    name = name + "_weak_" if weak is True else name

    for period in agents_dict:
        _priv_adaptation[period] = []
        _publ_adaptation[period] = []
        _priv_goal[period] = []
        _publ_goal[period] = []
        for entry in list(agents_dict[period].keys()):
            agent = agents_dict[period][entry]

            if agent['genre'] != 'DD':
                append = agent['LSS_tot'] if weak == False else agent['LSS_weak']
                genre = agent['genre']
                if genre in ['EP', 'TP', 'BB']:
                    _priv_adaptation[period].append(append)
                    _priv_goal[period].append(agent['profits'])
                    print(agent['name'], genre, append, agent['profits'])
                elif genre in ['DBB', 'TPM', 'EPM']:
                    print(agent['name'], genre, append, agent['current_state'])
                    _publ_adaptation[period].append(append)
                    _publ_goal[period].append(agent['current_state'])

    _thermal_cap = {}
    _wind_cap = {}
    _solar_cap = {}
    _thermal_gen = {}
    _wind_gen = {}
    _solar_gen = {}
    for period in mix_dict:
        _thermal_cap[period] = 0
        _wind_cap[period] = 0
        _solar_cap[period] = 0

        _thermal_gen[period] = 0
        _wind_gen[period] = 0
        _solar_gen[period] = 0

        for entry in list(mix_dict[period].keys()):
            plant = mix_dict[period][entry]

            cap = plant['MW']

            if plant['status'] == 'contracted':
                gen = plant['MWh']
            else:
                gen = 0

            if plant['source'] == 0:
                _thermal_cap[period] += cap
                _thermal_gen[period] += gen

            elif plant['source'] == 1:
                _wind_cap[period] += cap
                _wind_gen[period] += gen

            elif plant['source'] == 2:
                _solar_cap[period] += cap
                _solar_gen[period] += gen

    # ic(_priv_adaptation)
    # ic(DBB_adaptation)
    priv_adaptation = []
    publ_adaptation = []
    priv_goal = []
    publ_goal = []

    thermal_cap = []
    wind_cap = []
    solar_cap = []
    thermal_gen = []
    wind_gen = []
    solar_gen = []

    for period in _priv_adaptation:
        adaptations = sum(_priv_adaptation[period])
        goals = sum(_priv_goal[period])
        priv_adaptation.append(adaptations / number_of_priv)
        priv_goal.append(goals / number_of_priv)

    for period in _publ_adaptation:
        adaptations = sum(_publ_adaptation[period])
        goals = sum(_publ_goal[period])
        publ_adaptation.append(adaptations / number_of_publ)
        publ_goal.append(goals / number_of_publ)

    for period in _solar_gen:
        thermal_cap.append(_thermal_cap[period])
        wind_cap.append(_wind_cap[period])
        solar_cap.append(_solar_cap[period])
        thermal_gen.append(_thermal_gen[period])
        wind_gen.append(_wind_gen[period])
        solar_gen.append(_solar_gen[period])

    print(priv_adaptation)
    print(priv_goal)
    print(publ_adaptation)
    print(publ_goal)
    """
    5 - Speed of private adaptation : priv_adaptation_speed
    6 - Speed of public adaptation : publ_adaptation_speed
    7 - Speed of private goal : priv_goal_speed 
    8 - Speed of public goal : publ_goal_speed
    """

    priv_goal_speed = []
    publ_goal_speed = []
    priv_adaptation_speed = []
    publ_adaptation_speed = []

    for period in range(1, len(_priv_adaptation)):
        priv_goal_speed_append = (
                                         priv_goal[period] - priv_goal[period - 1]
                                 ) / priv_goal[period - 1] if priv_goal[period - 1] > 0 else 1
        publ_goal_speed_append = (
                                         publ_goal[period] - publ_goal[period - 1]
                                 ) / publ_goal[period - 1] if publ_goal[period - 1] > 0 else 1

        priv_adaptation_speed_append = (
                                               priv_adaptation[period] - priv_adaptation[period - 1]
                                       ) / priv_adaptation[period - 1] if priv_adaptation[period - 1] > 0 else 1
        publ_adaptation_speed_append = (
                                               publ_adaptation[period] - publ_adaptation[period - 1]
                                       ) / publ_adaptation[period - 1] if publ_adaptation[period - 1] > 0 else 1

        priv_goal_speed.append(priv_goal_speed_append)
        publ_goal_speed.append(publ_goal_speed_append)
        priv_adaptation_speed.append(priv_adaptation_speed_append)
        publ_adaptation_speed.append(publ_adaptation_speed_append)

    # Program to calculate moving average
    arrpriv_goal = priv_goal_speed
    arrpubl_goal = publ_goal_speed
    arrpriv_adaptation = priv_adaptation_speed
    arrpubl_adaptation = publ_adaptation_speed
    window_size = 12

    i = 0
    # Initialize an empty list to store moving averages
    priv_goal_speed_WA = []
    publ_goal_speed_WA = []
    priv_adaptation_speed_WA = []
    publ_adaptation_speed_WA = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arrpriv_goal) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        windowpriv_goal = arrpriv_goal[i: i + window_size]
        windowpubl_goal = arrpubl_goal[i: i + window_size]

        windowpriv_adaptation = arrpriv_adaptation[i: i + window_size]
        windowpubl_adaptation = arrpubl_adaptation[i: i + window_size]

        # Calculate the average of current window
        window_averagepriv_goal = round(sum(windowpriv_goal) / window_size, 2)
        window_averagepubl_goal = round(sum(windowpubl_goal) / window_size, 2)

        window_averagepriv_adaptation = round(sum(windowpriv_adaptation) / window_size, 2)
        window_averagepubl_adaptation = round(sum(windowpubl_adaptation) / window_size, 2)

        # Store the average of current
        # window in moving average list
        priv_goal_speed_WA.append(window_averagepriv_goal)
        publ_goal_speed_WA.append(window_averagepubl_goal)

        priv_adaptation_speed_WA.append(window_averagepriv_adaptation)
        publ_adaptation_speed_WA.append(window_averagepubl_adaptation)

        # Shift window to right by one position
        i += 1

    priv_goal = [(val - min(priv_goal)) / (max(priv_goal) - min(priv_goal)) for val in priv_goal]
    publ_goal = [(val - min(publ_goal)) / (max(publ_goal) - min(publ_goal)) for val in publ_goal]

    if norm is True:
        # publ_goal = [val / max(publ_goal) for val in publ_goal]

        max_adapt = max(max(priv_adaptation), max(publ_adaptation))
        min_adapt = min(min(priv_adaptation), min(publ_adaptation))

        """priv_adaptation = [(val - min(priv_adaptation)) / (max(priv_adaptation) - min(priv_adaptation)) for val in priv_adaptation]
        publ_adaptation = [(val - min(publ_adaptation)) / (max(publ_adaptation) - min(publ_adaptation)) for val in publ_adaptation]"""
        priv_adaptation = [(val - min_adapt) / (max_adapt - min_adapt) for val in priv_adaptation]
        publ_adaptation = [(val - min_adapt) / (max_adapt - min_adapt) for val in publ_adaptation]

    names_dict = {
        str(priv_adaptation): "private adaptation",
        str(priv_goal): "private goal achievement",
        str(publ_adaptation): "public adaptation",
        str(publ_goal): "public goal achievement",
        str(priv_goal_speed): "speed of private goal achievement",
        str(publ_goal_speed): "speed of public goal achievement",
        str(priv_adaptation_speed): "speed of private adaptation",
        str(publ_adaptation_speed): "speed of public adaptation",
        str(priv_goal_speed_WA): "speed of private goal achievement (weighted average)",
        str(publ_goal_speed_WA): "speed of public goal achievement (weighted average)",
        str(priv_adaptation_speed_WA): "speed of private adaptation (weighted average)",
        str(publ_adaptation_speed_WA): "speed of private adaptation (weighted average)"
    }

    #################################################################
    #                                                               #
    #                          FIRST GRAPH:                         #
    #                 Adaptation in relation to time                #
    #                                                               #
    #################################################################

    fig = plt.figure()
    ax = plt.axes()

    x = priv_adaptation
    y = publ_adaptation

    ax.plot(time, x, color='g', label=names_dict[str(x)])
    ax.plot(time, y, color='b', label=names_dict[str(y)])

    ax.legend()

    plt.xlabel("Time")
    plt.ylabel("Adaptation")

    title = "Adaptation of Private and Public agents over time"
    plt.title(title)

    fig.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        fig.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         SECOND GRAPH:                         #
    #            Speed of Adaptation in relation to time            #
    #                                                               #
    #################################################################

    fig2 = plt.figure()
    ax2 = plt.axes()

    # print(EP_speed_WA)
    # print(DBB_speed_WA)

    ax2.plot(list(range(0, len(priv_adaptation_speed))), priv_adaptation_speed, color='g',
             label='Speed of Adaptation of EPs')
    ax2.plot(list(range(0, len(publ_adaptation_speed))), publ_adaptation_speed, color='b',
             label='Speed of Adaptation of DBB')

    ax2.plot(list(range(0, len(priv_adaptation_speed_WA))), priv_adaptation_speed_WA, color='g',
             label='Speed of Adaptation of EPs (weighted average)', linestyle='dashed')
    ax2.plot(list(range(0, len(publ_adaptation_speed_WA))), publ_adaptation_speed_WA, color='b',
             label='Speed of Adaptation of DBB (weighted average)', linestyle='dashed')

    ax2.legend()

    plt.xlabel("Time")
    plt.ylabel("Adaptation")

    title = "Speed of adaptation of private and public agents over time"

    plt.title(title)

    fig2.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        fig2.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                          THIRD GRAPH:                         #
    #             Goal achievement in relation to time              #
    #                                                               #
    #################################################################

    fig3 = plt.figure()
    ax3 = plt.axes()

    ax3.plot(time, priv_goal, color='g', label='Goal achievement of Private agents')
    ax3.plot(time, publ_goal, color='b', label='Goal achievement of Public agents')

    ax3.legend()

    plt.xlabel("Time")
    plt.ylabel("Goal Achievement")

    title = "Goal achievement of Private and Public agents over time"
    plt.title(title)

    fig3.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        fig3.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         FOURTH GRAPH:                         #
    #         Speed of Goal achievement in relation to time         #
    #                                                               #
    #################################################################

    fig4 = plt.figure()
    ax4 = plt.axes()

    # print(EP_speed_WA)
    # print(DBB_speed_WA)

    ax4.plot(list(range(0, len(priv_goal_speed))), priv_goal_speed, color='g',
             label='Speed of Adaptation of EPs')
    ax4.plot(list(range(0, len(publ_goal_speed))), publ_goal_speed, color='b',
             label='Speed of Adaptation of DBB')

    ax4.plot(list(range(0, len(priv_goal_speed_WA))), priv_goal_speed_WA, color='g',
             label='Speed of Adaptation of EPs (weighted average)', linestyle='dashed')
    ax4.plot(list(range(0, len(publ_goal_speed_WA))), publ_goal_speed_WA, color='b',
             label='Speed of Goal achievement of DBB (weighted average)', linestyle='dashed')

    ax4.legend()

    plt.xlabel("Time")
    plt.ylabel("Adaptation")

    title = "Speed of adaptation of private and public agents over time"

    plt.title(title)

    fig4.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        fig4.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         FIFTH GRAPH:                          #
    #                                                               #
    #################################################################

    """
    Private adaptation in relation to public goals: are private agents adapting towards public goals?
    """

    figA = plt.figure()
    axA = plt.axes()

    color = range(0, len(publ_goal))

    x = publ_goal
    y = priv_adaptation

    points = axA.scatter(x, y, c=color, cmap='viridis')

    figA.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    # axA.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figA.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figA.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         SIXTH GRAPH:                          #
    #                                                               #
    #################################################################

    """
    Private adaptation in relation to public goals in terms of speeds
    """

    figAs = plt.figure()
    axAs = plt.axes()

    x = publ_goal_speed
    y = priv_adaptation_speed

    color = range(0, len(x))

    points = axAs.scatter(x, y, c=color, cmap='viridis')

    figAs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figAs.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figAs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                    SIXTH AND A HALF GRAPH:                    #
    #                                                               #
    #################################################################

    """
    Private adaptation in relation to public goals in terms of speeds
    """

    figAss = plt.figure()
    axAss = plt.axes()

    x = publ_goal_speed_WA
    y = priv_adaptation_speed_WA

    color = range(0, len(x))

    points = axAss.scatter(x, y, c=color, cmap='viridis')

    figAss.colorbar(points)

    # axAss.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figAss.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figAss.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                        SEVENTH GRAPH:                         #
    #                                                               #
    #################################################################

    """
    Public goal achievement in relation to public adaptation: are the adaptation processes leading towards goal completion for policy makers?
    """

    figB = plt.figure()
    axB = plt.axes()

    x = publ_goal
    y = priv_adaptation

    color = range(0, len(publ_adaptation))

    points = axB.scatter(publ_adaptation, publ_goal, c=color, cmap='viridis')

    # axB.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figB.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figB.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figB.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         EIGHTH GRAPH:                         #
    #                                                               #
    #################################################################

    """
    Public goal achievement in relation to public adaptation (speeds)
    """

    figBs = plt.figure()
    axBs = plt.axes()

    x = publ_goal_speed
    y = priv_adaptation_speed

    color = range(0, len(publ_goal_speed))

    points = axBs.scatter(publ_goal_speed, publ_adaptation_speed, c=color, cmap='viridis')

    # axBs.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figBs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figBs.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figBs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                    EIGHTH AND A HALF GRAPH:                   #
    #                                                               #
    #################################################################

    """
    Public goal achievement in relation to public adaptation (speeds weighted average)
    """

    figBss = plt.figure()
    axBss = plt.axes()

    x = publ_goal_speed_WA
    y = priv_adaptation_speed_WA

    color = range(0, len(x))

    points = axBss.scatter(x, y, c=color, cmap='viridis')

    # axBss.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figBs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figBss.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figBss.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         NINTH GRAPH:                          #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to private adaptation: are the adaptation processes leading towards goal completion for private agents?
    """

    figC = plt.figure()
    axC = plt.axes()

    x = priv_goal
    y = priv_adaptation

    color = range(0, len(publ_adaptation))

    points = axC.scatter(x, y, c=color, cmap='viridis')

    # axC.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figC.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figC.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figC.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                          TENTH GRAPH:                         #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to private adaptation: (speeds)
    """

    figCs = plt.figure()
    axCs = plt.axes()

    x = priv_adaptation_speed
    y = priv_goal_speed

    color = range(0, len(publ_goal_speed))

    points = axCs.scatter(x, y, c=color, cmap='viridis')

    # axCs.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figCs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figCs.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figCs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                     TENTH AND A HALF GRA                      #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to private adaptation: (speeds weighted average)
    """

    figCss = plt.figure()
    axCss = plt.axes()

    x = priv_adaptation_speed_WA
    y = priv_goal_speed_WA

    color = range(0, len(x))

    points = axCss.scatter(x, y, c=color, cmap='viridis')

    # axCss.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figCss.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figCss.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figCss.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                        ELEVENTH GRAPH:                        #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public adaptation: Is the public adaptation clashing with the private goals?
    """

    figD = plt.figure()
    axD = plt.axes()

    x = publ_adaptation
    y = priv_goal

    color = range(0, len(publ_adaptation))

    points = axD.scatter(x, y, c=color, cmap='viridis')

    # axD.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figD.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figD.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figD.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         TWELFTH GRAPH:                        #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public adaptation: (speeds)
    """

    figDs = plt.figure()
    axDs = plt.axes()

    x = publ_adaptation_speed
    y = priv_goal_speed

    color = range(0, len(publ_goal_speed))

    points = axDs.scatter(x, y, c=color, cmap='viridis')

    # axDs.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figDs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figDs.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figDs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                    TWELFTH AND A HALF GRAPH:                  #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public adaptation: (speeds)
    """

    figDss = plt.figure()
    axDss = plt.axes()

    x = publ_adaptation_speed_WA
    y = priv_goal_speed_WA

    color = range(0, len(x))

    points = axDss.scatter(x, y, c=color, cmap='viridis')

    # axDss.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figDss.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figDss.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figDss.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                       THIRTEENTH GRAPH:                       #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public goal achivement: are private and public goals clashing?
    """

    figE = plt.figure()
    axE = plt.axes()

    x = priv_goal
    y = publ_goal

    color = range(0, len(publ_adaptation))

    points = axE.scatter(x, y, c=color, cmap='viridis')

    figE.colorbar(points)

    # axE.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figE.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figE.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                         TWELFTH GRAPH:                        #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public goal achievement: (speeds)
    """

    figEs = plt.figure()
    axEs = plt.axes()

    x = priv_goal_speed
    y = publ_goal_speed

    color = range(0, len(publ_goal_speed))

    points = axEs.scatter(x, y, c=color, cmap='viridis')

    # axEs.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figEs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figEs.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figEs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                   TWELFTH AND A HALF GRAPH:                   #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public goal achievement: (speeds)
    """

    figEss = plt.figure()
    axEss = plt.axes()

    x = priv_goal_speed_WA
    y = publ_goal_speed_WA

    color = range(0, len(x))

    points = axEss.scatter(x, y, c=color, cmap='viridis')

    # axEss.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figEss.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figEss.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figEs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                       THIRTEENTH GRAPH:                       #
    #                                                               #
    #################################################################

    """
    Private adaptation in relation to public adaptation: who is accumulating more adaptation?
    """

    figF = plt.figure()
    axF = plt.axes()

    x = priv_adaptation
    y = publ_adaptation

    color = range(0, len(publ_adaptation))

    points = axF.scatter(x, y, c=color, cmap='viridis')

    # axF.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    figF.colorbar(points)

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figF.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figF.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                        FOURTEENTH GRAPH:                      #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public adaptation: (speeds)
    """

    figFs = plt.figure()
    axFs = plt.axes()

    x = priv_adaptation_speed
    y = publ_adaptation_speed

    color = range(0, len(publ_goal_speed))

    points = axFs.scatter(x, y, c=color, cmap='viridis')

    # axFs.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figFs.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figFs.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figFs.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                 FOURTEENTH AND A HALF GRAPH:                  #
    #                                                               #
    #################################################################

    """
    Private goal achievement in relation to public adaptation: (speeds)
    """

    figFss = plt.figure()
    axFss = plt.axes()

    x = priv_adaptation_speed_WA
    y = publ_adaptation_speed_WA

    color = range(0, len(x))

    points = axFss.scatter(x, y, c=color, cmap='viridis')

    # axFss.axline([0, 0], [min(max(x), max(y)), min(max(x), max(y))])

    figFss.colorbar(points)

    plt.xlabel(names_dict[str(x)])
    plt.ylabel(names_dict[str(y)])

    title = names_dict[str(y)] + " in relation to " + names_dict[str(x)]
    plt.title(title)

    figFss.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figFss.savefig(pathfile + file_name)

    #################################################################
    #                                                               #
    #                        Capacity graphs                        #
    #                                                               #
    #################################################################

    figCAP = plt.figure()
    axCAP = plt.axes()

    axCAP.plot(time, thermal_cap, color='g', label='Thermal capacity')
    axCAP.plot(time, wind_cap, color='b', label='Wind capacity')
    axCAP.plot(time, solar_cap, color='r', label='Solar capacity')

    axCAP.legend()

    plt.xlabel("Time")
    plt.ylabel("MWs over time")

    title = "MWs over time"
    plt.title(title)

    figCAP.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figCAP.savefig(pathfile + file_name)

    figGEN = plt.figure()
    axGEN = plt.axes()

    axGEN.plot(time, thermal_gen, color='g', label='Thermal generation')
    axGEN.plot(time, wind_gen, color='b', label='Wind generation')
    axGEN.plot(time, solar_gen, color='r', label='Solar generation')

    axGEN.legend()

    plt.xlabel("Time")
    plt.ylabel("MWhs over time")

    title = "MWhs over time per source"
    plt.title(title)

    figGEN.show() if show is True else None

    if save is True:
        file_name = title + '_' + '_' + name + ".png"
        figGEN.savefig(pathfile + file_name)

