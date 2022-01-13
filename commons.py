import random
# !pip install simpy #on colab it must be this pip install thing, dunno why
import numpy as np
from statistics import median
import config


def bla():
    """ example function, blem is a value inside the config file"""
    blem = config.blem  # gets the list
    print(blem)  # prints the list
    blem[0] += 1  # updates the list
    return


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


def striking_FF(list_o_entries, kappa):
    """ this is the striking function, it returns a dictionary the updated list of entries

    :param list_o_entries:
    :param kappa:
    :return:
    """
    for entry in list_o_entries:
        if ('strikes' in list(entry.keys()) and entry.get('strikes') < 0) or ('0_th_var' in list(entry.keys())):
            # we are dealing with one of the qualitative variables and it is the one that got striked or we are dealing with the zeroth

            change = min(np.random.poisson(1)), len(entry.get('ranks') - 1)  # with this we ensure that the chosen
            # thing is within range of the dictionary

            new = sorted(list(entry.get('ranks').items()), reverse=True)[change][0]
            entry.update({'current': new})

            # lastly we reset the strikes

            entry.update({
                'strikes': 10 * kappa
            })

    return list_o_entries


def new_targeting_FF(name):
    """ This function returns a dictionary containing the new_eta

    :param name:
    :return:
    """
    AGENTS_r, env = config.AGENTS_r, config.env

    target = AGENTS_r[name][env.now - 1]["dd_target"]['current']
    current = AGENTS_r[name][env.now - 1]['current_state']
    ambition = AGENTS_r[name][env.now - 1]['dd_ambition']['current']
    eta_acc = AGENTS_r[name][0]['dd_eta'][
        'current']  # this way we get the first accepted ETA, which is the duration of the target

    new_target = min(target, current) * ambition

    return {'new_eta': env.now + eta_acc, 'new_target': new_target}


def finding_FF(complete_dictionary, what,
               how='highest', cond_dict=None):
    """ this function returns the highest, lowest or the median value ('what') for any dictionary entry (MIX, TECHNOLOGIC, AGENTS, CONTRACTS) in the form of condiction dict = {'condition' : 'state of the condition'}

    :param complete_dictionary:
    :param what:
    :param how:
    :param cond_dict:
    :return:
    """
    if cond_dict is None:
        cond_dict = {}
    whats, whos, completes = [], [], []

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
                # ic(i.get(key).get(what.get(key)), i.get('name'))
                whats.append(i.get(key).get(what.get(key)))
                whos.append(i.get('name'))
                completes.append(i)

    if len(whats) == 0:
        # nothing was got, so we must put whats as zero and assign nothing
        whats = [0]
        whos = ['No one']
        completes = ['Nothing']

    if how == 'highest':
        idx = whats.index(max(whats))

    elif how == 'lowest':
        idx = whats.index(min(whats))

    elif how == 'median':
        idx = whats.index(median(whats))

    elif how == 'sum':
        whats = [sum(whats)]
        whos = ['twas a sum']
        idx = 0

    return {'name': whos[idx],
            'value': whats[idx],
            'complete_dict_entry': completes[idx]}


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
        denominator = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0} if public == False else {1: 0, 2: 0, 4: 0, 5: 0}
    else:
        denominator = {0: 0, 1: 0, 2: 0} if EorM == 'E' else {3: 0, 4: 0, 5: 0}
    var_dict = denominator.copy()  ### WE HAVE TO USE COPY, if not, we always divide by itself and get 1...

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
    """ If we are dealing with a public agent, then we need to get also the 12, 45 and 1245 entries in the dictionary """
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
            _)  # EDIT AUGUST 2021, we used to have this if to avoid zero pricing. Not necessary anymore, in fact we will need it. if var_dict.get(_) else STARTING_PRICE this is to avoid zero pricing for sources not used yet
        var_dict.update({_: j / i})

    return var_dict


def private_reporting_FF(genre,
                         EorM='No'):
    """ Internal function called by source_reporting_FF. this is the reporting_FF, it reports both the backwards and forwards for private agents. Technology producers don't report because they can't switch the source, but banks and energy providers yes. Backwards basically reports what happened

    :param genre:
    :param EorM:
    :return:
    """
    r, TECHNOLOGIC, AGENTS, DEMAND, env, AMMORT = config.r, config.TECHNOLOGIC, config.AGENTS, config.DEMAND, config.env, config.AMMORT

    backward_dict, forward_dict = {}, {}

    if EorM == 'No':
        Sources = [0, 1, 2, 3, 4, 5]

    else:
        Sources = [0, 1, 2] if EorM == 'E' else [3, 4, 5]

    for source in Sources:
        forward_dict.update({source: 0})
        chosen_TP = forward_dict.copy()  # only used for the BB

    """
    On to the forward_dict
    """

    for TP in TECHNOLOGIC.get(env.now - 1):
        # we have the highest MW in order to compare things, the maximum price
        technology = TECHNOLOGIC.get(env.now - 1).get(TP)
        demand_now = DEMAND.get(technology.get('EorM'))
        max_price = finding_FF(TECHNOLOGIC.get(env.now - 1), 'MW', 'highest', {'source': source})['value']
        max_price = finding_FF(TECHNOLOGIC.get(env.now - 1), 'MW', 'highest', {'EorM': EorM})[
            'value'] if max_price == 0 else max_price  # if there is no contracted capacity of the source, we attempt to the get the highest price for the eletrcitity or molecule part.
        lumps = np.floor(demand_now / technology['MW'])

        NPV = npv_generating_FF(r, technology.get('lifetime'), lumps, technology.get('MW'),
                                technology.get('building_time'), technology.get('CAPEX') * lumps,
                                technology.get('OPEX') * lumps, max_price, technology.get('CF'), AMMORT)

        if technology['source'] in Sources and NPV > forward_dict[technology['source']]:
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
                             {'genre': genre})['value'] if EorM == 'No' else \
            finding_FF(AGENTS.get(env.now - 1), specific_profits, 'sum', {'genre': genre, 'EorM': EorM})[
                'value']  # we now have the current profits for that source in that agent
        current += Sources[source]  # we now add to the current profits what we had
        backward_dict.update({source: current})  # and update the backward_dict dictionary

    return {"backward_dict": backward_dict, "forward_dict": forward_dict}


def public_reporting_FF(rationale):
    """
    Internal function called by source_reporting_FF. This is the reporting_FF, it reports both the backwards and forwards for public agents.
    :param rationale:
    :return:
    """
    r, TECHNOLOGIC, MIX, AGENTS, CONTRACTS, DEMAND, env = config.r, config.TECHNOLOGIC, config.MIX, config.AGENTS, config.CONTRACTS, config.DEMAND, config.env

    backward_dict = {}

    Sources = [1, 2, 4, 5]

    for source in Sources:
        backward_dict.update({source: 0})
    forward_dict = backward_dict.copy()

    """
    On to the forward_dict
    """

    # if the rationale is innovation or internalization, the policy maker basically analyzes agents: on the backward_dict for what happened and on the forward_dict for what could've happend. For the what could've it basically analyzes what would be the effects if all profits were reinvested. For emissions it is a little more complicated: it needs to get the avoided emissions for the backwards and get the whole demand for electricity or molecules and look what would be the avoided emissions if the best technology in each source was used.

    if rationale == 'green':
        # the rationale is green, so we need to look at the avoided emissions
        for TP in TECHNOLOGIC.get(env.now - 1):
            # we have the highest MW in order to compare things, the maximum price
            technology = TECHNOLOGIC.get(env.now - 1).get(TP)
            demand_now = DEMAND.get(technology.get('EorM'))
            lumps = np.floor(demand_now / technology['MW'])

            avoided_emissions = technology['avoided'] * lumps

            if technology['source'] in Sources and avoided_emissions > forward_dict[technology['source']]:
                forward_dict.update({technology['source']: avoided_emissions})

        for code in MIX.get(env.now - 1):
            plant = MIX.get(env.now - 1)
            if 'status' == 'contracted':
                # the plant was contracted at that period
                current = plant['avoided']  # we now have the current avoided emissions of that plant
                current += Sources[plant.get('source')]  # we now add to the current profits what we had
                backward_dict.update({plant.get('source'): current})  # and update the backward_dict dictionary

    else:
        # the rationale is either internalization (capacity) or innovation (R&D), so we have to check the agents dictionary instead
        for _ in AGENTS.get(env.now - 2):
            agent = AGENTS.get(env.now - 2).get(_)

            if agent.get('genre') == 'TP':
                profits = agent.get('profits')
                var = agent.get('RandD') if rationale == 'innovation' else agent.get(
                    'capacity')  # if the rationale is innovation we are looking into R&D, if not then the rationale is internalization and we are looking at the productive capacity
                # we now sum the two, because it would mean what would happen if all profits were reinvested into R&D
                """ and now to the backward_dict"""
                agent = AGENTS.get(env.now - 1).get(
                    _)  # we have to get the last information, we are looking at what actually happened
                current = agent['RandD'] if rationale == 'innovation' else agent.get('capacity')
                current += Sources[agent.get('source')]  # we now add to the current profits what we had
                backward_dict.update({agent.get('source'): current})  # and update the backward_dict dictionary

    return {"backward_dict": backward_dict, "forward_dict": forward_dict}


def sourcing_FF(backward_dict, forward_dict, index_dict, backwardness):
    """ Internal function called by source_reporting_FF. Transforms the results of the reporting function into the
     new_dd_source

    :param backward_dict:
    :param forward_dict:
    :param index_dict:
    :param backwardness:
    :return:
    """
    bwd, idx = backwardness, index_dict
    back, forw = backward_dict, forward_dict
    new_dd_source = {}

    for source in back:
        new_dd_source.update({
            source: bwd * back[source] + (1 - bwd) * forw[source]
        })

    return new_dd_source


def source_reporting_FF(name):
    """ This function produces the dd_source dict with the scores for sources according to the agent's characteristics.

    :param name:
    :return:
    """
    AGENTS_r = config.AGENTS_r
    agent = AGENTS_r[0][name]
    genre = agent['genre']

    if genre in ['EPM', 'TPM', 'DBB']:
        report = public_reporting_FF(agent['rationale'])

    else:
        report = private_reporting_FF(genre) if genre == 'BB' else private_reporting_FF(genre, agent['EorM'])

    new_dd_source = sourcing_FF(report['backward_dict'],
                                report['forward_dict'],
                                agent['index_per_source'],
                                agent['backwardness']['current'])

    return new_dd_source


def indexing_FF(name):
    AGENTS, env, CONTRACTS, INSTRUMENT_TO_SOURCE_DICT, MIX = config.AGENTS, config.env, config.CONTRACTS, config.INSTRUMENT_TO_SOURCE_DICT, config.MIX

    agent = AGENTS[env.now - 1][name]
    index = agent['index_per_source'].copy()

    for source in index:
        index[source] *= (1 - agent['dd_discount']['current'])

    if agent['genre'] == 'TPM':
        # we are dealing with the TPM, so it has to check the last round of incentives

        for contract in CONTRACTS[env.now - 1]:
            if 'sender' == 'TPM':
                # that contract was an incentive, so it goes into account

                index[contract['source']] += contract['value']

    elif agent['genre'] == 'EPM':

        # The EPM does something a little more complicated: carbon tax does not go into account; FiT are deploy-based and auctions are deploy-based
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
            index += agent['finance_index'][source] - AGENTS[env.now - 2][name]['finance_index'][source]

    return index


def thresholding_FF(threshold, disclosed_var, decision_var):
    """ This function checks if the treshold for changing the disclosed value was reached. If it was not, the disclosed value remains the same, if it was, then the disclosed value is changed to the current decision value. It returns

    :param threshold:
    :param disclosed_var:
    :param decision_var:
    :return:
    """
    threshold_upper, threshold_lower = disclosed_var * (1 + threshold), disclosed_var * (1 - threshold)

    if not threshold_lower < decision_var < threshold_upper:
        disclosed_var = decision_var

    return disclosed_var


def from_time_to_agents_FF(dictionary):
    """ This function gets the dictionaries that are created by time and arrange copys as other dictionaries reversed,
     i.e., time by agents

    :param dictionary:
    :return:
    """
    AGENTS, TECHNOLOGIC, AGENTS_r, TECHNOLOGIC_r, env = config.AGENTS, config.TECHNOLOGIC, config.AGENTS_r, config.TECHNOLOGIC_r, config.env

    for _ in dictionary.get(env.now):
        i = dictionary.get(env.now).get(_).copy()
        if dictionary == AGENTS:
            dictionary_r = AGENTS_r

        elif dictionary == TECHNOLOGIC:
            dictionary_r = TECHNOLOGIC_r

        else:
            print('not coded')

        dictionary_r.update({_: {env.now: i}})

    return


def evaluating_FF(name):
    """
    This function evaluates what happen to a certain agent at the previous period

    :param name:
    :return: dictionary as following {'action' : action, 'strikes' : strikes}
    """
    env, AGENTS_r = config.env, config.AGENTS_r

    add = {'EP': False,
           'TP': False,
           'BB': True,
           'EPM': True,
           'TPM': True,
           'DBB': True}
    # this controls if one agent can add or not

    change = {'EP': True,
              'TP': True,
              'BB': True,
              'EPM': True,
              'TPM': True,
              'DBB': True}
    # this controls if one agent can change or not

    # first we have to get the information of the agent from the specififed dictionary

    avg_time = AGENTS_r[name][env.now - 1]['dd_avg_time']['current']
    kappa = AGENTS_r[name][env.now - 1]['dd_kappas']['current']
    discount = AGENTS_r[name][env.now - 1]['dd_discount']['current']
    more_the_better = AGENTS_r[name][env.now - 1]['more_the_better']
    genre = AGENTS_r[name][env.now - 1]['genre']

    action = 'keep'
    strikes = [0, 0]

    if avg_time < env.now - 1:
        # if there is more time than the average time, then the agents do things and we can look into things now
        present, hist = 0, 0  # first we start these two numbers, they'll be used for the ratio
        if genre == 'DBB' or 'EPM' or 'TPM':
            # we are dealing with public agents

            eta_acc = AGENTS_r[name][env.now - 1]['dd_eta']['current']
            target = AGENTS_r[name][env.now - 1]['dd_target']['current']
            current = AGENTS_r[name][env.now - 1]['current_state']
            before = AGENTS_r[name][env.now - 1 - avg_time]['current_state']
            increase = (current - before) / before

            eta_exp = target / ((1 + increase) * current)

            present = eta_exp
            hist = eta_acc


        else:
            # we are dealing with private agents

            for period in AGENTS_r[name]:
                profit = (AGENTS_r[name][period]['profits'] * (1 - discount) ** (env.now - 1 - period))
                present += profit / avg_time if period in range(env.now - 1 - avg_time, env.now) else 0
                hist += profit / (env.now - 1)

        if present and hist == 0:
            ratio = 0  # this is to avoid division by zero

        ratio = min(present / hist, hist / present)  # to ensure that we get the percentage below 100
        dist = random.uniform(0, 1)

        if ((present > (1 + kappa) * hist) and more_the_better == True) or (
                (present < (1 - kappa) * hist) and more_the_better == False):
            # current is better than hist, so now we run the distribution
            strikes = [3, 1] if dist > ratio else [1, 0]
            if more_the_better == True:
                action = 'add' if add[genre] == True else 'keep'
            else:
                action = 'change' if change[genre] == True else 'keep'


        elif ((present > (1 + kappa) * hist) and more_the_better == False) or (
                (present < (1 - kappa) * hist) and more_the_better == True):
            # current is better than hist, so now we run the distribution
            strikes = [-3, -1] if dist > ratio else [-1, 0]
            if more_the_better == True:
                action = 'change' if change[genre] == True else 'keep'
            else:
                action = 'add' if add[genre] == True else 'keep'

    return {'action': action, 'strikes': strikes}


def post_evaluating_FF(strikes, name):
    AGENTS, env = config.AGENTS, config.env

    dd_qual_vars = AGENTS[env.now][name]['dd_qual_vars']

    if strikes != [0, 0]:
        # there was another strike

        for qual_var in dd_qual_vars:
            if qual_var > 0:
                entry = dd_qual_vars[qual_var]
                entry_value = AGENTS[env.now][name][entry]['strikes']

                AGENTS.update({entry: entry_value + strikes[qual_var]})

    return


def private_deciding_FF(name):
    AGENTS, BB_NAME_LIST, EP_NAME_LIST, TP_NAME_LIST, env = config.AGENTS, config.BB_NAME_LIST, config.EP_NAME_LIST, config.TP_NAME_LIST, config.env

    agent = AGENTS[env.now - 1][name].copy()
    avg_time = agent['dd_avg_time']['current']
    discount = agent['dd_discount']['current']
    kappa = agent['dd_kappas']['current']
    previous_var = agent['decision_var']

    if env.now - 1 - avg_time < 0:
        # we cannot search in negative periods
        start = 0
        end = env.now + avg_time  # we have to search unsimulated periods to get zero results and keep the ratio
    else:
        start = env.now - 1 - avg_time
        end = env.now

    genre = AGENTS[env.now - 1][name]["genre"]

    profits, medians = [], []

    for period in range(start, end):
        for i in ('highest', 'lowest'):
            profits.append(finding_FF(AGENTS[period], 'profits', i, {'genre': genre})['value'] * (
                    (1 - discount) ** (end - 1 - period)))
            medians.append(finding_FF(AGENTS[period], 'profits', 'median', {'genre': genre})['value'] * (
                    (1 - discount) ** (end - 1 - period)))

    ratio = (np.mean(medians) - AGENTS[end][name]["profits"]) / (max(profits - min(profits)))

    new_value = kappa * ratio + (1 - kappa) * previous_var

    return new_value


def public_deciding_FF(name):
    """ This function tells the ratio of 'effort' for the public agents as well as the current state of the analyzed
     variable

    :param name:
    :return:
    """
    AGENTS, AGENTS_r, MIX, env = config.AGENTS, config.AGENTS_r, config.MIX, config.env

    now = AGENTS_r[name][env.now - 1]

    rationale = now['dd_rationale']['current']
    avg_time = now['dd_avg_time']['current']
    discount = now['dd_discount']['current']
    target = now['dd_target']['current']
    eta_acc = now['dd_eta']['current']
    kappa = now['dd_kappa']['current']
    previous_var = now['decision_var']
    SorT = now['dd_SorT']['current']

    t_1, t_2 = AGENTS_r[name][env.now - 1 - avg_time], AGENTS_r[name][env.now - 1 - 2 * avg_time]

    ratio = previous_var

    if (2 * avg_time > env.now - 1 and t_1['dd_rationale']['current'] == t_2['dd_rationale']['current'] == rationale and
            t_1['dd_target']['current'] == t_2['dd_target']['current'] == target and t_1['dd_SorT']['current'] ==
            t_2['dd_SorT']['current']):
        # three conditions must be filled: we must have two avg_time cycles in order to compare then (normally 2 years), and that cycle must have the same rationale and the same target, otherwise the policy maker changed rationale or the target was chaned. If they are not filled, ratio remains as zero which tells the policy maker to keep doing whatever
        if rationale == 'green':
            # the policy maker wants less emissions
            dictionary = MIX
            restriction = {'status': 'contracted'}
            rationale = 'avoided'
        else:
            dictionary = AGENTS
            restriction = {'genre': 'TP'}
            rationale = 'RandD' if rationale == 'innovation' else 'capacity'
        results = []
        for period in range(env.now - 1 - avg_time, env.now - 1):
            current = finding_FF(dictionary.get(period), rationale, 'sum', restriction)['value']
            before = finding_FF(dictionary.get(period - avg_time), rationale, 'sum', restriction)['value']
            if SorT == 'T':
                increase = (current - before) / before
                eta_exp = target / ((1 + increase) * current)
                results.append(eta_exp)
            else:
                increase = (current - before)
                results.append(increase)

        if SorT == 'T':
            ratio = (1 - (eta_acc - env.now) / (max(eta_acc, np.mean(results)) - env.now)) ** kappa
        else:
            ratio = (np.mean(results) - results[-1]) / (max(results) - min(results))

    new_value = kappa * ratio + (1 - kappa) * previous_var

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
        dictionary = MIX
        restriction = {'status': 'contracted'}
        rationale = 'avoided'

    else:

        dictionary = AGENTS
        restriction = {'genre': 'TP'}
        rationale = 'RandD' if rationale == 'innovation' else 'capacity'

    current_state = finding_FF(dictionary.get(env.now - 1), rationale, 'sum', restriction)['value']

    return current_state


def npv_generating_FF(interest, time, lumps, MW, building_t, capex, opex, p, capacity_factor, ammort_t,
                      cash_flow_RISK=0, financing_RISK=0, true_capex_and_opex=False):
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

    CashFlow = []
    if true_capex_and_opex == False:
        caPEX = capex * lumps
        oPEX = opex * lumps
    else:
        caPEX = capex
        oPEX = opex

    Principal = (1 - financing_RISK) * caPEX * (1 + interest) ** building_t  # Present Value of the Principle
    Debt = Principal

    for t in range(time):

        inflow = ((p * MW * lumps * (24 * 30 * capacity_factor) - oPEX) * (1 - cash_flow_RISK))

        if Debt > 0:
            Debt *= (1 + interest)

            Fee = (Principal / (1 + ammort_t)) + Debt * interest

            Debt -= Fee

            inflow -= Fee

        CashFlow.append(inflow)

    NPV = [CashFlow[t] / ((1 + interest) ** t) for t in range(len(CashFlow))]

    return sum(NPV)


def policymaking_FF(dd, Policies, value,
                    add=False):
    """ This function adds a current policy to the pool of perenious policies as well as change the current policy.
     dd is the dictionary that changed, Policies is the list of current policies Now policies is the only output, which
      is the list of policies dictionary

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

    if add == True:
        # if we are adding a policy, we must first update its deadline and then add it to the
        Policy.update({
            'deadline': env.now + 1 + POLICY_EXPIRATION_DATE,
            'value': value
        })
        Policies.append(Policy.copy())

    """ now we check if what we changed was policy or incentivized source """

    if isinstance(dd['current'], int):
        # we only changed the source, so we keep everything else
        Policy.update({'source': dd['current']})

    else:
        # we changed something else, either instrument or the rationale

        if dd['current'] != ('green', 'innovation', 'capacity'):
            # we changed the instrument: there is no else here, because changing the rationale impacts the comparison between sources, specially for some time later

            """first, it the policy before it was an auction, we must drop the 'auction_countdown' and the 'auction_time' entries"""

            if 'auction_countdown' in Policy:
                # it was an auction
                Policy.pop('auction_countdown')
                Policy.pop('auction_time')

            Policy.update({'instrument': dd['current']})

            if Policy['current'] == 'auction':
                # we changed to an auction, so we must put an auction_countdown and and auction_time entries:

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


def capital_adequacy_rationing_FF(receivables_dict, risk_dict, wallet):
    """ This function produces the capital adequacy ratio according to the receivables, risk and the reserves (wallet) of a certain bank. car_ratio is a number

    :param receivables_dict:
    :param risk_dict:
    :param wallet:
    :return:
    """
    BASEL = config.BASEL
    recv_w_risk = {source: receivables_dict[source] * risk_dict[source] for source in receivables_dict}
    denominator = sum(recv_w_risk[source] for source in recv_w_risk)
    car_ratio = wallet / denominator if denominator > 0 else BASEL
    return car_ratio

def source_accepting_FF(accepted_sources, old):
    accepted_sources.update({
        old: True
    })

    return accepted_sources


def financing_FF(genre, target, name, my_wallet, my_receivables, value, financing_index,
                 guaranteeing=False):
    CONTRACTS, MIX, AGENTS, TECHNOLOGIC, r, BASEL, AMMORT, NPV_THRESHOLD, NPV_THRESHOLD_DBB, INSTRUMENT_TO_SOURCE_DICT, RISKS, env = config.CONTRACTS, config.MIX, config.AGENTS, config.TECHNOLOGIC, config.r, config.BASEL, config.AMMORT, config.NPV_THRESHOLD, config.NPV_THRESHOLD_DBB, config.INSTRUMENT_TO_SOURCE_DICT, config.RISKS, config.env

    new_wallet = my_wallet.copy()
    new_receivables = my_receivables.copy()

    tresHOLD = NPV_THRESHOLD if genre == 'BB' else NPV_THRESHOLD_DBB
    accepted_sources = target if genre == 'BB' else INSTRUMENT_TO_SOURCE_DICT.get(target)
    """ first the bank collects all the projects adressed to it """

    adressed_projects = {}
    adressed_projects_NPVs = []
    source_price = weighting_FF(env.now - 1, 'price', 'MWh', MIX)

    interest_r = r * (1 + value) if genre == 'BB' else r * (
            1 - value)  # if the agent is a private bank it increases the general interest rate by (1+risk), on the other hand, if the agent is the development bank it reduces the general interest rate by (1-effort)

    car_ratio = capital_adequacy_rationing_FF(new_receivables, RISKS, new_wallet)

    if env.now > 0 and len(CONTRACTS.get(env.now - 1)) > 0 and car_ratio >= BASEL:
        for _ in CONTRACTS.get(env.now - 1):
            i = CONTRACTS.get(env.now - 1).get(_)
            """ the bank only appends that project if its adressed to it, is a project and it would accept to finance such source"""
            if i.get('receiver') == name and i.get('status') == 'project':
                if genre == 'BB':
                    financing_risk = 0 if i.get('guarantee') == True else value
                else:
                    financing_risk = 0
                cash_flow_risk = 0 if i.get('auction_contracted') == True else value
                price = i.get('price') if i.get('auction_contracted') == True else source_price.get(i.get('source'))
                NPV = npv_generating_FF(interest_r, i.get('lifetime'), i.get('Lumps'), i.get('capacity'),
                                        i.get('building_time'), i.get('CAPEX'), i.get('OPEX'), price, i.get('CF'),
                                        AMMORT, cash_flow_risk, financing_risk, True)
                print(env.now, i.get('receiver'), i.get('TP'), NPV)
                if NPV > tresHOLD or env.now < 12:
                    adressed_projects_NPVs.append({'code': _, 'NPV': NPV})
                    adressed_projects.update({_: i.copy()})
                    print('project approved', i.copy())
        adressed_projects_NPVs = sorted(adressed_projects_NPVs, key=lambda x: x.get('NPV'), reverse=True)

    for project in adressed_projects_NPVs:
        i = project.get('code')
        j = adressed_projects.get(i)
        k = j.get('source')

        new_new_wallet = new_wallet - j.get('CAPEX')
        receiv_value = new_receivables.get(k) + j.get('CAPEX') * (
                (interest_r) ** j.get('building_time')) if guaranteeing == False else new_receivables.get(k)
        new_new_receivables = new_receivables.copy()
        new_new_receivables.update({k: receiv_value})
        car_ratio = capital_adequacy_rationing_FF(new_receivables, RISKS, new_wallet)

        if k in accepted_sources and new_new_wallet >= 0 and car_ratio >= BASEL:
            new_wallet = new_new_wallet
            new_receivables = new_new_receivables
            new_contract = j.copy()
            if guaranteeing == False:
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
                    'risk': value,
                    'principal': j.get('CAPEX') * ((interest_r) ** j.get('building_time'))})
            CONTRACTS.get(env.now).update({i: new_contract})
            # print(name, 'financed', new_contract)
        else:
            new_contract = j.copy()
            new_contract.update({'status': 'rejected'})
            CONTRACTS.get(env.now).update({i: new_contract})

    my_wallet = new_wallet
    my_receivables = new_receivables
    return {'wallet': my_wallet, 'receivables': my_receivables, 'financing_index': financing_index}


def dd_dict_generating_FF(current,
                          source=True, zero_to_start=True):
    """
    This function produces the dicts for the turtles of the system. zero_to_start is defaulted to True, if anything else then it must the whole dictionary, for example: {0 : 3, 0.5: 1, 1:0} for a kappa example
    """
    dd_dict_generated = {"current": current}

    if source and zero_to_start == True:
        rank_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    elif source == 'E' and zero_to_start == True:
        rank_dict = {0: 0, 1: 0, 2: 0}

    elif source == 'M' and zero_to_start == True:
        rank_dict = {3: 0, 4: 0, 5: 0}

    else:
        rank_dict = zero_to_start

    dd_dict_generated.update({'ranks': rank_dict})

    return dd_dict_generated
