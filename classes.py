class TP(object):
    def __init__(self, env):
        self.env = env
        self.name = STARTING_NAME  # name of the agent, is a string, normally something like TP_01
        self.genre = 'TP'  # genre, we do not use type, because type is a dedicated command of python, is also a string
        self.subgenre = STARTING_TECHNOLOGY.get(
            'source')  # subgenre or source, we used to use subgenre a lot, now it's kind of a legacy. Is a number, 1 is wind, 2 is solar, 4 is biomass and 5 is hydrogen. 0 and 3 are not used because they are the fossil options
        self.wallet = STARTING_WALLET  # wallet or reserves, or savings, etc. How much the agent has? is a number
        self.profits = 0  # profits of the agent, is a number
        self.capacity = STARTING_CAPACITY  # productive capacity of the agent, is a number used to determine the CAPEX of its technology
        self.RandD = 0  # how much money was put into R&D? Is a number
        self.EorM = STARTING_TECHNOLOGY.get(
            'EorM')  # does the agent use electricity or molecules? is a string (either 'E' or 'M') and is used a lot
        self.innovation_index = 0  # index of innovation. Kind of a legacy, was used to analyze innovation
        self.Technology = STARTING_TECHNOLOGY  # the technology dict. Is a dictionary with the characteristics of the technology
        self.self_NPV = {}  # The NPV of a unit of investment. Is a dictionary: e.g. self_NPV={'value' : 2000, 'MWh' : 30}
        self.RnD_threshold = STARTING_RnD_THRESHOLD  # what is the threshold of R&D expenditure necessary to innovate? Is a number
        self.capacity_threshold = STARTING_CAPACITY_THRESHOLD  # what is the threshold of investment in productive capacity in order to start decreasing CAPEX costs?
        self.dd_profits = {0: 0, 1: 0, 2: 0} if STARTING_TECHNOLOGY.get('EorM') == 'E' else {3: 0, 4: 0,
                                                                                             5: 0}  # same as profits, but as dict. Makes accounting faster and simpler
        self.dd_source = STARTING_DD_SOURCE  # This, my ganzirosis, used to be the Tactics. It is the first of the ranked dictionaries. It goes a little sumthing like dis: dd = {'current' : 2, 'ranks' : {0: 3500, 1: 720, 2: 8000}}. With that we have the current decision for the variable or thing and on the ranks we have the score for
        self.decision_var = STARTING_DECISION_VAR  # this is the value of the decision variable. Is a number between -1 and 1
        self.action = "keep"  # this is the action variable. It can be either 'keep', 'change' or 'add'
        self.dd_responsiveness = STARTING_RESPONSIVENESS  # this is the responsiveness, follows the current ranked dictionary
        self.dd_qual_vars = STARTING_QUAL_VARS  # this tells the agent the qualitative variables in a form {0 : 'name of the zeroth variable', 1 : 'name of the first variable', 2 : 'name of the second variable'}
        self.dd_backwardness = STARTING_BACKWARDNESS  # also a ranked dictionary, this one tells the backwardness of agents
        self.dd_avg_time = STARTING_DD_AVG_TIME  # also a ranked dictionary, this one tells the average time for deciding if change is necessary
        self.dd_discount = STARTING_DISCOUNT  # discount factor. Is a ranked dictionary
        self.cap_conditions = STARTING_CAP_CONDITIONS  # there are the cap conditions for this technology, being a dictionary following this example {'char' : 'CAPEX', 'cap' : 20000, 'increase' : 0.5}
        self.capped = False  # boolean variable to make the capping easier
        self.strategy = STARTING_STRATEGY  # the initial strategy of the agent, can be to reivnest into producetive capacity or R&D
        self.dd_strategies = STARTING_STRATEGIES  # initial strategy for the technology provider. Is a ranked dictionary

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
            self.dd_responsiveness,
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
               dd_responsiveness,
               dd_qual_vars,
               dd_backwardness,
               dd_avg_time,
               dd_discount,
               cap_conditions,
               capped,
               dd_strategies):

        global CONTRACTS
        global MIX
        global AGENTS
        global AGENTS_r
        global TECHNOLOGIC
        global TECHNOLOGIC_r
        global r
        global TACTIC_DISCOUNT
        global AMMORT
        global rNd_INCREASE
        global RADICAL_THRESHOLD

        while True:

            #################################################################
            #                                                               #
            #     Before anything, we must the current values of each of    #
            #        the dictionaries that we use and other variables       #
            #                                                               #
            #################################################################

            list_of_strikeables = [dd_responsiveness, dd_qual_vars, dd_backwardness, dd_avg_time, dd_discount,
                                   dd_strategies]

            source = dd_source.get('current')
            responsiveness = dd_responsiveness.get('current')
            qual_vars = dd_qual_vars.get('current')
            backwardness = dd_backwardness.get('current')
            avg_time = dd_avg_time.get('current')
            discount = dd_discount.get('current')
            strategy = dd_strategies.get('current')

            value = decision_var
            profits = 0  # in order to get the profits of this period alone

            #################################################################
            #                                                               #
            #    First, the Technology provider closes any new deals and    #
            #                        collect profits                        #
            #                                                               #
            #################################################################

            if env.now > 0:
                for _ in CONTRACTS.get(env.now - 1):
                    i = CONTRACTS.get(env.now - 1).get(_)
                    if i.get('receiver') == name and i.get('status') == 'payment':
                        wallet += i.get('value')
                        profits += i.get('value')
                        """ we also have to update the sales_MWh entry, to indicate to the policy makers how much MWh of each source is there  """
                        j = dd_profits.get('ranks')
                        j.update({source: j.get(source) + i.get('value')})

            #################################################################
            #                                                               #
            #    Now, on to check if change is on and if there is a strike  #
            #                                                               #
            #################################################################

            if env.now > 0 and (action == 'add' or 'change'):
                striked = striking_FF(list_of_strikeables)

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
                for _ in CONTRACTS.get(env.now - 1):
                    i = CONTRACTS.get(env.now - 1).get(_)
                    if i.get('receiver') == name and i.get('sender') == 'TPM':
                        wallet += i.get('value')

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

            """ 1)  we have to get the base-CAPEX and adjust it to the productive capacity of the TP (only if its technology is not transportable) """
            if Technology.get('transport') == False:
                """ if the technology is not transportable, then the productive capacity impacts on the CAPEX """
                if env.now == 0:
                    """ if we are in the first period, then we have to get the starting CAPEX and tell the TP that this is his base capex, because there is no base_capex already """
                    i = Technology.get('CAPEX')
                    Technology.update({"base_CAPEX": i})
                j = Technology.get('base_CAPEX')
                """ we have to produce the actual CAPEX, with is the base_CAPEX multiplied by euler's number to the power of the ratio of how many times the base capex is greater than the capacity itself multiplied by the threshold of capacity"""
                new_capex = min(j, (capacity_threshold / capacity) * j)
                Technology.update({
                    'CAPEX': j * new_capex
                })
            else:
                """ the technology is transportable (e.g. solar panels)"""
                i = Technology.get('CAPEX')
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
                """ then we get the 'a' which can either be a poisson + normal for innovation, or a simple binomial. values above or equal (for the imitation) 1 indicate that innovation or imitation occured """
                a = np.random.poisson(1) + np.random.normal(0, 1)

                if a >= 1:
                    """ we are dealing with innovation """
                    RnD_threshold *= rNd_INCREASE * a
                    """ we have to check where did the innovation occur"""
                    what_on = random.choice(['base_CAPEX', 'OPEX', 'MW'])
                    """ if innovation ocurred then we multiply it"""
                    if what_on == 'MW':
                        new_what_on = a * Technology.get(what_on)
                    else:
                        new_what_on = (1 / a) * Technology.get(what_on)
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
                i = TECHNOLOGIC.get(env.now - 1).get(name)
                price = weighting_FF(env.now - 1, 'price', 'MWh', MIX, EorM=EorM)
                interest_r = weighting_FF(env.now - 1, 'interest_r', 'MWh', MIX, EorM=EorM, discount=discount)
                self_NPV.update({'value': npv_generating_FF(r, i.get('lifetime'), 1, i.get('MW'),
                                                            i.get('building_time'), i.get('CAPEX'), i.get('OPEX'),
                                                            price.get(i.get('source')), i.get('CF'), AMMORT),
                                 'MWh': i.get('MW')
                                 })

            """ 8) we must also check if the capping process is on"""

            if len(MIX.get(env.now - 1)):
                # if there is no capping, we must first make sure that it has not started
                now = finding_FF(MIX.get(env.now - 1), 'MW', 'sum', {'EP': name})
                if now > cap_conditions['cap']:
                    capped == True

                else:
                    capped == False

            if capped == True:
                # capping process is on, so we have to make sure that the capacity increased
                previous = finding_FF(MIX.get(env.now - 2), 'MW', 'sum', {'EP': name})

                if now > previous:
                    # the capacity increased, so we have to apply the capping conditions
                    Technology.update({
                        cap_conditions['char']: Technology.get(cap_conditions['char']) * (
                                    1 + cap_conditions['increase'])
                    })

            """5) we also have to update the TECHNOLOGIC dictionary for the next period with the current technology (applying any changes if there were any whatsover)"""
            TECHNOLOGIC.get(env.now).update({name: Technology})

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

            AGENTS.get(env.now).update({name: {
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
                "dd_responsiveness": dd_responsiveness,
                "dd_qual_vars": dd_qual_vars,
                "dd_backwardness": dd_backwardness,
                "dd_avg_time": dd_avg_time,
                "dd_discount": dd_discount,
                "cap_conditions": cap_conditions,
                "capped": capped,
            }})

            profits_dedicting_FF(name)
            post_evaluating_FF(decisions['strikes'], name)

            yield env.timeout(1)


class TPM(object):
    def __init__(self, env):
        self.env = env
        self.genre = 'TPM'
        self.subgenre = 'TPM'
        self.name = 'TPM'
        self.wallet = STARTING_WALLET
        self.dd_policy = STARTING_DD_POLICY
        self.dd_source = STARTING_DD_SOURCE
        self.decision_var = STARTING_DECISION_VAR
        self.disclosed_var = STARTING_DECISION_VAR
        self.action = 'keep'
        self.dd_responsiveness = STARTING_DD_RESPONSIVENESS
        self.dd_qual_vars = STARTING_DD_QUAL_VARS
        self.dd_backwardness = STARTING_DD_BACKWARDNESS
        self.dd_avg_time = STARTING_DD_AVG_TIME
        self.dd_discount = STARTING_DD_DISCOUNT
        self.dd_policy = STARTING_DD_POLICY
        self.policies = STARTING_POLICIES
        self.dd_index = STARTING_DD_INDEX
        self.index_per_source = {1: 0, 2: 0, 4: 0, 5: 0}
        self.dd_eta = STARTING_DD_ETA
        self.dd_ambition = STARTING_DD_AMBITION
        self.dd_target = STARTING_DD_TARGET
        self.dd_rationale = STARTING_DD_RATIONALE

        self.action = env.process(self.run_TPM(self.genre,
                                               self.subgenre,
                                               self.name,
                                               self.wallet,
                                               self.dd_policy,
                                               self.dd_source,
                                               self.decision_var,
                                               self.disclosed_var,
                                               self.action,
                                               self.dd_responsiveness,
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
                                               self.dd_rationale))

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
                dd_responsiveness,
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
                dd_rationale):

        global CONTRACTS
        global MIX
        global AGENTS
        global TECHNOLOGIC
        global r
        global POLICY_EXPIRATION_DATE
        global AMMORT
        global TACTIC_DISCOUNT
        global INSTRUMENT_TO_SOURCE_DICT

        while True:

            #################################################################
            #                                                               #
            #              First, the TPM checks if it got an               #
            #            strike, adds or changes its main policy            #
            #                                                               #
            #################################################################

            list_of_strikeables = [dd_policy, dd_source, dd_responsiveness, dd_qual_vars, dd_backwardness, dd_avg_time,
                                   dd_discount, dd_policy, dd_index, dd_eta, dd_ambition, dd_target, dd_rationale]

            policy = dd_policy['current']
            source = dd_source['current']
            responsiveness = dd_responsiveness['current']
            backwardness = dd_backwardness['current']
            avg_time = dd_avg_time['current']
            discount = dd_discount['current']
            index = indexing_FF('TPM') if env.now > 0 else dd_index['current']
            eta_acc = dd_eta['current']
            ambition = dd_ambition['current']
            rationale = dd_rationale['current']
            value = disclosed_var

            if env.now > 0 and (action == 'add' or 'change'):
                striked = striking_FF(list_of_strikeables)  # with this we have a different list of strikeables

                for entry in range(0, len(list_of_strikeables)):

                    if list_of_strikeables[entry]['current'] == striked[entry]:
                        # that dictionary was not the changed one, so we can just update it
                        list_of_strikeables[entry] = striked[entry]

                    else:
                        # alright, that was the one that changed

                        Policies = policymaking_FF(dd, Policies,
                                                   disclosed_var) if action == 'change' else policymaking_FF(dd,
                                                                                                             Policies,
                                                                                                             disclosed_var,
                                                                                                             add=True)

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
                    instrument = entry.get('instrument')
                    source = entry.get('source')
                    budget = entry.get('budget') if 'budget' in entry else value * wallet
                    value = disclosed_var if 'value' not in entry else entry.get('value')

                if instrument == 'supply':

                    firms = []
                    for _ in AGENTS.get(env.now - 1):
                        i = AGENTS.get(env.now - 1).get(_)
                        if i.get('genre') == 'TP' and (i.get('source') in INSTRUMENT_TO_SOURCE_DICT.get(source)):
                            firms.append(_)

                    if len(firms) > 0:
                        """ we have to be certain that there are companies to be inbcentivised and now divides the possible incentive by the number of companies """
                        # print('incentivised_firms', incentivised_firms)
                        incentive = budget / len(firms)

                        """ and now we give out the incentives"""
                        for TP in incentivised_firms:
                            code = uuid.uuid4().int
                            CONTRACTS.get(env.now).update({
                                code: {
                                    'sender': name,
                                    'receiver': TP,
                                    'status': 'payment',
                                    'value': incentive}})
                else:
                    """
                    demmand-side incentives
                    """
                    print('TBD')

                """ and now back to the actual variables for the current policy"""
                instrument = Policies[0]['instrument']
                source = Policies[0]['source']
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
                disclosed_var = thresholding_FF(responsivity, disclosed_var, decision_var)

            #################################################################
            #                                                               #
            #    Before leaving, the agent must uptade the outside world    #
            #                                                               #
            #################################################################

            AGENTS.get(env.now).update({
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
                    "dd_responsiveness": dd_responsiveness,
                    "dd_qual_vars": dd_qual_vars,
                    "dd_backwardness": dd_backwardness,
                    "dd_avg_time": dd_avg_time,
                    "dd_discount": dd_discount,
                    "dd_policy": dd_policy,
                    "policies": policies,
                    "dd_index": dd_index,
                    "index_per_source": index_per_source,
                    "dd_eta": dd_eta,
                    "dd_ambition": dd_ambition,
                    "dd_target": dd_target,
                    "dd_rationale": dd_rationale,
                    "decision_var": decision_var,
                    "disclosed_var": disclosed_var,
                    "policy": policy,
                    "source": source,
                    "responsiveness": responsiveness,
                    "backwardness": backwardness,
                    "avg_time": avg_time,
                    "discount": discount,
                    "index": index,
                    "eta_acc": eta_acc,
                    "ambition": ambition,
                    "rationale": rationale,
                    "value": value,
                }})

            post_evaluating_FF(decisions['strikes'], name)

            yield env.timeout(1)


def Create(genre, traits):
    # PUBLIC AGENTS
    global SIM_TIME
    global STARTING_PPA_EXPIRATION
    global STARTING_PPA_LIMIT
    global STARTING_COUNTDOWN
    global STARTING_WALLET
    global STARTING_META_STRATEGY
    global STARTING_RIVAT
    global STARTING_POLICIES
    global STARTING_RANKS
    global STARTING_RANKS
    global STARTING_TACTICS
    global STARTING_NPV_THRESHOLD_DBB
    global STARTING_FINANCING_INDEX

    # PRIVATE AGENTS
    global STARTING_TECHNOLOGY
    global STARTING_NAME
    global STARTING_WALLET
    global STARTING_CAPACITY
    global STARTING_TACTICS
    global STARTING_STRATEGY
    global STARTING_RnD_THRESHOLD
    global STARTING_CAPACITY_THRESHOLD
    global STARTING_SUBGENRE
    global STARTING_eORm
    global STARTING_MW_dict
    global INITIAL_DEMAND
    global DEMAND_SPECIFICITIES
    global STARTING_RATIONALE
    global STARTING_PORTFOLIO
    global ACCEPTED_SOURCES
    global NUMBER_OF_TP_DICT
    global NPV_THRESHOLD_DBB
    global EP_NAME_LIST
    global TP_NAME_LIST
    global BB_NAME_LIST

    if env.now > 0:
        traits = AGENTS.get(env.now).get(traits)

    STARTING_WALLET = traits.get('wallet')  # everyone has a wallet

    if genre in ('EPM', 'TPM', 'DBB'):
        """ common things to all policy makers """
        policy_maker_tactics = {
            'innovation': {1: 0, 2: 0,
                           4: 0, 5: 0,
                           12: 0,
                           45: 0,
                           1245: 0},
            'capacity': {1: 0, 2: 0,
                         4: 0, 5: 0,
                         12: 0,
                         45: 0,
                         1245: 0},
            'expansion': {1: 0, 2: 0,
                          4: 0, 5: 0,
                          12: 0,
                          45: 0,
                          1245: 0}}

        STARTING_META_STRATEGY = {0:
                                      {0: traits.get('meta_strategy').get('first_to_change'),
                                       'strikes': 0,
                                       'max': traits.get('meta_strategy').get('max_strikes_for_the_first_option')}, 1:
                                      {1: traits.get('meta_strategy').get('second_to_change'),
                                       'strikes': 0,
                                       'max': traits.get('meta_strategy').get('max_strikes_for_the_second_option')}, 2:
                                      {2: traits.get('meta_strategy').get('third_to_change'),
                                       'strikes': 0,
                                       'max': traits.get('meta_strategy').get('max_strikes_for_the_third_option')}}
        STARTING_RIVAT = [traits.get('RIVAT').get('Rationale'),
                          traits.get('RIVAT').get('Instrument'),
                          traits.get('RIVAT').get('Value'),
                          traits.get('RIVAT').get('Action'),
                          traits.get('RIVAT').get('Target')]
        STARTING_POLICIES = []
        STARTING_RANKS = {'R':
                              {'innovation': traits.get('ranks').get('rationale').get('innovation'),
                               'expansion': traits.get('ranks').get('rationale').get('expansion'),
                               'capacity': traits.get('ranks').get('rationale').get('capacity')},
                          'I': {}}
        for instrument_in_quotes in traits.get('ranks').get('instrument'):
            instrument = traits.get('ranks').get('instrument').get(instrument_in_quotes)
            STARTING_RANKS.get('I').update({
                instrument_in_quotes: instrument
            })
        STARTING_TACTICS = {}
        for i in range(SIM_TIME):
            STARTING_TACTICS.update({i: policy_maker_tactics})
        if env.now > 0:
            STARTING_TACTICS = traits.get('Tactics')

        if genre == 'TPM':

            #################################################################
            #                                                               #
            #                    technology policy maker                    #
            #                                                               #
            #################################################################
            technology_policy_maker = TPM(env)

        elif genre == 'EPM':

            #################################################################
            #                                                               #
            #                      energy policy maker                      #
            #                                                               #
            #################################################################
            STARTING_PPA_EXPIRATION = traits.get('PPA_expiration')  # for how many months do PPA contracts stand?
            STARTING_PPA_LIMIT = traits.get('PPA_limit')  # how many months do companies have to build auction
            STARTING_COUNTDOWN = traits.get(
                'auction_countdown')  # how many months are spent between annoucing and deciding an auction?
            energy_policy_maker = EPM(env)

        else:

            #################################################################
            #                                                               #
            #                       Development Bank                        #
            #                                                               #
            #################################################################
            STARTING_NPV_THRESHOLD_DBB = traits.get('NPV_threshold_for_the_DBB')
            NPV_THRESHOLD_DBB = STARTING_NPV_THRESHOLD_DBB  # it must be global for the finance function
            STARTING_FINANCING_INDEX = {}
            for Source in traits.get('financing_index'):
                financing_index = traits.get('financing_index').get(Source)
                STARTING_FINANCING_INDEX.update({Source: financing_index})
            Development_Bank = DBB(env)
    else:
        """ so we are dealing with private agents """
        if genre == 'BB':

            #################################################################
            #                                                               #
            #                         Private Bank                          #
            #                                                               #
            #################################################################

            STARTING_SUBGENRE = traits.get('subgenre')
            STARTING_NAME = traits.get('name')
            BB_NAME_LIST.append(STARTING_NAME)
            STARTING_STRATEGY = traits.get('strategy')
            STARTING_PORTFOLIO = traits.get('portfolio')
            ACCEPTED_SOURCES = traits.get('accepted_sources')
            STARTING_RATIONALE = traits.get('rationale')

            STARTING_TACTICS = {}
            STARTING_TACTICS_LIST = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for j in range(SIM_TIME):
                for source in list(STARTING_TACTICS_LIST.keys()):
                    STARTING_TACTICS.update({j: STARTING_TACTICS_LIST})

            if env.now > 0:
                STARTING_TACTICS = traits.get('Tactics')

            STARTING_NAME = BB(env)

        elif genre == 'EP':

            #################################################################
            #                                                               #
            #                       Energy Producer                         #
            #                                                               #
            #################################################################

            STARTING_STRATEGY = traits.get('strategy')
            STARTING_eORm = 'E' if STARTING_STRATEGY[0] in [0, 1, 2] else 'M'
            STARTING_SUBGENRE = STARTING_eORm
            STARTING_NAME = traits.get('name')
            EP_NAME_LIST.append(STARTING_NAME)
            STARTING_PERIODICITY = traits.get('periodicity')
            STARTING_TOLERANCE = traits.get('tolerance')
            STARTING_PORTFOLIO = []
            STARTING_TACTICS = {}
            STARTING_TACTICS_LIST = {0: 0, 1: 0, 2: 0} if STARTING_STRATEGY[0] in [0, 1, 2] else {3: 0, 4: 0, 5: 0}
            STARTING_MW_dict = {}
            for t in range(SIM_TIME):
                for source in list(STARTING_TACTICS_LIST.keys()):
                    STARTING_TACTICS.update({t: STARTING_TACTICS_LIST})
                STARTING_MW_dict.update({t: 0})

            if env.now > 0:
                STARTING_TACTICS = traits.get('Tactics')

            Energy_Producer = EP(env)

        else:

            #################################################################
            #                                                               #
            #                     Technology Provider                       #
            #                                                               #
            #################################################################

            STARTING_NAME = traits.get('name')
            TP_NAME_LIST.append(STARTING_NAME)

            Tech_specs_dict = {1: {'subgenre': 1,
                                   "transport": False,
                                   "source_name": 'wind',
                                   'EorM': 'E'
                                   },
                               2: {'subgenre': 2,
                                   "transport": True,
                                   "source_name": 'solar',
                                   'EorM': 'E'
                                   },
                               4: {'subgenre': 4,
                                   "transport": False,
                                   "source_name": 'biogas',
                                   'EorM': 'M'
                                   },
                               5: {'subgenre': 5,
                                   "transport": False,
                                   "source_name": 'hydrogen',
                                   'EorM': 'M'
                                   }}
            traits.get('technology').update({
                'last_radical_innovation': 0,
                'last_marginal_innovation': 0,
                'green': True})
            traits.get('technology').update(
                Tech_specs_dict.get(traits.get('technology').get('source'))
            )

            STARTING_TECHNOLOGY = traits.get('technology')
            STARTING_TECHNOLOGY.update({'name': STARTING_NAME})

            STARTING_SUBGENRE = traits.get('technology').get('subgenre')
            NUMBER_OF_TP_DICT.update({
                STARTING_SUBGENRE: NUMBER_OF_TP_DICT.get(STARTING_SUBGENRE) + 1
            })
            STARTING_CAPACITY = traits.get('wallet')
            EorM = traits.get('technology').get('EorM')
            STARTING_STRATEGY = traits.get('strategy')
            if EorM == 'E':
                tactical = (0, 1, 2)
            else:
                tactical = (3, 4, 5)

            STARTING_TACTICS = {}
            for t in range(SIM_TIME):
                STARTING_TACTICS.update({
                    t: {tactical[0]: 0, tactical[1]: 0, tactical[2]: 0}
                })
            STARTING_RnD_THRESHOLD = traits.get('RnD_threshold')
            STARTING_CAPACITY_THRESHOLD = traits.get('capacity_threshold')

            if env.now > 0:
                STARTING_TACTICS = traits.get('Tactics')

            STARTING_NAME = TP(env)

    return print('genre :', genre, ', traits:', traits, '. \n')


