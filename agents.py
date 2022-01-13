#################################################################
#                                                               #
#                  This is the list of agents:                  #
#                    First, the public agents                   #
#                                                               #
#################################################################

_agents = {'DBB' : {"wallet" : 20*10**5,
                    "dd_policy" : {'current' : 'finance',
                                   'ranks' : {'finance': 1, 'guarantee': 0}},
                    "dd_source" : {'current' : 1,
                                   'ranks' : {1: 0, 2: 0, 12:0, 3:0, 4:0, 45 : 0, 1245:0}},
                    "decision_var" : 0.5,
                    "dd_kappas" : {'current' : 0.5,
                                   'ranks' : {0.2: 2, 0.8: 1, 0.5 : 0}},
                    "dd_qual_vars" : dd_qual_vars,
                    "dd_backwardness" : {'current' : 0.5,
                                   'ranks' : {0.2: 2, 0.8: 1, 0.5 : 0}},
                    "dd_avg_time" : {'current' : 12,
                                   'ranks' : {12: 2, 24: 1, 6: 3}},
                    "dd_discount" : {'current' : 0.01,
                                   'ranks' : {0.01: 2, 0.1: 1, 0.05: 3},
                    "policies" : policies,
                    "dd_index" : {'current' : 0.5,
                                   'ranks' : {0.2: 2, 0.8: 1, 0.5 : 0}},
                    "dd_eta" : dd_eta,
                    "dd_ambition" : dd_ambition,
                    "dd_target" : dd_target,
                    "dd_rationale" : dd_rationale,
                    "Portfolio" : Portfolio,
                    "accepted_sources" : accepted_sources,
                    "dd_SorT" : dd_SorT
                    },
           'EPM' : {"wallet": wallet,
                    "dd_source": dd_source,
                    "decision_var": 0.5,
                    "dd_kappas": dd_kappas,
                    "dd_qual_vars": dd_qual_vars,
                    "dd_backwardness": dd_backwardness,
                    "dd_avg_time": dd_avg_time,
                    "dd_discount": dd_discount,
                    "dd_policy": dd_policy,
                    "dd_index": policies,
                    "policies": dd_index,
                    "dd_eta": dd_eta,
                    "dd_ambition": dd_ambition,
                    "dd_target": dd_target,
                    "dd_rationale": dd_rationale,
                    "dd_SorT": dd_SorT
                    },
           'TPM' : {"wallet": wallet,
                    "dd_source": dd_source,
                    "decision_var": 0.5,
                    "dd_kappas": dd_kappas,
                    "dd_qual_vars": dd_qual_vars,
                    "dd_backwardness": dd_backwardness,
                    "dd_avg_time": dd_avg_time,
                    "dd_discount": dd_discount,
                    "dd_policy": dd_policy,
                    "policies": policies,
                    "dd_index": dd_index,
                    "dd_eta": dd_eta,
                    "dd_ambition": dd_ambition,
                    "dd_target": dd_target,
                    "dd_rationale": dd_rationale,
                    "dd_SorT": dd_SorT
                    },
           'TPs' : {'wind0': {name, wallet, capacity, Technology, RnD_threshold, capacity_threshold, dd_source,
                 decision_var, dd_kappas, dd_qual_vars, dd_backwardness, dd_avg_time, dd_discount, cap_conditions,
                 strategy, dd_strategies},
                    'wind1': {},
                    'solar0': {},
                    'solar1': {},
                    'biomass0': {},
                    'biomass1': {},
                    'hydrogen0': {},
                    'hydrogen1':{}
                    },
           'EPs' : {'E0': {accepted_sources, name, wallet, EorM, portfolio_of_plants, portfolio_of_projects,
                 periodicity, tolerance, last_acquisition_period, dd_source, decision_var, dd_kappas, dd_qual_vars,
                 dd_backwardness, dd_avg_time, dd_discount, dd_strategies, dd_index},
                    'E1': {},
                    'E2': {},
                    'M0': {},
                    'M1': {},
                    'M2': {}
                    },
           'BBs': {'0': {Portfolio, accepted_sources, name, wallet, dd_source, decision_var, dd_kappas, dd_qual_vars,
                 dd_backwardness, dd_avg_time, dd_discount, dd_strategies, dd_index},
                   '1': {},
                   '2': {},
                   '3': {}
                   }
           }


"SENDER" : "technology policy maker"
"receiver":
"value":
