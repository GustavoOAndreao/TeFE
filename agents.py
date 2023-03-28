#################################################################
#                                                               #
#                  This is the list of agents:                  #
#                    First, the public agents                   #
#                                                               #
#################################################################

from config import *
from classes import *

TP_0 = TP(env=env,
          name= 'TP_0',
          wallet= 20*10**5,
          capacity= 20*10**3,
          Technology= {'name': 'TP_wind',
                       "green": True,
                       "source": 1,
                       "source_name": 'solar',
                       "CAPEX": config.WIND['CAPEX'],
                       'OPEX': config.WIND['OPEX'],
                       "dispatchable": True,
                       "transport": True,
                       "CF": config.WIND['CF'],
                       "MW": config.WIND['MW'],
                       "lifetime": config.WIND['lifetime'],
                       'building_time': config.WIND['building_time'],
                       'last_radical_innovation': 0,
                       'last_marginal_innovation': 0,
                       'emissions': 0,
                       'avoided_emissions': 250},
          RnD_threshold= 4,
          capacity_threshold= 3,
          decision_var= 0.5,
          cap_conditions= {},
          impatience= [10, 25, 30],
          past_weight= [0.75],
          LSS_thresh= [0.25, 0.1],
          memory= [12, 6],
          discount= [0.02],
          strategy= ['capacity', "innovation"])

TP_1 = TP(env= env,
          name= 'TP_1',
          wallet= 30*10**5,
          capacity= 20000,
          Technology= {'name': 'TP_1',
                       "green": True,
                       "source": 2,
                       "source_name": 'solar',
                       "CAPEX": config.SOLAR['CAPEX'],
                       'OPEX': config.SOLAR['OPEX'],
                       "dispatchable": True,
                       "transport": True,
                       "CF": config.SOLAR['CF'],
                       "MW": config.SOLAR['MW'],
                       "lifetime": config.SOLAR['lifetime'],
                       'building_time': config.SOLAR['building_time'],
                       'last_radical_innovation': 0,
                       'last_marginal_innovation': 0,
                       'emissions': 0,
                       'avoided_emissions': 50},
          RnD_threshold= 4,
          capacity_threshold= 3,
          decision_var= 0.7,
          cap_conditions= {},
          impatience= [10, 30],
          past_weight= [0.4],
          LSS_thresh= [0.5, 0.75],
          memory= [12, 24],
          discount= [0.001, 0.1],
          strategy= ['innovation']
          )

EP_0 = EP(env=env,
          name='EP_0',
          wallet=150000000,
          portfolio_of_plants={},
          portfolio_of_projects={},
          periodicity=6,
          tolerance=[12, 6, 24],
          last_acquisition_period=0,
          source=[{0: 1000}, {1: 500}, {2: 0}],
          decision_var=0.25,
          LSS_thresh=[0.25, 0.1],
          impatience=[10],
          memory=[12],
          discount=[.0001],
          past_weight=[0.5],
          current_weight=[0.25])

config.EP_NAME_LIST.append('EP_0')

EP_1 = EP(env=env,
          name='EP_1',
          wallet=150000000,
          portfolio_of_plants={},
          portfolio_of_projects={},
          periodicity=6,
          tolerance=[12],
          last_acquisition_period=0,
          source=[{2: 1000}, {1: 500}, {0: 0}],
          decision_var=0.25,
          LSS_thresh=[0.25],
          impatience=[5],
          memory=[12],
          discount=[.0001],
          past_weight=[0.5],
          current_weight=[0.25])

config.EP_NAME_LIST.append('EP_1')

EP_2 = EP(env=env,
          name='EP_2',
          wallet=150000000,
          portfolio_of_plants={},
          portfolio_of_projects={},
          periodicity=6,
          tolerance=[12],
          last_acquisition_period=0,
          source=[{2: 100}, {1: 50}, {0: 0}],
          decision_var=0.75,
          LSS_thresh=[1],
          impatience=[5],
          memory=[12],
          discount=[.001],
          past_weight=[1, 0.5],
          current_weight=[0.25])

config.EP_NAME_LIST.append('EP_2')

DBB = DBB(env=env,
          wallet=20 * 10 ** 12,
          instrument=['finance'],
          source=[{2: 500}, {1: 1000}, {0: 0}],
          decision_var=0.5,
          LSS_thresh=[.1, .5, .75],
          past_weight=[0.8],
          memory=[12],
          discount=[0.001],
          policies=[],
          impatience=[20],
          disclosed_thresh=[0.2],
          rationale=['green'])

DD = Demand(env=env,
            initial_demand=INITIAL_DEMAND,
            when=60,
            increase=INITIAL_DEMAND * 0.5)
