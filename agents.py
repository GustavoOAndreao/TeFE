#################################################################
#                                                               #
#                  This is the list of agents:                  #
#                    First, the public agents                   #
#                                                               #
#################################################################

from config import *
from classes import *

EP_0 = EP(env=env,
          name='EP_0',
          wallet=150000000,
          portfolio_of_plants={},
          portfolio_of_projects={},
          periodicity=6,
          tolerance=[12],
          last_acquisition_period=0,
          source=[{0: 1000}, {1: 500}, {0: 0}],
          decision_var=0.25,
          LSS_thresh=[0.25],
          impatience=[5],
          memory=[12],
          discount=[.0001],
          past_weight=[0.5],
          current_weight=[0.25])

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

DBB = DBB(env=env,
          wallet=20 * 10 ** 7,
          policy={'instrument': 'finance',
                  'source': 1},
          source=[{1: 1000}, {2: 500}, {0: 0}],
          decision_var=0.5,
          LSS_thresh=[.25, .5],
          past_weight=[1],
          memory=[12],
          discount=[0.001],
          policies=[],
          impatience=[10],
          disclosed_thresh=[0.2],
          rationale=['green'])

DD = Demand(env=env,
            initial_demand=INITIAL_DEMAND,
            when=120,
            increase=INITIAL_DEMAND * 0.1)
