#################################################################
#                                                               #
#                  This is the list of agents:                  #
#                    First, the public agents                   #
#                                                               #
#################################################################
import random
import sys
import builtins
# from IPython.lib import deepreload
# builtins.reload = deepreload.reload
import importlib
from importlib import reload

reload(sys.modules[__name__])

importlib.invalidate_caches()
import __init__
reload(__init__)
from __init__ import agents_name

# print('agents', agents_name)

# import config
# from config import *
# from classes import *



file2 = __import__(agents_name[0])  # importlib.import_module('_agents____1_YES_YES')
reload(file2)  # reload(__import__('_agents____1_YES_YES'))
# importlib.invalidate_caches()
# file2 = importlib.import_module('_agents____1_YES_YES')


for attr in vars(file2):
    try:
        locals()[attr] = vars(file2)[attr]
    except:
        None

# del sys.modules['_agents____1_YES_YES']
# sys.modules['_agents____1_YES_YES'] = importlib.import_module('_agents____1_YES_YES')

# import _agents____1_YES_YES
# reload(_agents____1_YES_YES)
# from   _agents____1_YES_YES import *
