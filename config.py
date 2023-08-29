#################################################################
#                                                               #
#                        Packages go here                       #
#                                                               #
#################################################################

# check before importing
import importlib
import sys
from importlib import reload
# from main import config_name
import simpy

reload(sys.modules[__name__])

importlib.invalidate_caches()
import __init__
reload(__init__)
from __init__ import config_name

# print('config', config_name)

import pathlib
from importlib import resources

importlib.invalidate_caches()

# from __main__ import env

# !pip install simpy #on colab it must be this pip install thing, dunno why

doug = 10
buffer_period = 200

# import _config____1_YES_YES
# reload(_config____1_YES_YES)
# from   _config____1_YES_YES import *

file1 = __import__(config_name[0])  # importlib.import_module('_config____1_YES_YES')
reload(file1)  # reload(__import__('_config____1_YES_YES'))
# importlib.invalidate_caches()
# file1 = importlib.import_module('_config____1_YES_YES')


for attr in vars(file1):
    locals()[attr] = vars(file1)[attr]
# del sys.modules['_config____1_YES_YES']
# sys.modules['_config____1_YES_YES'] = importlib.import_module('_config____1_YES_YES')

# print(vars())


