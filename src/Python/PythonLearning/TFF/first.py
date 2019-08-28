#@test {"skip": true}

# NOTE: If you are running a Jupyter notebook, and installing a locally built
# pip package, you may need to edit the following to point to the '.whl' file
# on your local filesystem.

# NOTE: The high-performance executor components used in this tutorial are not
# yet included in the released pip package; you may need to compile from source.

# NOTE: Jupyter requires a patch to asyncio.
from __future__ import absolute_import, division, print_function

import nest_asyncio
nest_asyncio.apply()


import collections
import warnings
from six.moves import range
import numpy as np
import six
import tensorflow as tf

warnings.simplefilter('ignore')

tf.compat.v1.enable_v2_behavior()

import tensorflow_federated as tff

np.random.seed(0)

NUM_CLIENTS = 2 

# NOTE: If the statement below fails, it means that you are
# using an older version of TFF without the high-performance
# executor stack. Call `tff.framework.set_default_executor()`
# instead to use the default reference runtime.
#if six.PY3:
#      tff.framework.set_default_executor(
#                    tff.framework.create_local_executor(NUM_CLIENTS))
#      tff.framework.set_default_executor()

tff.federated_computation(lambda: 'Hello, World!')()

#do not know how.
# good
