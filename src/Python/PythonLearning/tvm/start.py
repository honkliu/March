import matplotlib.pyplot as plt 
import numpy as np

import tvm as tvm

from tvm import relay
from tvm.relay import testing

from tvm.contrib import graph_runtime

x = np.linspace(0, 20, 100)

plt.plot(x, np.sin(x))
plt.show()
