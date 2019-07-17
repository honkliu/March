import numpy as np

import tvm
from tvm import rpc
from tvm.contrib import util

n = tvm.convert(1024)
A = tvm.placeholder((n,), name='A')
B = tvm.compute((n,), lambda i: A[i] + 1.0, name='B')
s = tvm.create_schedule(B.op)

local_demo = True

if local_demo:
    target = 'llvm'
else:
    target = 'llvm -target=armv7l-linux-gnueabihf'

func = tvm.build(s, [A, B], target=target, name='add_one')
# save the lib at a local temp folder
temp = util.tempdir()
path = temp.relpath('lib.tar')
func.export_library(path)