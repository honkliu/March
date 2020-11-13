from openmindsdk.modelbuilder.utils import Brick
import os
import sys

from tensorflow.core.framework import op_def_pb2
from google.protobuf.text_format import Merge
from google.protobuf.json_format import MessageToDict


def get_ops():
    sdk_path = os.path.dirname(__file__)
    ops_file = os.path.join(sdk_path, 'mmdnn_ops.pbtxt')

    ops = op_def_pb2.OpList()
    with open(ops_file) as fn:
        Merge(fn.read(), ops)
    ops_list = [MessageToDict(op) for op in ops.op]
    return ops_list
