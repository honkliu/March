# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit

class NeuronOutput(BaseLayer):
    def __init__(self, layer_conf):
        super(NeuronOutput, self).__init__(layer_conf)

    def forward(self, string, string_len=None):
        pass


class NeuronOutputConf(BaseConf):
    def __init__(self, **kwargs):
        super(NeuronOutputConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.name = ''

    @DocInherit
    def declare(self):
        super(NeuronOutputConf, self).declare()

    @DocInherit
    def inference(self):
        super(NeuronOutputConf, self).inference()

    @DocInherit
    def verify(self):
        super(NeuronOutputConf, self).verify()




