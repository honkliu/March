# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


from block_zoo.BaseLayer import BaseLayer, BaseConf
from utils.DocInherit import DocInherit

class NeuronInput(BaseLayer):
    def __init__(self, layer_conf):
        super(NeuronInput, self).__init__(layer_conf)

    def forward(self, string, string_len=None):
        pass


class NeuronInputConf(BaseConf):
    def __init__(self, **kwargs):
        super(NeuronInputConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.name = ''

    @DocInherit
    def declare(self):
        super(NeuronInputConf, self).declare()

    @DocInherit
    def inference(self):
        super(NeuronInputConf, self).inference()

    @DocInherit
    def verify(self):
        super(NeuronInputConf, self).verify()





