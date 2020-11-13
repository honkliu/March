# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from block_zoo.BaseLayer import BaseLayer, BaseConf
from .encoder_decoder import *
from utils.DocInherit import DocInherit
from utils.exceptions import ConfigurationError
import copy

class EncoderDecoderConf(BaseConf):
    """ Configuration of Encoder-Decoder

    Args:
        encoder (str): encoder name
        encoder_conf (dict): configurations of encoder
        decoder (str): decoder name
        decoder_conf (dict): configurations of decoder

    """
    def __init__(self, **kwargs):
        #self.encoder_conf_cls = eval(kwargs['encoder'] + "Conf")(**kwargs['encoder_conf'])
        #self.decoder_conf_cls = eval(kwargs['decoder'] + "Conf")(**kwargs['decoder_conf'])

        super(EncoderDecoderConf, self).__init__(**kwargs)

    @DocInherit
    def default(self):
        self.encoder_name = "SLUEncoder"
        self.decoder_name = "SLUDecoder"

        self.encoder_conf = dict()
        self.decoder_conf = dict()

    @DocInherit
    def declare(self):
        self.num_of_inputs = 1
        self.input_ranks = [3]

    @DocInherit
    def inference(self):
        super(EncoderDecoderConf, self).inference()

    @DocInherit
    def verify_before_inference(self):
        super(EncoderDecoderConf, self).verify_before_inference()

    @DocInherit
    def verify(self):
        super(EncoderDecoderConf, self).verify()




class EncoderDecoder(BaseLayer):
    """ The encoder decoder framework

    Args:
        layer_conf (EncoderDecoderConf): configuration of a layer
    """
    def __init__(self, layer_conf):
        super(EncoderDecoder, self).__init__(layer_conf)
        self.layer_conf = layer_conf

        self.encoder = eval(layer_conf.encoder_name)(layer_conf.encoder_conf_cls)
        self.decoder = eval(layer_conf.decoder_name)(layer_conf.decoder_conf_cls)

    def forward(self, string, string_len):
        """ process inputs with encoder & decoder

        Args:
            string (Variable): [batch_size, seq_len, dim]
            string_len (ndarray): [batch_size]

        Returns:
            Variable : decode scores with shape [batch_size, seq_len, decoder_vocab_size]
        """
        encoder_output, encoder_context = self.encoder(string, string_len)
        decoder_scores = self.decoder(string, string_len, encoder_context, encoder_output)
        return decoder_scores, string_len
