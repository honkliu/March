class BaseConf:
    def default(self):
        self.parameter1 = None
        self.parameter2 = None


class BaseLayer:
    pass


class BiQRNN(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len):
        pass

class BiQRNNConf(BaseConf):
    pass


class Linear(BaseLayer):
    def __init__(self, layer_conf):
        pass
    def forward(self, string, string_len=None):
        pass


class LinearConf(BaseConf):
    def default(self):
        self.hidden_dim = 128
        self.batch_norm = True     # currently, batch_norm for rank 3 inputs is disabled
        self.activation = 'PReLU'
        self.last_hidden_activation = True
        self.last_hidden_softmax = False


class FullAttention(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string1, string1_len, string2, string2_len, string1_HoW, string1_How_len, string2_HoW, string2_HoW_len):
        pass


class FullAttentionConf(BaseConf):
    pass


class Combination(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, *args):
        pass


class CombinationConf(BaseConf):
    pass

class Dropout(BaseLayer):
    def __init__(self, layer_conf):
        pass
    def forward(self, string, string_len=None):
        pass

class DropoutConf(BaseConf):
    pass


class HighwayLinear(BaseLayer):
    def __init__(self, layer_conf):
        pass
    def forward(self, string, string_len):
        pass

class HighwayLinearConf(BaseConf):
    pass

class LinearAttention(BaseLayer):
    def __init__(self, layer_conf):
        pass
    def forward(self, string, string_len=None):
        pass

class LinearAttentionConf(BaseConf):
    pass


class Seq2SeqAttention(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len, string2, string2_len=None):
        pass

class Seq2SeqAttentionConf(BaseConf):
    pass

class SLUEncoder(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len):
        pass

class SLUEncoderConf(BaseConf):
    pass

import torch.nn as nn

class Add2D(nn.Module):
    def __init__(self, layer_conf):
        pass

    def forward(self, *args):
        pass

class Add2DConf(BaseConf):
    pass

class Add3D(nn.Module):
    def __init__(self, layer_conf):
        pass

    def forward(self, *args):
        pass

class Add3DConf(BaseConf):
    pass

class ElementWisedMultiply2D(nn.Module):
    def __init__(self, layer_conf):
        pass

    def forward(self, *args):
        pass

class ElementWisedMultiply2DConf(BaseConf):
    pass

class MatrixMultiply(nn.Module):
    def __init__(self, layer_conf):
        pass
    def forward(self, *args):
        pass

class MatrixMultiplyConf(BaseConf):
    pass

class Minus2D(nn.Module):
    def __init__(self, layer_conf):
        pass

    def forward(self, *args):
        pass

class Minus2DConf(BaseConf):
    pass

class Minus3D(nn.Module):
    def __init__(self, layer_conf):
        pass

    def forward(self, *args):
        pass

class Minus3DConf(BaseConf):
    pass

class LayerNorm(nn.Module):
    def __init__(self,layer_conf):
        pass
    def forward(self, string, string_len):
        pass

class LayerNormConf(BaseConf):
    pass

class Concat2D(nn.Module):
    def __init__(self, layer_conf):
        pass

    def forward(self, *args):
        pass

class Concat2DConf(BaseConf):
    pass

class Concat3D(nn.Module):
    def __init__(self,layer_conf):
        pass

    def forward(self, *args):
        pass

class Concat3DConf(BaseConf):
    pass

class MultiHeadAttention(nn.Module):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len):
        pass

class MultiHeadAttentionConf(BaseConf):
    pass

class MLP(nn.Module):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len):
        pass

class MLPConf(BaseConf):
    pass

class Transformer(nn.Module):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len):
        pass

class TransformerConf(BaseConf):
    pass

class BiGRU(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len):
        pass

class BiGRUConf(BaseConf):
    pass

class BiGRULast(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len):
        pass

class BiGRULastConf(BaseConf):
    pass

class BiLSTM(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len):
        pass

class BiLSTMConf(BaseConf):
    pass

class BiLSTMAtt(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len):
        pass

class BiLSTMAttConf(BaseConf):
    pass

class BiLSTMLast(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len):
        pass

class BiLSTMLastConf(BaseConf):
    pass

class Conv(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len=None):
        pass

class ConvConf(BaseConf):
    pass

class ConvPooling(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len=None):
        pass

class ConvPoolingConf(BaseConf):
    pass

class EncoderDecoder(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len):
        pass

class EncoderDecoderConf(BaseConf):
    pass

class Flatten(nn.Module):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len):
        pass

class FlattenConf(BaseConf):
    pass

class Pooling(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len=None):
        pass

class PoolingConf(BaseConf):
    pass

class SLUDecoder(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, string, string_len, context, encoder_outputs):
        pass

class SLUDecoderConf(BaseConf):
    pass

class Embedding(BaseLayer):
    def __init__(self, layer_conf):
        pass

    def forward(self, inputs, use_gpu=False, transform_variable=False):
        pass

class EmbeddingConf(BaseConf):
    pass