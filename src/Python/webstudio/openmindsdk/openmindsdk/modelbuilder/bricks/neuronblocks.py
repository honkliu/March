from itertools import chain

from .utils_bricks import SubGraphIn, SubGraphOut
from ..utils import ModelBrickWrapper

from ...utils import ClassCoder, ModelCoder
from ...utils import dynamic_load, f, Callable

from inspect import getmembers, signature


__all_blocks = ['BiQRNN', 'Linear', 'FullAttention', 'Combination', 'Dropout', 'HighwayLinear', 'LinearAttention',
                'Seq2SeqAttention', 'SLUEncoder', 'Add2D', 'Add3D', 'ElementWisedMultiply2D', 'MatrixMultiply',
                'Minus2D', 'Minus3D', 'LayerNorm', 'Concat2D', 'Concat3D', 'MultiHeadAttention', 'MLP', 'Transformer',
                'BiGRU', 'BiGRULast', 'BiLSTM', 'BiLSTMAtt', 'BiLSTMLast', 'Conv', 'ConvPooling',
                'EncoderDecoder', 'Flatten', 'Pooling', 'SLUDecoder', 'Embedding']
neuron_blocks_pth = 'openmindsdk.modelbuilder.bricks.neuronblock_interface'


class NeuronBlockBase(ModelBrickWrapper):
    FILENAME = __file__
    FRAMEWORK = "NeuronBlocks"

    def __declare__(self):
        super(ModelBrickWrapper, self).__declare__()    # call __declare__ from ModelBrickWrapper's father
        try:
            self.conf_class_name = self.BASEMODEL + 'Conf'
            print("conf_class_name:" + self.conf_class_name)
            conf_class = dynamic_load(self.conf_class_name)()
        except:
            self.conf_class_name = f('{neuron_blocks_pth}.BaseConf')
            conf_class = dynamic_load(self.conf_class_name)()
        conf_class.default()
        for name, thing in getmembers(conf_class):
            if not (name.startswith('__') or callable(thing)):
                self.register_fe_param(name, thing)
        layer_class = dynamic_load(self.BASEMODEL)
        ports = list(signature(layer_class.forward).parameters.keys())
        assert ports[0] == 'self', ports
        if ports[1] == 'args':
            self.set('input-labels', ['args'])
        else:
            out_ = []
            for t, tlen in zip(ports[1::2], ports[2::2]):
                # assert tlen.lower() == t.lower() + '_len', (t, tlen)
                if tlen.lower() != t.lower() +'_len':
                    continue
                out_.append(t)
            self.set('input-labels', out_)

    def run_this_brick(self, **kwargs):
        gen, cfg =  kwargs['code_gen'], kwargs['cfg_gen']
        # generate code
        gen.add_module(self.conf_class_name)
        gen.add_module(self.BASEMODEL)
        conf_ = self.variable('conf')
        conf_params = {k['fe_name']:self.get(k['fe_name']) for k in self.fe_parameters if self.get(k['fe_name']) != k['default']}
        gen.method_init_.add_a_line('# code from {}'.format(self.get('instance-name')))
        gen.method_init_.add_a_line('{} = {}'.format(conf_, Callable(self.conf_class_name).code(argdict=conf_params)))
        gen.method_init_.add_a_line('self.{} = {}({})'.format(self.variable('layer'), self.BASEMODEL.split('.')[-1], conf_))

        gen.method_forward_.add_a_line('# code from {}'.format(self.get('instance-name')))
        inputs_ = list()
        for port, sources in self.get_inputs(flatten=False, catenate=False).items():
            if port == 'args':
                inputs_.extend(list(chain.from_iterable([[s, s+'_len'] for s in sources])))
            else:
                inputs_.extend(list(chain.from_iterable([['{}={}'.format(port,s), '{}_len={}_len'.format(port,s)] for s in sources])))
        for out_ in self.get('output-labels'):
            out_var = self.port_name(out_)
            gen.method_forward_.add_a_line('{}, {}_len = self.{} ({})'.format(out_var, out_var, self.variable('layer'), ','.join(inputs_)))
        # here is to generate the json file used by NeuronBlocks
        input_insts = dict()
        conf_params = {k['fe_name']:self.get(k['fe_name']) for k in self.fe_parameters}
        for port, source in self.get('inputs').items():
            input_insts.setdefault(port,[]).extend([src[0] for src in source])
        cfg.setdefault('architecture', []).append({
            'layer': self.BASEMODEL.split('.')[-1],
            'layer_id': self.get('instance-name'),
            'inputs':list(chain.from_iterable([input_insts[k] for k in self.get('input-labels')])),
            'conf':conf_params
        })


class NeuronBlockIn(SubGraphIn):
    FRAMEWORK = 'NeuronBlocks'
    FILENAME = __file__

    def run_this_brick(self, **kwargs):
        gen, cfg =  kwargs['code_gen'], kwargs['cfg_gen']
        gen.method_forward_.add_func_args(self.port_name())


class NeuronBlockOut(SubGraphOut):
    FILENAME = __file__
    FRAMEWORK = 'NeuronBlocks'

    def __declare__(self):
        super().__declare__()
        self.set('is-subgraph-output', False)

    def generate_code(self):
        cfg = dict()
        gen = ModelCoder('model_' + self.id(), ClassCoder)
        gen.add_a_base('{}.BaseLayer'.format(neuron_blocks_pth))
        gen.method_init_ = gen.add_a_method('__init__', args=['self'])
        gen.method_forward_ = gen.add_a_method('forward', args=['self'])
        self.traceback_run(state=dict(), code_gen=gen, cfg_gen=cfg)
        return gen, cfg

    def run_this_brick(self, **kwargs):
        gen, cfg =  kwargs['code_gen'], kwargs['cfg_gen']
        outputs_ = list()
        for port, src in self.get_inputs(flatten=False, catenate=True).items():
            p_ = self.port_name(port)
            gen.method_forward_.add_a_line('{}, {}_len = {}, {}_len'.format(p_, p_, src, src))
            outputs_.extend([p_, p_ + '_len'])
        gen.method_forward_.add_a_line('return {}'.format(', '.join(outputs_)))


for blk in __all_blocks:
    globals()[blk] = type(blk, (NeuronBlockBase,), {'BASEMODEL':f('{neuron_blocks_pth}.{blk}'), 'SHOW_IN_GALLERY':True})