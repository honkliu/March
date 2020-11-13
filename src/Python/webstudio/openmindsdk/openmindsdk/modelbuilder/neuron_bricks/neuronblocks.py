from itertools import chain

from ..utils import dynamic_load, f, Callable, ModelBrickWrapper, os

from inspect import getmembers, signature

class NeuronBlockBase(ModelBrickWrapper):
    FILENAME = __file__
    FRAMEWORK = "NeuronBlocks"

    def __declare__(self):
        super(ModelBrickWrapper, self).__declare__()    # call __declare__ from ModelBrickWrapper's father
        try:
            self.conf_class_name = self.BASEMODEL + 'Conf'
            print("conf_class_name:" + self.conf_class_name)
            conf_class = dynamic_load(self.conf_class_name)()
        except Exception as e:
            self.conf_class_name = f('block_zoo.BaseLayer.BaseConf')
            conf_class = dynamic_load(self.conf_class_name)()
        conf_class.default()
        for name, thing in getmembers(conf_class):
            if name in ['input_channel_num', 'num_of_inputs', 'input_ranks']:
                continue
            if name.startswith('_') or name.startswith('__'):
                continue
            if callable(thing):
                continue
            self.register_fe_param(name, thing)
        layer_class = dynamic_load(self.BASEMODEL)
        ports = list(signature(layer_class.forward).parameters.keys())
        assert ports[0] == 'self', ports
        if ports[1] == 'args':
            self.set('input-labels', ['args'])
        else:
            out_ = []
            for t, tlen in zip(ports[1::2], ports[2::2]):
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

    def get_group(self):
        base_model = dynamic_load(self.BASEMODEL)
        group = base_model.__module__.split('.')[-2]
        if group == 'block_zoo':
            group = 'basic'
        return group


pth = '../NeuronBlocks/block_zoo/'
__group = []
__group_blocks = {}
__basic_block = []
for file in os.listdir(pth):
    file_name = file.split('.')[0]
    if file_name.startswith('__') or file_name == 'BaseLayer':
        continue
    if os.path.isfile(pth + file):
        __basic_block.append(file_name)
    if os.path.isdir(pth + file):
        __group.append(file)
for group in __group:
    blocks = []
    group_pth = pth + group + '/'
    for file in os.listdir(group_pth):
        file_name = file.split('.')[0]
        if file_name.startswith('__'):
            continue
        if os.path.isfile(group_pth + file):
            blocks.append(file_name)
    __group_blocks[group] = blocks
__group.append('basic')
__group_blocks['basic'] = __basic_block
__group.append('function')
__group_blocks['function'] = ['NeuronInput', 'NeuronOutput']

for group in __group:
    blocks = __group_blocks[group]
    for block in blocks:
        if group == 'basic':
            neuron_blocks_pth = 'block_zoo.' + block
        elif group == 'function':
            neuron_blocks_pth = 'openmindsdk.modelbuilder.neuron_bricks.' + group + '.' + block
        else:
            neuron_blocks_pth = 'block_zoo.' + group + '.' + block
        globals()[block] = type(block, (NeuronBlockBase,),
                              {'BASEMODEL': f('{neuron_blocks_pth}.{block}'), 'SHOW_IN_GALLERY': True})



