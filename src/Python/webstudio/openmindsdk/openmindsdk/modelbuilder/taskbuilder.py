from __future__ import division
import os
import sys
import json
import shutil
import warnings
import collections
from typing import TextIO
from inspect import isclass, ismodule

# import form common utils (openmindsdk)
from ..utils import dynamic_load, pretty_print, sub_unvalid_chars, dynamic_load_module, FuncCoder, to_file
from ..utils import Configurable, ModelCoder, SourceCoder, ClassCoder
from ..utils import safe_open, notify, random_string, safe_chdir, f

# import form local utils (modelbuilder)
from .utils import convert_fe_json_to_be_json, gen_model_runner, dump_notebook, convert_fe_json_to_be_json_gojs
from .utils import Brick
from .bricks import Starter, ComputeNodeLocal
from ..utils.code_gen import CodeBlockGen


class ModelPipeline:
    '''
    to generate a pipelined model, which includes the following stages
    - starter
    - reader
    - model componet
    '''

    def __init__(self, config: list, from_fe: bool=False):
        ''' config is list (or tuple) of dict, each item (dict) configures a stage'''
        assert isinstance(config, (list, tuple,))
        self.model = None
        self.stages = []

        starter_type = "openmindsdk.modelbuilder.bricks.utils_bricks.Starter"
        starter_cfg = [x for x in config if x['type'] == starter_type]
        if len(starter_cfg) == 1:
            self.__add_stage(starter_cfg, from_fe)
        else:
            self.__add_stage({
                "type": starter_type,
            }, from_fe=False)

        for cfg in (x for x in config if x['type'] != starter_type):
            assert isinstance(cfg, dict)
            self.__add_stage(cfg, from_fe)

        self.node = self.find_role('compute-node', unique=True)
        if self.node is None:
            self.__add_stage(ComputeNodeLocal().list_config())
        self.node = self.find_role('compute-node', unique=True)

        self.model_func = None
        pretty_print([x.get('instance-name') for x in self.stages])

    def __add_stage(self, config: dict, from_fe: bool=False):
        config0 = {k:v for k,v in config.items() if k != 'parameters'} if from_fe else config
        brick = Brick.call_myself(config0)
        if from_fe:
            brick.update_from_fe(config['parameters'])
        brick.connect(self)

    def find_role(self, role: str, unique=False):
        roles = [s for s in self.stages if s.query(role=role)]
        if not unique:
            return roles
        else:
            return roles[0] if len(roles) == 1 else None

    def find_instance(self, name: str):
        insts = list(filter(lambda x: x.query(instance_name=name), self.stages))
        assert len(insts) <= 1, '%d instance found instance-name == %s' % (len(insts), name)
        return insts[0] if len(insts) == 1 else None

    def find(self, key: str):
        inst = self.find_instance(key)
        if inst is not None:
            return inst
        return self.find_role(key, unique=True)

    def list_config(self, fn: TextIO=sys.stdout, file: str='backend.json'):
        cfg = [s.list_config() for s in self.stages]
        if file is not None:
            filename = self.node.register(file)
            with safe_open(filename, 'w') as fp:
                pretty_print(cfg, 'json', fp)
        if fn is not None:
            pretty_print(cfg, 'yaml', fn)
        return cfg

    def update_config(self, config: dict):
        for key, value in config.items():
            tokens = key.split("::")
            target = self.find(tokens[0])
            if target is not None:
                in_ = Configurable()
                in_.nested_set(tokens[1:], value)
                target.update(in_.list_config())
            elif key.startswith("nni-placeholder") and isinstance(value, dict):
                self.update_config(value)

    def explore_data(self):
        self.find_role('executor', unique=True).traceback_run(dict())

    def generate_model(self, get_model=None):
        self.model_func = get_model
        if get_model is None:
            filename = self.node.local(self.model_source_code)
            with safe_chdir(os.path.dirname(filename)):
                self.model_func = dynamic_load("get_model", c_path=os.path.basename(filename), quiet=False)

    def generate_code(self, layer_type={}, model_parameters = {}):
        #self.list_config(fn=None, file=self.stages[0].get('graph-name')+'.json')
        code_file = SourceCoder()
        for out_ in self.find_role('model-component-output'):
            if out_.FRAMEWORK.lower() in ['pytorch', 'neuronblocks']:
                content, cfg_out = out_.generate_code()
                to_file(cfg_out, self.node.register(self.stages[0].get('graph-name')+'.nb.json'))
            else:
                content = out_.generate_code()
            code_file.codeblocks.setdefault(content.name, content)
            wrapper = out_.generate_model_wrapper(content)
            if wrapper is not None:
                code_file.codeblocks.setdefault('get_model', wrapper)
                out_.model_inputs = content.inputs
            reporter = out_.generate_model_summary(content)
            if reporter is not None:
                code_file.codeblocks.setdefault('model_summary', reporter)
            if model_parameters['enable_nni'] == 'true':
                callbackGetter = out_.generate_model_callback(content)
                if callbackGetter is not None:
                    code_file.codeblocks.setdefault('model_callback', callbackGetter)
                defaultParamsGetter = out_.generate_default_params(content, model_parameters)
                if defaultParamsGetter is not None:
                    code_file.codeblocks.setdefault('default_params', defaultParamsGetter)
            datagetter = out_.generate_get_data(model_parameters)
            if datagetter is not None:
                code_file.codeblocks.setdefault('get_data', datagetter)
        # parse code
        order_dict = self.parse_source_code(code_file, layer_type)

        # task code
        task_code = gen_model_runner(model_parameters, code_file, self)

        if 'dataset' in model_parameters and model_parameters['dataset'] != 'custom':
            run_codeblock = CodeBlockGen('run_code')
            run_codeblock.add_dependency('train')
            run_codeblock.add_a_line("if __name__ == '__main__':")
            lines = '\n'.join(['    ' + line for line in task_code.split('\n')])
            run_codeblock.add_a_line(lines)
            code_file.codeblocks.setdefault('run_code', run_codeblock)

        # print file name to stdout
        self.model_source_code = sub_unvalid_chars(self.stages[0].get('graph-name')) + '.py'
        code_file.write_to_file(self.node.register(self.model_source_code))
        # notebook
        dump_notebook(
            self.node.register(self.stages[0].get('graph-name') + '.ipynb'),
            self.node.local(self.model_source_code),
            framework=out_.FRAMEWORK,
            parameters={'run-task-code': task_code}
        )
        self.store()
        return order_dict

    def parse_source_code(self, code_file, layer_type):
        layer_name_dict = collections.OrderedDict() #{layer_cnt : instance_name}
        layer_cnt_dict = {}
        subgraph_layerCnt_dict = {}

        default_func_name = code_file.codeblocks.get('get_model').dependencies
        if len(default_func_name) == 0:
            return {}
        assert len(default_func_name) == 1

        for default_func in default_func_name:
            # get structure of subgraph
            sub_funcs = code_file.codeblocks.get(default_func).dependencies
            for sub_func in sub_funcs:
                sub_dict = self.parse_subgraph_code(code_file, sub_func)
                subgraph_layerCnt_dict[sub_func] = sub_dict

            # get order of layers in model
            lines = code_file.codeblocks.get(default_func).lines
            for line in lines:
                if ('_out_port_0' not in line) and ('=' in line):
                    left, right = line.split('=', 1)
                    instance_name =left.strip().replace('__','')
                    layer_name = right.split('(')[0].strip()
                    if layer_name in layer_cnt_dict:
                        layer_cnt_dict[layer_name] += 1
                    else:
                        layer_cnt_dict[layer_name] = 1
                    layer_cnt = layer_name+str(layer_cnt_dict[layer_name])
                    layer_name_dict[layer_cnt.lower()] = instance_name
                elif '=' in line:
                    left, right = line.split('=', 1)
                    func_name = right.split('(')[0].strip()
                    if func_name in sub_funcs:
                        # line with subgraph
                        siso_name = left.replace('_out_port_0','').strip()
                        sub_dict = subgraph_layerCnt_dict[func_name]
                        for k, v in sub_dict.items():
                            if k in layer_cnt_dict:
                                layer_cnt_dict[k] += v
                            else:
                                layer_cnt_dict[k] = v
                        siso_layer_type = layer_type[siso_name]
                        layer_cnt = siso_layer_type+str(layer_cnt_dict[siso_layer_type])
                        layer_name_dict[layer_cnt.lower()] = siso_name
        return layer_name_dict

    def parse_subgraph_code(self, code_file, func_name):
        sub_funcs = code_file.codeblocks.get(func_name).dependencies
        struc_dict = {} #layer_name : cnt

        # get layers in subgraphs
        if len(sub_funcs) != 0:
            for sub_func in sub_funcs:
                sub_dict = self.parse_subgraph_code(code_file, sub_func)
                for k,v in sub_dict.items():
                    if k in struc_dict:
                        struc_dict[k] += v
                    else:
                        struc_dict[k] = v

        # get layers except subgraph layer
        lines = code_file.codeblocks.get(func_name).lines
        for line in lines:
            if ('_out_port_0' not in line) and ('=' in line):
                layer_name = line.split('=', 1)[1].split('(')[0].strip()
                if layer_name in struc_dict:
                    struc_dict[layer_name] += 1
                else:
                    struc_dict[layer_name] = 1
        return struc_dict

    def model_summary(self):
        filename = self.node.local(self.model_source_code)
        with safe_chdir(os.path.dirname(filename)):
            summary_fn = dynamic_load("model_summary", c_path=os.path.basename(filename), quiet=False)
        if summary_fn is None:
            warnings.warn(f('fail to load model_summary from {filename}'))
            return
        summary_fn(self.node.register(self.stages[0].get('graph-name') + '.summary.txt'))
        self.store()

    def train(self):
        optimizer = self.find_role('optimizer', unique=True)
        optimizer.compile(self.model)
        return self.launcher.train(self.model, self.get_model_func)

    def build_and_train(self):
        builder.generate_code()
        builder.generate_model()
        builder.explore_data()
        result = builder.train()
        self.store()
        return result

    def store(self):
        assert self.node is not None
        self.node.copy_result_files()

    def notify(self, status: str, messge: str=None, task: str = None, **kwargs):
        content = {
            'status': status,
            'message': messge,
            'task': task
        }
        self.node.report(content, from_storage=status=='Succeed')
        content.update(kwargs)
        notify(content, url=self.stages[0].get('notify-url'))

    def command(self, *args):
        commands = [sys.executable, os.path.basename(__file__), '--config', builder.node.storage('config.json')]
        commands.extend(args)
        return commands


def export_json_forfe(dst_dir: str = 'operators'):
    pth = 'openmindsdk.modelbuilder.bricks'

    def isvalid(b):
        return isclass(b) and issubclass(b, Brick) and b.SHOW_IN_GALLERY and b.__module__.startswith(pth)

    exclude_groups = set(['layers', 'base_layer', 'wrappers', 'input_layer'])
    rename_map = {
        'local': 'local_connected',
        'cudnn_recurrent': 'recurrent',
        'convolutional_recurrent': 'recurrent',
    }

    bricks_config_set = {}

    all_possible_bricks = dynamic_load_module(pth)

    for _name in all_possible_bricks:
        brick_class = dynamic_load('{}.{}'.format(pth, _name))
        assert brick_class is not None, 'fail to load {} from {}'.format(_name, pth)
        if not isvalid(brick_class):
            continue
        brick = brick_class()
        framework = brick.FRAMEWORK
        if framework not in bricks_config_set:
            bricks_config_set[framework] = {}
        group = brick.get_group()
        if group in exclude_groups:
            continue
        if group in rename_map:
            group = rename_map[group]
        if group not in bricks_config_set[framework]:
            bricks_config_set[framework][group] = { 'display_group': brick.get_display_group(group), 'bricks': [] }

        config_forfe = brick.list_config_forfe()
        bricks_config_set[framework][group]['bricks'].append(config_forfe)

    for framework, framework_config_set in bricks_config_set.items():
        for group_name, group in framework_config_set.items():
            print(framework, group_name)
            if group_name == 'core':
                group_name = 'core_layer'
            group_dir = os.path.join(os.path.join(dst_dir, framework), group_name)
            if os.path.exists(group_dir):
                shutil.rmtree(group_dir, ignore_errors = True)
            os.makedirs(group_dir)
            with open(os.path.join(group_dir, 'config.py'), 'w') as config_py:
                config_py.write('name = \'{}\'\n'.format(group['display_group']))

            for brick_config in group['bricks']:
                brick_dir = os.path.join(group_dir, 'operators', brick_config['title'])
                os.makedirs(brick_dir)
                with open(os.path.join(brick_dir, 'config.json'), 'w') as config_json:
                    json.dump(brick_config, config_json, indent = 4)

def preview_model(json_data, file_dir):
    try:
        model_parameters = json_data['modelParameters']
        config = convert_fe_json_to_be_json_gojs(json_data)

        name_type_dict = {}  # {'instance-name' : 'layer-type'}
        name_key_dict = {}  # {'instance-name' : 'key'}
        siso_subgraph_dict = {}  # {'instance-name' : 'subgraph_id'}

        subgraph_last_dict = {}  # {'subgraph-id' : 'last-layer-type'}
        layer_dict = {}  # {'instance-name' : 'input-layer'}

        for i in config:
            name_type_dict[i['instance-name']] = i['layer']
            name_key_dict[i['instance-name']] = i['key']
            if i['layer'] == "SubGraphSiSo":
                siso_subgraph_dict[i['instance-name']] = i['parameters']['subgraph-id']

        # find the last layer of each subgraph
        subgraph_todo = {}
        for i in config:
            if 'SubGraphOut' == i['layer']:
                subgraph_id = i['parameters']['model-id']
                last_layer = i['inputs']['in-port-0'][0][0]
                if name_type_dict[last_layer] == 'SubGraphSiSo':
                    subgraph_todo[subgraph_id] = siso_subgraph_dict[last_layer]
                else:
                    subgraph_last_dict[subgraph_id] = name_type_dict[last_layer]

        while len(subgraph_todo) != 0:
            for k in list(subgraph_todo.keys()):
                if subgraph_todo[k] in subgraph_last_dict:
                    subgraph_last_dict[k] = subgraph_last_dict[subgraph_todo[k]]
                    subgraph_todo.pop(k)

        for siso, subgraph_id in siso_subgraph_dict.items():
            name_type_dict[siso] = subgraph_last_dict[subgraph_id]

        builder = ModelPipeline(config, from_fe=True)
        graph_dir = os.path.dirname(file_dir)
        graph_pth = os.path.split(graph_dir)
        builder.stages[0].set('graph-name', graph_pth[-1])
        builder.stages[0].set('project-name', graph_pth[-2])
        builder.stages[0].set('trial-working', graph_dir)
        builder.stages[0].set('trial-storage', graph_dir)
        order_dict = builder.generate_code(layer_type = name_type_dict, model_parameters = model_parameters)
        builder.model_summary()

        return "Succeed", name_key_dict, order_dict
    except Exception as e:
        return e, {}, {}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', help='dump config.json for fe to the specified directory')
    parser.add_argument('--config', help="basic config file (json)")
    parser.add_argument('--front-end', action='store_true', default=False, help="indicate the config is from Front end")
    parser.add_argument('--preview', action='store_true', help="preview the models and pipeline from Frontend output")
    parser.add_argument('--submit', help="submit jobs to run")
    parser.add_argument('--train', action='store_true', help="to retrain")
    parser.add_argument('--nni-trial', action='store_true', help='nni trial')
    parser.add_argument('--history', action = 'store_true', help = "show result of previous trial")
    parser.add_argument('--notify-url', default='http://127.0.0.1:5000/push', help='the url which accepts notify')

    args = parser.parse_args()
    print(args)

    if args.list:
        export_json_forfe(args.list)
        print('exported successfully')
        sys.exit(0)

    assert args.config
    # isntance construction and parameter update
    with open(args.config, 'r') as fp:
        config = json.load(fp)

    try:
        if args.front_end:
            config = convert_fe_json_to_be_json(config)
        builder = ModelPipeline(config, from_fe=args.front_end)
        graph_dir = os.path.dirname(args.config)
        graph_pth = os.path.split(graph_dir)
        builder.stages[0].set('graph-name', graph_pth[-1])
        builder.stages[0].set('project-name', graph_pth[-2])
    except Exception as e:
        notify({'status': 'Failed', 'message': 'config file contains errors {}, please check'.format(e)}, args.notify_url)
        raise

    try:
        # interface (FE+BE) methods
        if args.history:
            builder.node.register_all()
            builder.notify('Succeed', 'Job history has been updated', 'History')
            sys.exit(0)

        if args.preview:
            builder.stages[0].set('trial-working', graph_dir)
            builder.stages[0].set('trial-storage', graph_dir)
            builder.stages[0].update_notify_url(args.notify_url)
            builder.generate_code()
            builder.notify('Running', 'Source code is generated')
            builder.model_summary()
            builder.notify('Succeed', 'Model summary has been updated', 'Preview')
            sys.exit(0)

        if args.submit == 'train':
            builder.stages[0].update_notify_url(args.notify_url)
            builder.generate_code()
            open_url = builder.node.submit_training_job()
            builder.notify('Succeed', 'Traning job has been submitted', open_url=open_url)
            sys.exit(0)

        if args.submit == 'nni':
            builder.generate_code()
            tunor = builder.find_role('hyper-tunor', unique=True)
            commands = ['nnictl', 'create', '--config', tunor.store_nni_config()]
            builder.node.submit_job(commands)
            sys.exit(0)

        if args.nni_trial:
            import nni
            builder.stages[0].update_trial_id('trial-' + random_string(size=8))
            param = nni.get_parameters()
            print(param)
            builder.update_config(param)
            result = builder.build_and_train()
            # nni.report_final_result(0 if result is None else result['loss'])
            sys.exit(0)

        # build
        if args.train:
            builder = ModelPipeline(config, from_fe=args.front_end)
            builder.build_and_train()
            builder.notify('Succeed', 'Trainig has been done')
            sys.exit(0)
    except Exception as e:
        builder.notify('Failed', 'Exception messae is %s, please check log file for detail' % e)
        raise

def getElements():
    pth = 'openmindsdk.modelbuilder.bricks'

    def isvalid(b):
        return isclass(b) and issubclass(b, Brick) and b.SHOW_IN_GALLERY and b.__module__.startswith(pth)

    exclude_groups = set(['layers', 'base_layer', 'input_layer', 'callbacks', 'optimizers', 'wrappers',
                          'pretrained_networks'])
    rename_map = {
        'local': 'local_connected',
        'cudnn_recurrent': 'recurrent',
        'convolutional_recurrent': 'recurrent',
        'cosineconvolution2d': 'contrib',
        'srelu': 'contrib',
        'subpixelupscaling': 'contrib',
        'groupnormalization': 'contrib',
        'crf': 'contrib',
        'cosinedense': 'contrib',
        'capsule': 'contrib',
        'instancenormalization': 'contrib',
        'swish': 'contrib',
        'sinerelu': 'contrib',
        'pelu': 'contrib',
    }

    bricks_config_set = {}

    all_possible_bricks = dynamic_load_module(pth)
    for _name in all_possible_bricks:
        brick_class = dynamic_load('{}.{}'.format(pth, _name))
        assert brick_class is not None, 'fail to load {} from {}'.format(_name, pth)
        if not isvalid(brick_class):
            continue
        brick = brick_class()
        framework = brick.FRAMEWORK
        if framework.lower() not in ['keras', 'common']:
            continue
        if framework not in bricks_config_set:
            bricks_config_set[framework] = {}
        group = brick.get_group()
        if group in exclude_groups:
            continue
        if group in rename_map:
            group = rename_map[group]
        if group not in bricks_config_set[framework]:
            bricks_config_set[framework][group] = {'display_group': brick.get_display_group(group), 'bricks': []}
        config_forfe = brick.list_config_forfe()
        if group == 'function':
            remove_list=['KerasModelInput', 'KerasModelOutput', 'SubGraphIn', 'SubGraphOut', 'SubGraphSiSo']
            if config_forfe['title'] not in remove_list:
                continue
        bricks_config_set[framework][group]['bricks'].append(config_forfe)

    subgraph_func_dict = bricks_config_set['Common']['function']
    keras_func_dict = bricks_config_set['Keras']['function']
    comb_list = keras_func_dict['bricks']+subgraph_func_dict['bricks']
    keras_func_dict['bricks'] = comb_list

    return bricks_config_set

def getNeuronElements():
    pth = 'openmindsdk.modelbuilder.neuron_bricks'

    def isvalid(b):
        return isclass(b) and issubclass(b, Brick) and b.SHOW_IN_GALLERY and b.__module__.startswith(pth)

    bricks_config_set = {}

    all_possible_bricks = dynamic_load_module(pth)
    for _name in all_possible_bricks:
        brick_class = dynamic_load('{}.{}'.format(pth, _name))
        assert brick_class is not None, 'fail to load {} from {}'.format(_name, pth)
        if not isvalid(brick_class):
            continue
        brick = brick_class()
        framework = brick.FRAMEWORK
        if framework.lower() not in ['neuronblocks']:
            continue
        if framework not in bricks_config_set:
            bricks_config_set[framework] = {}
        group = brick.get_group()
        if group not in bricks_config_set[framework]:
            bricks_config_set[framework][group] = {'display_group': brick.get_display_group(group), 'bricks': []}
        config_forfe = brick.list_config_forfe()
        bricks_config_set[framework][group]['bricks'].append(config_forfe)
    return bricks_config_set
