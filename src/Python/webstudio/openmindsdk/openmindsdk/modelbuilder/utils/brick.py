"""
This file defines the basic component *Brick* in the task builder.
more Brick types defined in bricks/utils_bricks.py
"""
import os
import uuid
import re
import shutil
import sys
import json
import numpy as np

from ...utils import ModelCoder, Callable
from ...utils import dynamic_load, f, sub_unvalid_chars
from .json_convert_utils import UltiFlowBrick


class Brick(UltiFlowBrick):
    """ Brick is the basic module that to drag and drop and comprise a task
    """
    FILENAME = None
    FRAMEWORK = 'Common'
    SHOW_IN_GALLERY = True

    def __declare__(self):
        super().__declare__()
        self.batch_set([
            'type',  # type of class
            'instance-name',  # instance name
            'role'  # role can be starter, reader, model-component
        ])
        self.parent = None  # parent of a brick is the taskbuilder

    def __customize__(self, config):
        super().__customize__(config)
        if self.FILENAME is not None:
            self.set_if_undefined('type', value = self.__get_class_name())

    def get_display_name(self):
        """
        display name is to be shown in the library tool in GUI
        the path to be shown like FRAMEWORK::group::diaplay_name
        :return:
        """
        assert self.get('type') is not None, f('why None type of {self.__get_class_name()}')
        return self.get("type").split(".")[-1]

    def get_group(self):
        """
        group is shown in the GUI
        :return:
        """
        return 'function'

    def get_display_group(self, group=None):
        """
        make group to be better shown
        :param group:
        :return:
        """
        if group is None:
            group = self.get_group()
        return ' '.join([token.capitalize() for token in group.split('_')])

    def port_name(self, name_: str):
        """

        :param name_: port name
        :return: an unique id for a port in the way {instance-name}_{port-name}
        """
        assert name_ in self.get('input-labels') or name_ in self.get('output-labels'), \
            'unknown port: {}, should be in {} or {}'.format(name_, ','.join(self.get('input-labels')), ','.join(self.get('output-labels')))
        return sub_unvalid_chars(self.get('instance-name')) + '_' + sub_unvalid_chars(name_)

    def traceback_run(self, state: dict, **kwargs):
        """
        run the method for a graph in the order of connecting, every node run only once
        :param state: a dict to indicate the intermediate state
        :param kwargs:
        :return:
        """
        self.traceback(state, **kwargs)
        return self.run_this_brick(**kwargs)

    def run_this_brick(self, **kwargs):
        return None

    def traceback(self, state: dict, **kwargs):
        for port, sources in self.get('inputs').items():
            for src in sources:
                src_inst_name, src_port = src[0], src[1]
                if src_inst_name not in state.keys():
                    state[src_inst_name] = self.parent.find_instance(src_inst_name).traceback_run(state, **kwargs)

    def get_inputs(self):
        """
        get the inputs
        :return: a dictionary with the input-port-name as key, and a list of all instance-port-name connected to this port as value
            { in-port-1: [[src-inst-1, port-1], [src-inst-2, port-2]]}
        """
        args = {p:[] for p in self.get('input-labels')}
        for port, sources in self.get('inputs').items():
            for src in sources:
                src_inst, src_port = src[0], src[1]
                args[port].append(self.parent.find_instance(src_inst).port_name(src_port))
        return args

    # utils
    def __get_class_name(self):
        """
        get the class name of this brick
        :return: should be like openmindsdk.modelbuilder.bricks.{file-name}.{class-name}
        """
        classname = self.__class__.__name__
        if self.FILENAME is None:
            return classname
        # assert self.FILENAME is not None, 'class {} has None FILENAME'.format(classname)
        path = os.path.normpath(os.path.relpath(self.FILENAME)).split(os.sep)
        path[-1] = os.path.splitext(path[-1])[0]
        path.append(classname)
        # should be openmindsdk.modelbuilder.bricks.{}.{}
        if len(path) > 5:
            path = path[-5:]
        return '.'.join(path)

    # pipeline operations
    def connect(self, builder: 'ModelPipeline'):
        """
        connect this brick to a parent pipeline
        :param builder:
        :return:
        """
        self.parent = builder
        self.name_myself()
        builder.stages.append(self)

    def query(self, instance_name: str=None, role: str=None):
        """
        return whether the brick matches the instance-name or role
        :param instance_name:  target {instance-name}
        :param role: target role
        :return: True / False
        """
        flag = False
        if instance_name is not None and self.get('instance-name') == instance_name:
            flag = True
        if role is not None and self.get('role').startswith(role):
            flag = True
        return flag

    def pget(self, *keys):
        """
        get a valid value of keys, if keys not exist, go to self.parent.states[0] to have a try
        :param args:
        :return:
        """
        if self.defined(*keys):
            return super().get(*keys)
        assert self.parent is not None, f('parent is not set when getting {args}')
        return self.parent.stages[0].get(*keys)

    def name_myself(self):
        """
        give a unique name for thie brick -> {instance-name}
        :return:
        """
        if self.parent is None:
            name = self.get('type') + '-' + str(uuid.uuid1())
        else:
            basetype = self.get('type').split('.')[-1]
            name = str(len(self.parent.stages)) + '-' + basetype
        self.set_if_undefined('instance-name', value = name)

    @staticmethod
    def call_myself(config: dict):
        """
        call a brick according to the config
        :param config: a dict should contain {'type': call_name, ...}
        :return:
        """
        type_ = config['type']
        inst = dynamic_load(type_)
        if inst is None:
            raise Exception(f('{type_} is not callable'))
        return inst(config)


class DataReaderBrick(Brick):
    """
    DataReaderBrick defines basic functions to read the dataset
    it can pass data through (x_data, y_data) or by a generator returning a batch by next()
    """
    def __declare__(self):
        super().__declare__()
        self.set('role', 'data-reader')
        self.batch_set(['x-shape', 'y-shape', 'batch-size'], [[None], [None], 0])
        self.add_fe_parameters('x-shape', 'y-shape', 'batch-size')

        self.x_data, self.y_data, self.data_gen = None, None, None
        self.valid_x_data, self.valid_y_data, self.valid_data_gen = None, None, None

    def get_data_ready(self):
        raise NotImplementedError

    def explore_data(self):
        raise NotImplementedError


class ModelBrick(Brick):
    """
    ModelBrick can be
    (1) pre-trained or pre-defined substructure of a DNN model
    (2) a single layer
    we support to generate a model from multiple pipelined bricks
    """

    def __declare__(self):
        super().__declare__()
        self.set('role', 'model-component')
        self.set('trainable', True)
        self.set('input-labels', ['in-port-0'])
        self.set('output-labels', ['out-port-0'])

    def get_inputs(self, flatten: bool=True, catenate: bool=True):
        def cat(a: list):
            if len(a) == 1:
                return a[0]
            return '[{}]'.format(','.join(a))
        args = super().get_inputs()
        if catenate:
            args = {k: cat(v) for k, v in args.items()}
        if flatten:
            assert len(args) == 1
            return list(args.values())[0]
        return args

    def run_this_brick(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        self.generate_code_inner(kwargs['code_gen'])

    def generate_code_inner(self, gen: ModelCoder):
        """
        to generate the code of this brick
        :param gen:
        :return:
        """
        raise NotImplementedError

    def variable(self, var_name: str):
        """
        define a variable name with the instance-name as prefix
        :param var_name:
        :return: variable name with prefix of instance name
        """
        return sub_unvalid_chars(self.get('instance-name')) + '_' + sub_unvalid_chars(var_name)


class ModelBrickWrapper(ModelBrick):
    """
    this is the wrapper a layer or model from the framework
    this class should only be used as the base class of a type function, so itself will not be shown
    """
    BASEMODEL = None
    SHOW_IN_GALLERY = False

    def __declare__(self):
        super().__declare__()
        bmodel = Callable(self.BASEMODEL)
        self.bmodel_gen = bmodel
        self.set('basemodel', bmodel.list_config())
        self.add_fe_parameters("basemodel::argdict::*")

    def get_display_name(self):
        name = self.BASEMODEL
        name = name.split(".")[-1]
        return name

    def get_group(self):
        base_model = dynamic_load(self.BASEMODEL)
        return base_model.__module__.split('.')[-1]


class Launcher(Brick):
    """
    Launcher is the brick to execute training and predicting
    """
    def __declare__(self):
        super().__declare__()
        self.set('role', 'model-launcher')

    def train(self, model=None):
        raise NotImplementedError

    def predict(self, model=None):
        raise NotImplementedError


class NodeBrick(Brick):
    """ NodeBrick provides the functions to
    (1) manage the job config / result (local or remote dicrectory)
    (2) submit or query job
    """
    def __declare__(self):
        super().__declare__()
        self.set('role', 'compute-node')
        self.files = dict()
        self.copied = dict()
        self.result_files = []

    def register(self, filename, description: str="", action: list=[]):
        """
        register a file that will be copied to shared memory and feedback to user
        :param filename:
        :param description:
        :param action: view / plot / download ...
        :return: filename with prefix of working directory
        """
        self.result_files.append(filename)
        self.files.update({filename:{'description':description, 'action':['view']+action}})
        self.copied[filename] = False
        return self.local(filename)

    def local(self, *filename):
        """
        get the whole path of the file in working directory
        :param filename:
        :return: the path in working-directory
        """
        workdir = self.pget('trial-working')
        if len(filename) == 0:
            return workdir
        return os.path.join(workdir, *filename)

    def storage(self, *filename):
        """
        get the whole path of this file in storage
        :param filename:
        :return: whole path
        """
        workdir = self.pget('trial-storage')
        if len(filename) == 0:
            return workdir
        return os.path.join(workdir, *filename)

    def copy_result_files(self):
        raise NotImplementedError

    def submit_job(self, commands: list=[]):
        raise NotImplementedError

    def submit_training_job(self):
        return self.submit_job(self.parent.command('--train'))

    def report(self, content: dict, from_storage: bool=True):
        content.update({
            'results': self.files,
            'directory': self.storage() if from_storage else self.local(),
        })


def deprecated(func):
    def wrapped(*args, **kwargs):
        obj = func(*args, **kwargs)
        setattr(obj, 'SHOW_IN_GALLERY', False)