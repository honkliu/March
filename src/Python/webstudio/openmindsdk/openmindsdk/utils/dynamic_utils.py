import inspect
import importlib
import os
import sys
import collections
import keras_contrib

from .config_utils import Configurable
from .importing import f


def dynamic_load(c_class, c_module=None, c_package=None, c_path=None, quiet=True):
    '''
    dynamically load a class, None-parameter means current module and package
    @param c_class, c_module, c_package: use importlib.import_module
    @param c_class, c_path: use importlib.util.spec_from_file_location, c_path is path/to/python/file
    '''
    try:
        if c_path:
            spec = importlib.util.spec_from_file_location(c_class, c_path)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
            return getattr(foo, c_class)
        if '.' in c_class:
            assert(c_module is None)
            tmp = c_class.split('.')
            c_class = tmp[-1]
            c_module = '.'.join(tmp[:-1])
        if c_module is None:
            return globals()[c_class]
        else:
            m = importlib.import_module(c_module, package=c_package)
            # importlib.reload(m)
            return getattr(m, c_class)
    except Exception as e:
        if not quiet:
            print(e)
            print(c_class, c_module, c_package, c_path)
        return None

    
class Bindable:
    """define a structure that contains function name and function signature"""
    
    def __init__(self, name: str=None, func=None):
        if name is not None:
            self.init(name, func)

    def init(self, name, func=None):
        self.name = name
        self.func = func if func is not None else dynamic_load(name)
        assert self.func is not None, f('not callable {name}')
        self.sig = inspect.signature(self.func)
        self.ba = self.sig.bind_partial()
        return self

    def default(self, value_book:list=[(inspect._empty, 'Empty')]):
        """
        return the default arguments from signature
        :return:
        """
        default_arg = collections.OrderedDict()
        for k, v in self.sig.parameters.items():
            if k == 'kwargs':
                continue
            default_arg[k] = v.default
        for old_val, new_val in value_book:
            default_arg.update((k,new_val) for k,v in default_arg.items() if v == old_val)
        return default_arg
    
    def bind(self, *args, **kwargs):
        """bind arguments to invoke the function or to store arguments"""
        self.ba = self.sig.bind(*args, **kwargs)
        return self
    
    def to_dict(self):
        return {
            'name': self.name,
            'args': self.ba.args,
            'kwargs': self.ba.kwargs,
        }
    
    def from_dict(self, dic: dict):
        self.init(name=dic['name'])
        self.bind(*dic['args'], **dic['kwargs'])
        return self
    
    def invoke(self):
        return self.func(*self.ba.args, **self.ba.kwargs)


def dynamic_load_module(c_module):
    classes = set()
    try:
        for name, obj in inspect.getmembers(sys.modules[c_module]):
            if inspect.isclass(obj):
                classes.add(name)
        return classes
    except:
        return set()

    
def kw2dict(**kwargs):
    return kwargs


class Callable(Configurable, Bindable):
    '''
    wrap a callable function or class into data structure
    Callable
        |- Settings
            |- name
            |- argdict
            |- default
        |- func
    '''
    def __init__(self, name:str=None, func=None, ):
        if name is None:    # an empty placeholder
            return
        Configurable.__init__(self)
        Bindable.__init__(self, name, func)
        default_arg = self.default()
        self.set('default', default_arg)
        self.set('argdict', self.default())

    def copy_from(self, other):
        super().__init__(other.list_config())
        self.func = other.func

    def set_arg(self, key, value):
        self.nested_set(['argdict', key], value)

    def call(self, argdict: dict={}, use_default: bool=False):
        arg = dict(self.get('default')) if use_default else dict(self.get('argdict'))
        arg.update(argdict)
        Callable.correct_args(arg)
        return self.func(**arg)

    def call2(self, use_default: bool=False, **kwargs):
        return self.call(argdict=kwargs, use_default=use_default)
    
    def code(self, argdict: dict={}, use_default: bool=False, keys_to_eval: list=[], instance_name=""):
        arg = dict(self.get('default')) if use_default else dict(self.get('argdict'))
        arg.update(argdict)
        Callable.correct_args(arg)
        default_ = self.get('default')
        arg_ = {k: v for k, v in arg.items() if k not in default_ or v != default_[k]}
        args = []
        # add 'name'
        if instance_name!='':
            args.append('name =' + '"' + instance_name + '"')
        for k, v in arg_.items():
            if type(v) == str and k not in keys_to_eval:
                args.append(k + '=' + '"' + v.__str__() + '"')
            else:
                args.append(k + '=' + v.__str__())
        return '%s (%s)' % (self.name.split('.')[-1], ', '.join(args))

    @staticmethod
    def correct_args(arg):
        to_del = [k for k in arg.keys() if k[0]+k[-1]=='<>']
        for key in to_del:
            del arg[key]
        if 'kwargs' in arg.keys():
            kwargs = arg['kwargs']
            if kwargs is not None and kwargs != inspect._empty:
                assert isinstance(kwargs, dict), type(kwargs)
                for k,v in arg['kwargs'].items():
                    arg[k] = v
            del arg['kwargs']
        return arg

