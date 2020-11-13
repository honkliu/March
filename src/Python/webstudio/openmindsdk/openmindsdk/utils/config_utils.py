import yaml
import inspect

from collections import OrderedDict
from .io_utils import from_file, to_file
from .importing import f


def mergable(a, b):
    """
    if b can be merged into a, do it and return True; otherwise return False
    :param a:
    :param b:
    :return: True / False
    """
    if type(a) != type(b):
        return False
    if isinstance(a, list):
        a += b
        return True
    if isinstance(a, set):
        a |= b
        return True
    return False


def nested_merge(a: dict, b: dict, path=None, conflict_resolve='take_b'):
    """
    merge b into a recursively
    ref: https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge

    :param a:
    :param b:
    :param path:
    :param conflict_resolve: conflict_resolve can be 'take_a', 'take_b', 'exception'
    :return:
    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                nested_merge(a[key], b[key], path + [str(key)])
            elif mergable(a[key], b[key]):
                pass
            elif a[key] == b[key] or conflict_resolve == 'take_a':
                pass  # same leaf value
            elif conflict_resolve == 'take_b':
                a[key] = b[key]
            elif conflict_resolve == 'exception':
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]


def __is_nested(keys):
    assert isinstance(keys, (list, tuple)), f('unsupported keys {keys}')


def nested_set(dic: dict, keys: (list, tuple), value):
    """
    d = {}
    nested_set(d, ['person', 'address', 'city'], 'New York')
    print(d) => {'person': {'address': {'city': 'New York'}}}
    ref https://stackoverflow.com/questions/13687924/setting-a-value-in-a-nested-python-dictionary-given-a-list-of-indices-and-value

    :param dic:
    :param keys:
    :param value:
    :return: None
    """
    __is_nested(keys)
    for k in keys[:-1]:
        if k not in dic.keys() or dic[k] is None:
            dic[k] = dict()
        dic = dic[k]
        assert(isinstance(dic, dict))
    dic[keys[-1]] = value


def nested_get(dic: dict, keys: (list, tuple)):
    """
    get the element of a nested dict with given key path

    :param dic:
    :param keys: key path
    :return: dic [k0] [k1] ... [k[-1]]
    """
    __is_nested(keys)
    result = dic
    for a in keys:
        result = result[a]
    return result


class Configurable:
    """
    the container of a nested dictionary (or Ordered dictionary)
    """

    def __init__(self, config={}, container=OrderedDict):
        """

        :param config: dictionary to be merged into
        :param container: dict or OrderedDict
        """
        assert isinstance(config, dict), f('only support dict input, but {type(config)} got')
        self.container_type = container
        self.__declare__()
        self.__customize__(config)

    def __declare__(self):
        """
        to declare the initial (default) parameters here
        :return: None
        """
        if not hasattr(self, 'data'):
            self.data = self.container_type()

    def __customize__(self, config):
        """
        merge with given config and do post processing here
        :param config:
        :return: None
        """
        nested_merge(self.data, config)

    def to_file(self, filename: str):
        """
        dump to file, the output format depends on the file extension
        :param filename: only support json or yaml format
        :return:
        """
        to_file(self.data, filename)
        
    def list_config(self):
        return self.data

    # interface to set
    def set(self, key, value):
        """
        normal dictionary like set
        :param key:
        :param value:
        :return: None
        """
        self.data[key] = value

    def batch_set(self, key_names: list, values: list=[], default=None):
        """
        ditionary like set applied to multiple key-value
        :param key_names: list of keys
        :param values: list of values
        :param default: if value not specified
        :return: None
        """
        for i, key in enumerate(key_names):
            self.data[key] = values[i] if i < len(values) else default

    def nested_set(self, keys: list, value):
        """
        nested est
        :param keys:
        :param value:
        :return:
        """
        nested_set(self.data, keys, value)

    def update(self, other: dict):
        """
        merge with other
        :param other:
        :return:
        """
        nested_merge(self.data, other)

    # interface to test
    def has(self, *keys):
        """ test whether a key path is included (don't care whether value is defined)

        :param keys: key path of a nested dict
        :return: True or False
        """
        try:
            p = nested_get(self.data, keys)
            return True
        except KeyError:
            return False
        except Exception as e:
            raise Exception(e)

    def defined(self, *keys, undef_vals: list=[None]):
        """
        has(keys) and get(keys) not in undef_vals
        :param keys:
        :return:
        """
        return self.has(*keys) and self.get(*keys) not in undef_vals

    def set_if_undefined(self, *keys, value):
        """
        if not defined, set to value
        :param keys:
        :param value:
        :return:
        """
        if not self.defined(*keys):
            self.nested_set(keys, value)

    # interface to get
    def get(self, *keys):
        """
        like dict[key], raise KeyError if key not exists
        :param keys:
        :param default:
        :return:
        """
        return nested_get(self.data, keys)

    @staticmethod
    def factory(DerivedClass: str, BaseClass, config_dict, attr_dict):
        """
        the factory to generate a derived class from a given base class
        :param DerivedClass: derived class name
        :param BaseClass:
        :param config_dict:
        :param attr_dict:
        :return:
        """
        def __declare__(self):
            # inherit variables in BaseClass
            super().__declare__(self)
            # static parameter changes
            self.update(config_dict)
            for key, value in attr_dict.items():
                setattr(self, key, value)
        newclass = type(DerivedClass, (BaseClass,),{"__declare__": __declare__})
        return newclass