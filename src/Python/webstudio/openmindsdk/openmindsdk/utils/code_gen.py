import inspect
from collections import OrderedDict
import sys
from typing import TextIO, List, Any
from .code_visualize import show_graph
from .io_utils import safe_open
from .config_utils import nested_merge
from .importing import f


indent_str ='    '


class LineContainer:
    "the container for one line of code"
    lines = List[str]

    def __init__(self, *lines: str, indent: int=0):
        self.lines = []
        self.indent = indent
        for line in lines:
            self.add_a_line(line)

    def add_a_line(self, line: str):
        lines = line.split('\n')
        self.lines.extend([indent_str*self.indent + x for x in lines])

    def to_str(self):
        return self.lines

    def write(self, fn: TextIO):
        for line in self.to_str():
            print(line, file=fn)


class ModuleCoder:
    "a code helper to manage modules importing"

    def __init__(self):
        self.modules = dict()
        self.aliases = dict()

    def add_module(self, m: str, relative: str='', alias: str=None):
        """add import statement accoring to m
        m == xx.yy => from xx import yy
        m == xx (or xx.yy) (alias == zz) => import xx (or xx.yy) as zz
        m == xx (alias == None) => import xx (as xx)
        """
        msp = m.split('.')
        if len(msp) == 1 or alias is not None:
            alias = m if alias is None else alias
            m = relative + m
            self.aliases[alias] = m
            return
        s_module, s_class = '.'.join(msp[:-1]), msp[-1]
        s_module = relative + s_module
        self.modules.setdefault(s_module, set()).add(s_class)

    def merge_from(self, other):
        "merge from other ModuleCoder"
        nested_merge(self.modules, other.modules)
        nested_merge(self.aliases, other.aliases)

    def write(self, fn):
        "write code lines"
        for a, m in self.aliases.items():
            s = f('import {m}') if a==m else f('import {m} as {a}')
            LineContainer(s).write(fn)
        for key, value in self.modules.items():
            classes = ', '.join(value)
            LineContainer(f('from {key} import {classes}')).write(fn)


class CodeBlockGen(LineContainer):
    "code generator for a code block"

    def __init__(self, name: str, indent=0):
        super().__init__(indent=indent)
        self.name = name
        self.modules = ModuleCoder()
        self.dependencies = set()
        self.refactor_done = False

    def add_module(self, m: str, **kwargs):
        self.modules.add_module(m, **kwargs)

    def add_dependency(self, dep: str):
        "add a dependency (should also be a code block in the same file)"
        self.dependencies.add(dep)

    def merge_from(self, other, shift: bool=True):
        "push another code block here"
        self.modules.merge_from(other.modules)
        self.dependencies |= other.dependencies
        extra_indent = self.indent if shift else 0
        for line in other.to_str():
            self.add_a_line(indent_str*extra_indent + line)

    def refactor(self):
        """some lines can only be generated after the whole block finished
        such as add blank line before and after the block
        """
        self.lines.insert(0,'')
        self.lines.append('')
        self.refactor_done = True

    def to_str(self):
        if not self.refactor_done:
            self.refactor()
        return super().to_str()


class FuncCoder(CodeBlockGen):
    "code generator for a function"

    def __init__(self, name: str, code_type='func', args: list=[], returns: list=[], indent=0):
        super().__init__(name, indent+1)
        self.code_type = code_type
        self.func_args = []
        order = 0
        for item in args:
            self.func_args.append({'name': item, 'order': order})
            order = order + 1
        self.func_returns = [] + returns

    def __append_to_attribute(self, attr: str, element, order=0):
        a = getattr(self, attr)
        assert isinstance(a, list), (attr, a)
        if isinstance(element, list):
            for item in element:
                a.append({'name': item, 'order': order})
        else:
            a.append({'name': element, 'order': order})
        setattr(self, attr, a)

    def add_func_args(self, a, order=0):
        self.__append_to_attribute('func_args', a, order)

    def add_func_returns(self, r):
        self.__append_to_attribute('func_returns', r)

    def refactor(self):
        self.func_args = sorted(self.func_args, key=lambda x: x['order'], reverse=False)
        args = []
        for item in self.func_args:
            args.append(item['name'])
        args = ', '.join(args)
        if self.code_type == 'func':
            self.lines.insert(0, LineContainer(f('def {self.name} ({args}):'), indent=self.indent-1).to_str()[0])
        elif self.code_type == 'class':
            self.lines.insert(0, LineContainer(f('class {self.name} ({args}):'), indent=self.indent - 1).to_str()[0])
        super().refactor()



class ClassCoder(CodeBlockGen):
    """
    code generator for a class
    """

    def __init__(self, name: str, indent=0, bases: set=set()):
        super().__init__(name, indent)
        self.bases = set() | bases
        self.methods = OrderedDict()

    def add_a_base(self, base: str):
        self.bases.add(base.split('.')[-1])
        self.modules.add_module(base)

    def add_a_method(self, name: str, **kwargs):
        return self.methods.setdefault(name, FuncCoder(name, indent=self.indent+1, **kwargs))

    def refactor(self):
        if len(self.bases) > 0:
            bases = ', '.join(self.bases)
            self.add_a_line(f('class {self.name} ({bases}):'))
        else:
            self.add_a_line(f('class {self.name}:'))
        for name_, method_ in self.methods.items():
            self.merge_from(method_)


class SourceCoder:
    "source code file generation"

    def __init__(self):
        self.modules = ModuleCoder()
        self.codeblocks = OrderedDict()

    def add_codeblock(self, name, type_=CodeBlockGen, **kwargs):
        return self.codeblocks.setdefault(name, type_(name, **kwargs))

    def write_to_file(self, filename: str):
        def __next_write(codeblocks:dict, flag_written:dict):
            writable = []
            for cb_name, cb in codeblocks.items():
                if flag_written[cb_name]:
                    continue
                flags = [flag_written[k] for k in cb.dependencies]
                if all(flags):
                    writable.append(cb_name)
            return writable

        with safe_open(filename, 'w') as fn:
            # write module
            for cb in self.codeblocks.values():
                self.modules.merge_from(cb.modules)
            self.modules.write(fn)
            # write code blocks
            flag_written = {k: False for k in self.codeblocks.keys()}
            while True:
                to_write = __next_write(self.codeblocks, flag_written)
                if len(to_write) == 0:
                    break
                for name_ in to_write:
                    self.codeblocks[name_].write(fn)
                    flag_written[name_] = True
            assert all(flag_written), 'function call loops in {}'.format([k for k, v in flag_written.items() if not v])


class ModelCoder:
    "code generator (structure) of a model"

    def __init__(self, name: str, container=FuncCoder):
        self.container = container(name=name)
        self.inputs = OrderedDict()  # {variable-name: instance-name}
        self.outputs = []  # model outputs
        for name, thing in inspect.getmembers(self.container):
            if not hasattr(self, name):
                setattr(self, name, thing)

