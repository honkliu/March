# coding:utf-8
import copy
import sys
from ast import literal_eval
from openmindsdk.utils import Configurable
import json
import os


###############################################################################
# Frontend to Backend
###############################################################################
'''
parse the operators json, return a dict contains the info of the operators
the result key is the id of the operators
the result value is the info of this operator include: type, id, title,
'''

def parse_operators(operators: dict):
    operatorsInfo = {}
    instance_cnt_map = {}
    for (key, value) in operators.items():
        if not isinstance(value, dict):
            continue
        if not (('internal' in value)
                and isinstance(value['internal'], dict)
                and ('properties' in value['internal'])
                and isinstance(value['internal']['properties'], dict)
                and ('title' in value['internal']['properties'])):
            continue

        info = {}
        info['id'] = key
        info['type'] = value['internal']['properties']["class_name"]
        info['title'] = value['internal']['properties']['title']
        model_type = info['type'].split('.')[-1][:10]
        if model_type=='Tensorflow' or model_type=="SubGraphSi":
            if info['title'] not in instance_cnt_map:
                instance_cnt_map[info['title']] = 1
                info['instance-name'] = info['title']
            else:
                print("Two or more instances are named '%s'." % info['title'])
                Exception()
        else:
            instance_cnt_map[info['title']] = 1 if info['title'] not in instance_cnt_map else instance_cnt_map[info['title']] + 1
            info['instance-name'] = info['title'] + str(instance_cnt_map[info['title']])
        operatorsInfo[key] = info

    return operatorsInfo


'''
parse the operators json, return a dict contains the info of the operators
the result key is the id of the operators
the result value is the info of this operator include: type, id, title,
'''

def parse_operators_gojs(operators: list):
    operatorsDict = {}
    instance_cnt_map = {}
    for item in operators:
        if not isinstance(item, dict):
            continue
        if not 'key' in item.keys():
            continue
        if item['title'] not in instance_cnt_map:
            instance_cnt_map[item['title']] = 1
        else:
            instance_cnt_map[item['title']]= instance_cnt_map[item['title']] + 1
        #item['instance-name'] = item['title'] + str(instance_cnt_map[item['title']])
        item['instance-name'] = item['title']
        operatorsDict[str(item['key'])]= item
    return operatorsDict

'''
parse the parameters of the operator result is a dict
the key is operator id
value is a dict contains the parameters of this operator
'''


def parse_parameters(parameters: dict):
    parametersInfo = {}

    for (key, value) in parameters.items():
        if (not isinstance(value, dict)) or (not value):
            continue

        info = {}

        for (valueKey, valueValue) in value.items():
            if not isinstance(valueValue, dict):
                info[valueKey] = valueValue
            else:
                '''if the valueValue is a dict, it is a checkboxes or radios than convert it to be a list'''
                parameters_list = []

                for (k, v) in valueValue.items():
                    if isinstance(v, bool) and True == v:
                        parameters_list.append(k)

                if len(parameters_list):
                    info[valueKey] = parameters_list

        parametersInfo[key] = info

    return parametersInfo


'''
parse the link information
the key is the operator's id
the value is a list contains the from operator info
'''


def parse_links(links: dict):
    linksInfo = {}
    for value in links.values():
        if (not isinstance(value, dict)) \
                or (not 'toOperator' in value) \
                or (not 'fromOperator' in value) \
                or (not 'fromConnector' in value) \
                or (not 'toConnector' in value): \
                continue

        toOperator = str(value['toOperator'])

        if toOperator not in linksInfo:
            linksInfo[toOperator] = []

        linksInfo[toOperator].append({'fromOperator': str(value['fromOperator']),
                                      'fromConnector': str(value['fromConnector']),
                                      'toConnector': str(value['toConnector'])})

    return linksInfo

'''
parse the link information from GoJS version
the key is the operator's id
the value is a list contains the from operator info
'''

def parse_links_gojs(links: list):
    linksInfo = {}
    for link in links:
        if (not isinstance(link, dict)) \
                or (not 'to' in link) \
                or (not 'from' in link) \
                or (not 'fromPort' in link) \
                or (not 'toPort' in link): \
                continue

        toOperator = str(link['to'])

        if toOperator not in linksInfo:
            linksInfo[toOperator] = []

        linksInfo[toOperator].append({'fromOperator': str(link['from']),
                                      'fromPort': str(link['fromPort']),
                                      'toPort': str(link['toPort'])})
    for key in linksInfo.keys():
        linksInfo[key] = sorted(linksInfo[key], key=lambda x: x['toPort'], reverse=False)

    return linksInfo
'''
this function convert the FE json file to another format that the BE can read
return None if the FE Json is not correct
'''


def convert_fe_json_to_be_json(feJson: dict):
    '''the id and title of the Model file'''
    id = None
    title = None

    '''the process contains the info of the Model graph'''
    process = None

    '''the operators the build by the FE'''
    operators = None

    '''the parameters of the operators'''
    parameters = None

    '''the links info of the operators'''
    links = None

    if 'title' in feJson.keys():
        title = feJson['title']

    if 'id' in feJson.keys():
        id = feJson['id']

    process = feJson['process']
    assert isinstance(process, dict)

    operatorsInfo = parse_operators(process['operators'])

    parametersInfo = parse_parameters(process['parameters'])

    linksInfo = parse_links(process['links'])

    if (operatorsInfo is None) or (not operatorsInfo):
        return None

    retJson = []

    for (id, info) in operatorsInfo.items():
        brick = {}

        '''brick info'''
        brick['type'] = info['type']
        brick['instance-name'] = info['instance-name']

        '''links info'''
        if id in linksInfo:
            edgeInfo = {}
            for edge in linksInfo[id]:
                if edge["fromOperator"] in operatorsInfo:
                    edgeInfo.setdefault(edge['toConnector'], []).append(
                        [operatorsInfo[edge["fromOperator"]]["instance-name"], edge['fromConnector']]
                    )
            if 0 != len(edgeInfo):
                brick['inputs'] = edgeInfo

        '''parameters info'''
        brick['parameters'] = {}
        if id in parametersInfo:
            for (k, v) in parametersInfo[id].items():
                brick['parameters'][k] = v

        retJson.append(brick)

    '''return the result'''
    return retJson

def convert_fe_json_to_be_json_gojs(feJson: dict):
    '''the id and title of the Model file'''
    title = None

    '''the operators the build by the FE'''
    operators = None

    '''the parameters of the operators'''
    parameters = None

    '''the links info of the operators'''
    links = None

    if 'class' in feJson.keys():
        title = feJson['class']

    if ('nodeDataArray' not in feJson.keys()) or ('linkDataArray' not in feJson.keys()):
        return None

    operatorsInfo = parse_operators_gojs(feJson['nodeDataArray'])
    linksInfo = parse_links_gojs(feJson['linkDataArray'])
    retJson = []

    for key in operatorsInfo:
        brick = {}
        operator = operatorsInfo[key]

        '''brick info'''
        # TODO
        brick['key'] = operator['key']
        brick['layer'] = operator['type']
        brick['type'] = operator['class_name']
        brick['instance-name'] = operator['instance-name']


        '''links info'''
        if key in linksInfo.keys():
            edgeInfo = {}
            for edge in linksInfo[key]:
                if edge["fromOperator"] in operatorsInfo:
                    edgeInfo.setdefault("in-port-0", []).append(
                        [operatorsInfo[edge["fromOperator"]]["instance-name"], "out-port-0"]
                    )
            if 0 != len(edgeInfo):
                brick['inputs'] = edgeInfo

        '''parameters info'''
        brick['parameters'] = operator["parameters"]


        retJson.append(brick)

    '''return the result'''
    return retJson



###############################################################################
# Backend to Frontend
###############################################################################
def get_arg_type(arg_val):
    arg_type = 'str'
    if arg_val is not None:
        arg_type = arg_val.__class__.__name__
    return arg_type


def set_default_val(para_dict, arg_type, arg_val, enum_values = None):
    if arg_type not in ['select', 'list']:
        arg_val = arg_val.__str__()
    if arg_type in ["float", "int", "str"]:
        para_dict["config"]["default"] = arg_val
    elif arg_type in ["bool"]:
        para_dict["config"]["default"]["value"] = arg_val
        para_dict["config"]["options"][0]["value"] = arg_val
    elif arg_type in ["list"]:
        assert(len(arg_val) > 0)
        para_dict["config"]["fieldTypeConfig"]["default"] = arg_val[0].__str__() # convert dict to str
    elif arg_type in ["select"]:
        assert isinstance(enum_values, list) and len(enum_values) > 0
        assert arg_val in enum_values, (arg_val, enum_values)
        para_dict["config"]["options"] = [{"value": item, "label": item} for item in enum_values]
        para_dict["config"]["default"] = arg_val


config_type_mapping = {
    "float": {
        "id": "number",
        "label": "Number:",
        "type": "ultiflow::number",
        "config": {
            "default": 0
        }
    },
    "int": {
        "id": "number",
        "label": "Number:",
        "type": "ultiflow::number",
        "config": {
            "default": 0
        }
    },
    "str": {
        "id": "single_line_text",
        "label": "Text example:",
        "type": "ultiflow::text",
        "config": {
            "default": ""
        }
    },
    "code": {
        "id": "textarea",
        "label": "Textarea (modified attr):",
        "type": "ultiflow::textarea",
        "config": {
            "attr": {
                "rows": 10
            },
            "default": ""
        }
    },
    "bool": {
        "id": "choices_checkboxes",
        "label": "Choices (checkboxes):",
        "type": "ultiflow::choices",
        "config": {
            "options": [
                {"value": "value", "label": "Enable"}
            ],
            "type": "checkbox",
            "default": {
                "value": True
            }
        }
    },
    "list": {
        "id": "list",
        "label": "List:",
        "type": "ultiflow::list",
        "config": {
            "fieldType": "ultiflow::text",
            "fieldTypeConfig": {
                "default": ""
            }
        }
    },
    "tuple": {
        "id": "list",
        "label": "List:",
        "type": "ultiflow::list",
        "config": {
            "fieldType": "ultiflow::text",
            "fieldTypeConfig": {
                "default": ""
            }
        }
    },
    "select": {
        "id": "select",
        "label": "Select:",
        "type": "ultiflow::select",
        "config": {
            "options": [
            ],
            "default": "first"
        }
    }
}


###############################################################################
# Frontend to Backend
###############################################################################
class UltiFlowBrick(Configurable):
    """
    Block container to deal with UltiFlow GUI
    """

    def __declare__(self):
        """
        define GUI-related parameters
        :return:
        """
        super().__declare__()
        self.set('input-labels', ['in'])
        self.set('output-labels', ['out'])
        self.set('inputs', {})
        self.set('allow-input-merge', False)
        self.fe_parameters = []  # [{source:, prefix:, type:}]

    def add_fe_parameters(self, *be_names: str, format_: str="{}", type_: str='str', enum_values: list=None):
        """
        add necessary information to display a UltiFlow operator
        :param be_names: backend keys path, pattened in a::b -- a::b::* will add all parameters in the path a::b
        :param format_: how GUI show the label
        :param type_: str, list, select
        :param enum_values: if type_==select, the possible values
        :return:
        """
        for be_name in be_names:
            assert isinstance(be_name, str), be_name
            tokens = be_name.split('::')
            if tokens[-1] == '*':
                for name in self.get(*tokens[:-1]).keys():
                    real_name = '::'.join(list(tokens[:-1]) + [name])
                    self.add_fe_parameters(real_name, format_=format_, type_=type_, enum_values=enum_values)
                continue
            operator_config = {
                'be_name': be_name,
                'fe_name': format_.format(tokens[-1]),
                'type': type_,
                'default': self.get(*tokens),
                'enums': enum_values
            }
            if type_ == 'select':
                assert enum_values is not None, 'please specify the enum values'
                assert self.get(*tokens) in enum_values, (enum_values, be_name, tokens, self.get(*tokens))
            self.fe_parameters.append(operator_config)

    def register_fe_param(self, be_name: str, default_, **kwargs):
        """
        add (if not exist) a parameter (if nested, put together in the pattern a::b) and register to the fe_parameters
        :param be_name:
        :param default_:
        :param kwargs: arguments in method add_fe_parameters
        :return:
        """
        "add a parameter that will be shown in the GUI"
        be_pth = be_name.split('::')
        self.nested_set(keys=be_pth, value=default_)
        self.add_fe_parameters(be_name, **kwargs)

    def dump_parameters_forfe(self):
        """
        dump the operator config for all elements in fe_parameters
        :return:
        """
        try:
            paras = []
            fe_keys = ["code"]
            for cfg in self.fe_parameters:
                be_name, fe_name, arg_type, arg_val = cfg['be_name'], cfg['fe_name'], cfg['type'], cfg['default']
                if arg_type not in config_type_mapping and fe_name not in fe_keys:
                    print("%s is not in config type mapping" % arg_type)
                    continue
                key = fe_name if fe_name in fe_keys else arg_type
                para_item = copy.deepcopy(config_type_mapping[key])
                if arg_val is not None:
                    set_default_val(para_item, arg_type, arg_val, cfg['enums'])
                para_item["id"] = fe_name
                para_item["label"] = fe_name
                paras.append(para_item)
            return paras
        except Exception as e:
            print(self.list_config())
            raise Exception(e)

    def list_config_forfe(self):
        """
        output the operators to disk
        :return:
        """
        config = {}
        name = self.get_display_name()
        config["id"] = '{}::{}::{}'.format(self.FRAMEWORK, self.get_group(), name)
        config["title"] = name
        config['type'] = 'operator'
        config['inputs'] = {k:{"label":k} for k in self.get('input-labels')}
        config['outputs'] = {k:{"label":k} for k in self.get('output-labels')}
        config['parameters'] = self.dump_parameters_forfe()
        config['class_name'] = self.get('type')
        config['framework'] = self.FRAMEWORK
        config['class'] = self.get("type").split(".")[-2]
        return config

    def update_from_fe(self, fe_dict: dict):
        """
        update user specified values from GUI
        :param fe_dict: config from GUI to BE
        :return:
        """
        if fe_dict == None or len(fe_dict) == 0:
            return

        isUnMatch = False
        cur_params = []
        for cfg in self.fe_parameters:
            be_name, fe_name, _type = cfg['be_name'], cfg['fe_name'], cfg['type']
            cur_params.append(fe_name)
            if fe_name not in fe_dict:
                isUnMatch = True
                continue
            value = fe_dict[fe_name]
            if value is None or value == '':
                continue
            try:
                if type(value) == list or type(value) == tuple:
                    value = [literal_eval(x) for x in value]
                value = literal_eval(value)
            except:
                pass
            self.nested_set(be_name.split('::'), value)
            self.__customize__({})
        for in_params in fe_dict:
            if in_params not in cur_params:
                isUnMatch = True

        if isUnMatch:
            print("WARN: {}'s parameter unmatch current version.".format(self.get_display_name()), file = sys.stderr)

    def get_display_name(self):
        raise NotImplementedError

    def get_group(self):
        raise NotImplementedError

    def get_display_group(self, group=None):
        raise NotImplementedError



