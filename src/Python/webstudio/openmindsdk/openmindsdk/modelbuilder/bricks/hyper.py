'''
{
  "SourceName": "automl tunor based on hyperopt or nni",
  "Url": "http://hyperopt.github.io/hyperopt/, https://github.com/Microsoft/nni",
  "Description": "An open source AutoML toolkit for neural architecture search and hyper-parameter tuning."
}
'''
from ...utils.config_utils import nested_set
from ...utils import Callable

from ..utils import Brick

from .utils_bricks import ComputeNodeLocal

from hyperopt import hp, fmin, Trials
from hyperopt import tpe, rand
import pickle
import yaml
import json


def nested_extract(dic: dict, func, result: list=None, path: list=None):
    """if func(key,value) is not None, append (path/to/item, func(key,value)) to the result"""
    if result is None:
        result = list()
    if path is None:
        path = list()
    for key, value in dic.items():
        test = func(key, value)
        if test is not None:
            result.append([path + [str(key)], test])
        if isinstance(value, dict):
            nested_extract(value, func, result, path + [str(key)])
    return result


class NeuralNetworkIntelligence(ComputeNodeLocal):
    ""
    FILENAME = __file__
    
    def __declare__(self):
        super().__declare__()
        with open('nni_default.yml') as fn:
            config = yaml.load(fn)
        self.update({'nni-config': config})
        self.set('search-space', {})
        param_list = ['experimentName', 'trialConcurrency', 'maxTrialNum']
        self.add_fe_parameters(*['nni-config::{}'.format(x) for x in param_list])
        self.add_fe_parameters('tuner::builtinTunerName', 'search-space')
        #
        self.config_file = 'nni.yml'
    
    def store_nni_config(self):
        self.nested_set(['trial', 'command'], ' '.join(self.parent.command('--nni-trial')))
        with open(self.get('nni-config', 'searchSpacePath'), 'w') as fn:
            json.dump(self.get('search-space'), fn)
        with open(self.config_file, 'w') as fn:
            yaml.dump(dict(self.list_config()), fn)
        return self.config_file
    
    def submit_training_job(self):
        pass

class HyperoptTunor(Brick):
    """Tunor read config dict, generate search space and samples of parameters
    hyperopt based parameter generator
    """
    FILENAME = __file__
    SHOW_IN_GALLERY = False

    def __declare__(self):
        super().__declare__()
        self.set('role', 'hyper-tunor')
        self.set('suggest', 'tpe')
        self.set('max-evals', 10)
        self.set('trials-store', 'trials.hyperopt')
        #
        self.search_space = None
        self.loss_func = None

    def define_search_space(self, config: dict, objective_fn):
        self.objective_fn = objective_fn
        # get tunables from dictionary
        tunables = list()
        nested_extract(config, HyperoptTunor.__tester, tunables)
        for t in tunables:
            t[0][-1] = t[0][-1][1:-1]  # remove <>
        # define search space
        choices_dict = dict()
        for t in tunables:
            label = ':'.join(t[0])
            funcname = t[1][0] if t[1][0].startswith('hyperopt.') else 'hyperopt.' + t[1][0]
            args = [label] + t[1][1:]
            nested_set(choices_dict, t[0], Callable(name=funcname).func(*args))
        choices_list = [{'model-id': key, 'hyper-parameters': value} for key, value in choices_dict.items()]
        self.search_space = hp.choice('model-selector', choices_list)

    def tune(self):
        algo = tpe.suggest
        if self.get('suggest', find_precedent=False) == 'random':
            algo = rand.suggest
        trials = Trials()  #self.restore_trials()
        self.best = fmin(
            self.objective_fn,
            space=self.search_space,
            algo=algo,
            max_evals=self.get('max-evals', find_precedent=False),
            trials=trials,
            verbose=1
        )
        return self.best
    
    def clear_trials(self):
        if self.get('trials-store') is None:
            return
        self.parent.find_role('compute-node').remove(self.get('trials-store'))
        
    def restore_trials(self):
        try:  # try to load an already saved trials object, and increase the max
            filename = self.get('trials-store')
            if filename is not None:
                trials = pickle.load(open(filename, "rb"))
                print("Found saved Trials! Loading %d trials" % (len(trials)))
        except Exception as e:  # create a new trials object and start searching
            print(e)
        return Trials()
    
    def store_trials(self):
        pass

    @staticmethod
    def __tester(key, value):
        mark_key = 'is-tunable'
        if key[0] + key[-1] == '<>' and isinstance(value, dict) and mark_key in value:
            return value[mark_key]
        return None

