"""
{
  "SourceName": "Keras-Common-Supervisors",
  "Url": "https://blog.keras.io/",
  "Description": "Common supervisors such as MLP or LSTM implemented in Keras."
}
"""

from ..utils import ModelBrick, Launcher, ModelCoder, Brick, FuncCoder, ModelBrickWrapper
from .utils_bricks import SubGraphIn, SubGraphOut
from ...utils import Callable, f

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
import os
import json


class KerasModelInput(SubGraphIn):
    """The start point of a model"""
    FILENAME = __file__
    FRAMEWORK = 'Keras'
    BASEMODEL = 'keras.layers.Input'

    def declare_variable(self, caller: ModelCoder):
        x, x_shape = self.get('variable-name'), self.get('x-shape')
        assert isinstance(x_shape, (list, tuple)), "x-shape should be list or tuple, now is {}".format(type(x_shape).__name__)
        if self.get('shape-include-batch-size'):
            caller.add_a_line('{} = Input(batch_shape={})'.format(x, x_shape))
        else:
            caller.add_a_line('{} = Input(shape={})'.format(x, x_shape))


class KerasModelOutput(SubGraphOut):
    "The end point of a model"
    FILENAME = __file__
    FRAMEWORK = 'Keras'
    BASEMODEL = 'keras.models.Model'

    def __declare__(self):
        super().__declare__()
        self.set('is-subgraph-output', False)

    def generate_model_wrapper(self, gen: ModelCoder, func_name: str='get_model'):
        caller = FuncCoder(func_name)
        caller.add_dependency(gen.name)

        for x, inst_name in gen.inputs.items():
            self.parent.find(inst_name).declare_variable(caller)

        in_, out_ = ', '.join(gen.inputs.keys()), ', '.join(gen.outputs)
        caller.add_a_line('{} = {} ({})'.format(
            out_, gen.name, ', '.join([x+'='+x for x in gen.inputs.keys()])
        ))
        caller.add_a_line('return Model([%s], [%s])' % (in_, out_))
        return caller

    def generate_model_summary(self, gen: ModelCoder, func_name: str='get_model'):
        fn = FuncCoder('model_summary')
        fn.add_dependency(gen.name)
        fn.add_func_args('filename')

        codes = f('''#generate summary to file
model = {func_name}()
with open(filename, 'w') as fn:
    model.summary(print_fn=lambda x: fn.write(x+"\\n"))        
''')
        fn.add_a_line(codes)
        return fn

    def generate_model_callback(self, gen: ModelCoder, func_name: str='get_model'):
        classFn = FuncCoder('SendMetrics', code_type='class')
        classFn.add_func_args('Callback')
        codes = f('''#Keras callback to send metrics to NNI framework
def on_epoch_end(self, epoch, logs):
    #Run on end of each epoch
    nni.report_intermediate_result(logs["val_acc"])
''')
        classFn.add_a_line(codes)
        return classFn

    def generate_default_params(self, gen: ModelCoder, model_parameters, func_name: str='get_model'):
        fn = FuncCoder('generate_default_params')
        default_params = {
            "batch_size": 32,
            "epochs": 5,
            "loss": "categorical_crossentropy",
            "optimizer": "SGD",
            "schedule_decay": 0.004,
            "beta_1": 0.9,
            "decay": 0.,
            "rho": 0.9,
            "beta_2": 0.999,
            "lr": 0.001,
            "momentum": 0.9,
        }
        default_params['batch_size'] = int(model_parameters['batch_size'])
        default_params['epochs'] = int(model_parameters['epochs'])
        default_params['loss'] = model_parameters['loss']
        default_params['optimizer'] = model_parameters['optimizer']
        default_params['optimizer_params'] = model_parameters['optimizer_params']
        for key in model_parameters['optimizer_params']:
            default_params[key] = float(model_parameters['optimizer_params'][key])
        codes = f('''return {default_params}''')
        fn.add_a_line(codes)
        return fn

    def generate_get_data(self, model_parameters):
        if 'dataset' in model_parameters:
            dataset = model_parameters["dataset"]
            fn = FuncCoder('get_data')
            if dataset == 'custom':
                fn.add_a_line('#add your code to get data, return (x_train, y_train), (x_test, y_test)')
                fn.add_a_line('pass')
                return fn
            fn.add_module('keras.utils.to_categorical')
            fn.add_a_line('# get public data')
            if dataset != 'custom':
                fn.add_module('keras.datasets.%s.load_data' % dataset)
            # todo: add reuters data process logic
            if dataset == 'mnist' or dataset == 'fashion_mnist':
                self.add_data_process(fn, 28, 28, 1, 10)
            elif dataset == 'cifar10':
                self.add_data_process(fn, 32, 32, 3, 10)
            elif dataset == 'cifar100':
                self.add_data_process(fn, 32, 32, 3, 100)
            elif dataset == 'imdb':
                fn.add_a_line('(x_train, y_train), (x_test, y_test) = load_data(num_words=20000)')
                fn.add_module('keras.preprocessing.sequence')
                fn.add_a_line('x_train = sequence.pad_sequences(x_train, 80)')
                fn.add_a_line('x_test = sequence.pad_sequences(x_test, 80)')
            elif dataset == 'boston_housing':
                fn.add_a_line('(x_train, y_train), (x_test, y_test) = load_data()')
            fn.add_a_line('return (x_train, y_train), (x_test, y_test)')
            return fn
        return None

    def add_data_process(self, fn: FuncCoder, img_row, img_col, channels, num_classes):
        fn.add_a_line('(x_train, y_train), (x_test, y_test) = load_data()')
        fn.add_a_line('x_train = x_train.reshape(x_train.shape[0], %d, %d, %d)' % (img_row, img_col, channels))
        fn.add_a_line('x_test = x_test.reshape(x_test.shape[0], %d, %d, %d)' % (img_row, img_col, channels))
        fn.add_a_line('x_train = x_train.astype(\'float32\')')
        fn.add_a_line('x_test = x_test.astype(\'float32\')')
        fn.add_a_line('x_train /= 255')
        fn.add_a_line('x_test /= 255')
        fn.add_a_line('y_train = to_categorical(y_train, %d)' % num_classes)
        fn.add_a_line('y_test = to_categorical(y_test, %d)' % num_classes)

class KerasModelCustomLayer(ModelBrick):
    FILENAME = __file__
    FRAMEWORK = 'Keras'

    def __declare__(self):
        super().__declare__()
        self.set('trainable', True)
        self.set('code', """from keras import backend as K
from keras.engine.topology import Layer
from keras import activations
class MyLayer(Layer):

    def __init__(self, output_dim, activation=None, use_bias=True, **kwargs):
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        name='bias', initializer='zeros')
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        output = K.dot(x, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
         """)
        self.set('parameter', '{}')
        self.set('output-labels', [])
        self.set('input-labels', [])
        self.add_fe_parameters("code")
        self.add_fe_parameters("parameter")

    def generate_code_inner(self, gen: ModelCoder):
        gen.add_custom_code(self.get("instance-name"), self.get("code"))
        exec(self.get('code'))


class KerasModelComponent(ModelBrickWrapper):
    """Basic wrapping of keras layers / sub-models"""
    FILENAME = None
    FRAMEWORK = 'Keras'
    SHOW_IN_GALLERY = False
    BASEMODEL = None

    def __declare__(self):
        super().__declare__()
        self.set('trainable', True)
        self.set('require-input', False)
        self.set('allow-input-merge', self.get_group() == 'merge')

    def generate_code_inner(self, gen: ModelCoder):
        x = self.get_inputs()
        gen.add_module(self.BASEMODEL)
        arg = self.get('basemodel', 'argdict')
        if self.get('require-input'):
            arg['input_tensor'] = x
        f = self.variable('_')
        gen.add_a_line('%s = %s' %(f, self.bmodel_gen.code(arg, keys_to_eval=['input_tensor'])))

        if not self.get('trainable'):
            gen.add_a_line('%s.trainable = False' % (f))
        out_ = ', '.join([self.variable(x) for x in self.get('output-labels')])
        gen.add_a_line('{} = {} ({})'.format(out_, f, x))



class KerasModelReference(ModelBrick):
    FILENAME = __file__
    SHOW_IN_GALLERY = True
    FRAMEWORK = 'Keras'

    def __declare__(self):
        super().__declare__()
        self.set('trainable', True)
        self.set('reference', '')
        self.set('layer-class', '')
        self.set('parameter-dict', '{}')
        self.add_fe_parameters('reference')
        self.add_fe_parameters('parameter-dict')
        self.add_fe_parameters('layer-class')

    def generate_code_inner(self, gen: ModelCoder):
        custom_layer = self.parent.find_instance(self.get('reference'))
        assert custom_layer is not None
        x = self.get_inputs()
        custom_layer.generate_code(gen)
        f = self.variable('_')
        print(self.get('layer-class'))
        print(self.get('parameter-dict'))
        arg_dict = self.get('parameter-dict')
        assert type(arg_dict) == dict
        arg_list = []
        for k, v in arg_dict.items():
            if type(v) == str:
                arg_list.append(k + ' = "' + v.__str__() + '"')
            else:
                arg_list.append(k + ' = ' + v.__str__())
        gen.add_a_line('%s = %s(%s)' % (f, self.get('layer-class'), ', '.join(arg_list)))
        out_ = ', '.join([self.variable(x) for x in self.get('output-labels')])
        gen.add_a_line('%s = %s (%s)' % (out_, f, x))
        super().generate_code_inner(gen)


class KerasOptimizerBase(Brick):
    """Basic wrapping of keras optimizers"""
    FILENAME = __file__
    FRAMEWORK = 'Keras'
    SHOW_IN_GALLERY = False
    BASEMODEL = None

    def __declare__(self):
        super().__declare__()

        self.set('role', 'optimizer')
        bmodel = Callable(self.BASEMODEL)
        self.bmodel_gen = bmodel
        self.set('basemodel', bmodel.list_config())

        # compiler
        loss_list = ['hinge', 'binary_crossentropy', "categorical_crossentropy", "mean_suqared_error"]
        self.set('loss', 'categorical_crossentropy')
        self.set("metrics", ["accuracy"])

        self.add_fe_parameters("basemodel::argdict::*", 'metrics')
        self.add_fe_parameters('loss', type_="select", enum_values = loss_list)

    def get_group(self):
        return 'optimizers'

    def compile(self, model):
        opt_param = self.get('basemodel', 'argdict')
        optimizer = self.bmodel_gen.call(opt_param)
        compile_param = {
            'optimizer': optimizer,
            'loss': self.get('loss'),
            'metrics': self.get('metrics'),
        }
        # here is dealing with a bug, if frontend fixs, remvoe the lines
        if isinstance(compile_param['loss'], list):
            compile_param['loss'] = 'categorical_crossentropy'

        model.compile(**compile_param)
        return model

for bmodel_name in ['Adadelta', 'Adagrad', 'Adam', 'Adamax','Nadam', 'RMSprop', 'SGD']:
    dmodel_name = 'KerasOpt' + bmodel_name
    globals()[dmodel_name] = type(
        dmodel_name, (KerasOptimizerBase,), {
            "BASEMODEL": 'keras.optimizers.{}'.format(bmodel_name),
            "SHOW_IN_GALLERY": True
        }
    )


class KerasCallBackBase(Brick):
    """Basic wrapping of training callbacks"""
    FILENAME = __file__
    FRAMEWORK = 'Keras'
    SHOW_IN_GALLERY = False
    BASEMODEL = None

    def __declare__(self):
        super().__declare__()

        self.set('role', 'training-callback')
        bmodel = Callable(self.BASEMODEL)
        self.bmodel_gen = bmodel
        self.set('basemodel', bmodel.list_config())
        self.add_fe_parameters("basemodel::argdict::*")

    def get_group(self):
        return 'callbacks'

    def get_callback(self):
        cbargs = self.get('basemodel', 'argdict')
        if 'filename' in cbargs:
            cbargs['filename'] = self.parent.node.register(cbargs['filename'])
        if 'filepath' in cbargs:
            cbargs['filepath'] = self.parent.node.register(cbargs['filepath'])
        return self.bmodel_gen.call(cbargs)

for bmodel_name in ['CSVLogger', 'EarlyStopping', 'ReduceLROnPlateau']:
    dmodel_name = 'KerasCb' + bmodel_name
    globals()[dmodel_name] = type(
        dmodel_name, (KerasCallBackBase,), {
            "BASEMODEL": 'keras.callbacks.{}'.format(bmodel_name),
            "SHOW_IN_GALLERY": True
        }
    )


class KerasModelLauncher(Launcher):
    FRAMEWORK = 'Keras'
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()

        # trainer
        trainer_list = ['fit_generator']
        for t in trainer_list:
            c = Callable(name=t, func=getattr(Sequential(), t))
            self.nested_set(['trainer-parameter-schema', t], c.list_config()['argdict'])
        self.set('trainer-selector', 'fit_generator')
        self.set('trainer-parameter', {"epochs": 2, 'steps_per_epoch': 200})

        self.add_fe_parameters("trainer-parameter::*")


    def train(self, model, get_model_func):
        "now only support fit_generator"
        callbacks = []
        for cb in self.parent.find_role('training-callback'):
            callbacks.append(cb.get_callback())

        self.parent.notify('Running', 'Training starts ...')
        reader_tr = self.parent.find_role(role='data-reader', unique=True)
        trainer_param = self.get('trainer-parameter')
        trainer_param.update({
            'generator': reader_tr.data_gen,
            'validation_data': reader_tr.valid_data_gen,
            'validation_steps': reader_tr.valid_data_gen.samples,
            'callbacks': callbacks,
            'verbose': 2,
        })
        print(trainer_param)
        hist = model.fit_generator(**trainer_param).history
        accu = hist['val_acc'][-1]
        return {'loss': -accu, 'attachments': hist}
