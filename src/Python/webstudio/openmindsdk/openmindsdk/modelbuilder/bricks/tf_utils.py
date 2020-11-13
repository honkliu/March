from ..utils import ModelBrick, Launcher
from .keras_utils import KerasModelInput, KerasModelOutput
from ...utils import dynamic_load, Bindable, f, Callable
from ...utils import ModelCoder, FuncCoder, CodeBlockGen

import os
import shutil
import tensorflow as tf
import tensorflow.layers
import inspect


class TensorflowModelInput(KerasModelInput):
    """The start point of a model"""
    FILENAME = __file__
    FRAMEWORK = 'Tensorflow'
    BASEMODEL = 'tensorflow.keras.layers.Input'


class TensorflowModelOutput(KerasModelOutput):
    "The end point of a model"
    FILENAME = __file__
    FRAMEWORK = 'Tensorflow'
    BASEMODEL = 'tensorflow.keras.models.Model'

    def __declare__(self):
        super().__declare__()
        self.set('is-subgraph-output', False)

    def generate_model_wrapper(self, gen: ModelCoder, func_name: str='get_model'):
        caller = CodeBlockGen(func_name)
        caller.add_dependency(gen.name)
        caller.add_a_line(f('{func_name} = {gen.name}'))
        caller_keras = super().generate_model_wrapper(gen, func_name='get_model_keras')
        caller.merge_from(caller_keras)
        return caller

    def generate_model_summary(self, gen: ModelCoder):
        return super().generate_model_summary(gen, func_name='get_model_keras')


class TensorflowModelComponent(ModelBrick):
    """Basic wrapping of TensorFlow layers / sub-models"""
    FILENAME = None
    FRAMEWORK = 'Tensorflow'
    SHOW_IN_GALLERY = False
    BASEMODEL = None

    def __declare__(self):
        super().__declare__()
        bmodel = Bindable(self.BASEMODEL)
        self.bmodel_gen = Callable(self.BASEMODEL)
        for k, v in bmodel.sig.parameters.items():
            if v.default == inspect._empty or not callable(v.default):
                self.register_fe_param("basemodel::" + k, v.default if v.default is not inspect._empty else None)

    def generate_code_inner(self, gen: ModelCoder):
        x = self.get_inputs()
        gen.add_module(self.BASEMODEL)
        arg = self.get('basemodel')
        inst = self.variable('_')
        call_str = self.bmodel_gen.code(arg, keys_to_eval=['input'])
        gen.add_a_line(f('{inst} = {call_str}'))
        if not self.get('trainable'):
            gen.add_a_line(f('{inst}.trainable = False'))
        out_ = ', '.join([self.variable(x) for x in self.get('output-labels')])
        gen.add_a_line(f('{out_} = {inst} ({x})'))

    def get_display_name(self):
        name = self.BASEMODEL
        name = name.split(".")[-1]
        return name

    def get_group(self):
        base_model = dynamic_load(self.BASEMODEL)
        return base_model.__module__.split('.')[-1]


class TensorflowLauncher(Launcher):
    FRAMEWORK = 'Tensorflow'
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()

        # optimizer
        optmizer_list = ['Adadelta', 'AdagradDA', 'Adagrad', 'Adam', 'Ftrl', 'Momentum', 'ProximalAdagrad', 'RMSProp', 'SyncReplicas']
        for opt in optmizer_list:
            opt_class = Callable('tensorflow.train.{}Optimizer'.format(opt))
            self.nested_set(['optimizer-parameter', opt], opt_class.list_config()['argdict'])
        self.set('optimizer-selector', 'Adam')

        # loss
        loss_func_list = ['absolute_difference', 'cosine_distance', 'hinge_loss', 'huber_loss', 'log_loss', 'mean_pairwise_squared_error', 'mean_squared_error', 'sigmoid_cross_entropy', 'softmax_cross_entropy', 'sparse_softmax_cross_entropy']
        for loss in loss_func_list:
            loss_class = Callable('tensorflow.losses.' + loss)
            self.nested_set(['loss-parameter', loss], loss_class.list_config()['argdict'])
        self.set('loss-selector', 'softmax_cross_entropy')

        #trainer
        self.set('trainer-parameter', {"epochs": 2, "max-steps":1000})


        self.add_fe_parameters("trainer-parameter::epochs")
        self.add_fe_parameters('optimizer-selector', type_='select', enum_values=optmizer_list)
        self.add_fe_parameters('optimizer-parameter::*', format_="{}-parameters")
        self.add_fe_parameters('loss-selector', type_='select', enum_values=loss_func_list)
        self.add_fe_parameters('loss-parameter::*', format_="{}-parameters")

    def compile(self, get_model_func):
        #_, _id = get_model_func(None)
        #return None, _id
        return None, 0

    def train(self, model, get_model_func):
        def _model(X):
            fc=tf.contrib.layers.flatten(X)
            out = tf.layers.dense(fc, 10)
            return out, 0

        def _train_input_func():
            dataset_reader = self.parent.find_role(role='data-reader', unique=True)
            X, Y = dataset_reader.data_gen.next()
            return tf.stack(X), tf.stack(Y)

        def _eval_input_func():
            dataset_reader = self.parent.find_role(role='data-reader', unique=True)
            X, Y = dataset_reader.valid_data_gen.next()
            return tf.stack(X), tf.stack(Y)

        def _model_fn(features, labels, mode):
            model_train, _id = get_model_func(features)
            #model_train, _id = _model(features)

            #TODO only support classifaction now
            pred_classes = tf.argmax(model_train, axis=1)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

            loss_str = self.get('loss-selector')
            loss_parameters = self.get('loss-parameter')[loss_str]

            #TODO other losses
            if loss_str == 'softmax_cross_entropy':
                loss_parameters['logits'] = model_train
                loss_parameters['onehot_labels'] = labels
            elif loss_str == 'sigmoid_cross_entropy':
                loss_parameters['logits'] = model_train
                loss_parameters['muti_class_labels'] = labels
            elif loss_str in ['sparse_softmax_cross_entropy', 'hinge_loss']:
                loss_parameters['logits'] = model_train
                loss_parameters['labels'] = labels

            loss = Callable('tensorflow.losses.' + loss_str).call(argdict=loss_parameters)
            logging_hook = tf.train.LoggingTensorHook({"loss": loss}, every_n_iter=10)

            if mode == tf.estimator.ModeKeys.TRAIN:
                opt_str = self.get('optimizer-selector')
                optimizer = Callable('tensorflow.train.{}Optimizer'.format(opt_str)).call(argdict=self.get('optimizer-parameter')[opt_str])

                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op = train_op)
            if mode == tf.estimator.ModeKeys.EVAL:
                labels = tf.argmax(labels, axis=1)
                eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels, predictions=pred_classes)}
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

        tf.logging.set_verbosity(tf.logging.INFO)
        trainer_param = self.get('trainer-parameter')

        result_dir=os.path.join(self.parent.node.local(), 'results')
        shutil.rmtree(result_dir, ignore_errors = True)
        estimator = tf.estimator.Estimator(model_fn = _model_fn, model_dir=result_dir)
        valid_split = self.get('validation-split', role='data-reader')
        if 'max-steps' in trainer_param:
            max_steps = trainer_param['max-steps']
        dataset_reader = self.parent.find_role(role='data-reader', unique=True)

        steps=None
        if hasattr(dataset_reader, 'get_steps_each_epoch'):
            steps = dataset_reader.get_steps_each_epoch() * trainer_param['epochs']
        estimator.train(input_fn=_train_input_func, steps=steps, max_steps=max_steps)
        if valid_split > 0:
            eval_results = estimator.evaluate(input_fn=_train_input_func, steps=max_steps)
        else:
            eval_results = estimator.evaluate(input_fn=_eval_input_func, steps=max_steps)

        return eval_results
