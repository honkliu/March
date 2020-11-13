"""runner is a wrapper for a model to train and evaluate
"""
from __future__ import division
from math import ceil

from .config_utils import Configurable
from .dynamic_utils import dynamic_load, Bindable, kw2dict

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D, DepthwiseConv2D
from keras.layers.normalization import BatchNormalization


layer_types = {
    'conv2d': (Conv2D),
    'bn': (BatchNormalization),
    'fc': (Dense),
}


def identify_layer(layer, type_: str):
    "check whether layer is an instance of type"
    assert type_ in layer_types.keys()
    return isinstance(layer, layer_types[type_])


def keras_count_macs(mdl):
    """calculate the number of multiplications of a model (or layer)"""
    # if model, to call for each mdl
    if isinstance(mdl, Model):
        return sum(keras_count_macs(x) for x in mdl.layers)
        
    if identify_layer(mdl, 'fc'):
        w = mdl.get_weights()[0]
        return w.shape[0] * w.shape[1]
    
    # since DepthwiseConv2D is subclass of Conv2D, calculate it same as normal conv2d
    # except assert its cout equal to 1
    if identify_layer(mdl, 'conv2d'):
        kernel_h, kernel_w, cin, cout = mdl.get_weights()[0].shape
        if isinstance(mdl, DepthwiseConv2D):
            assert cout == 1, 'DepthwiseConv2D has cout not equal to 1 ({})'.format(cout)
        stride_h, stride_w = mdl.strides
        input_h, input_w = mdl.input_shape[1:3]
        return kernel_h * kernel_w * cin * ceil(input_h/stride_h) * ceil(input_w/stride_w) * cout
    
    if identify_layer(mdl, 'bn'):
        return sum(mdl.input_shape[1:])

    return 0

class KerasRunner(Configurable):
    """runner for a keras model
    
    Limited by time and resource, now only support image classifier and data generator input
    """
   
    def load_image_folder(self):
        "load image directory with transforms"
        data_dirs = self.tryget('data-directories')
        transform_args = self.tryget('image-transforms')
        generator_extra_args = self.tryget('generator-extra-args', default=dict())
        assert len(data_dirs) == len(transform_args)
        image_transforms = {k: ImageDataGenerator(**v) for k,v in transform_args.items()}
        self.data_gens = {k: image_transforms[k].flow_from_directory(v, **generator_extra_args) for k,v in data_dirs.items()}
    
    def load_model(self):
        """load a saved model form {model-path}"""
        model_path = self.tryget('model-path')
        custom_objects = {k: dynamic_load(v) for k,v in self.tryget('custom-objects', default={}).items()}
        self.model = None
        if model_path is not None:
            self.model = load_model(model_path, custom_objects=custom_objects)
            print('model loaded from {}'.format(model_path))
            if not hasattr(self.model, 'metrics'): # not compiled
                self.compile_model(self.model)
    
    def compile_model(self, model):
        """compile a model from {optimizer} and {compile-arguments}"""
        optimizer = Bindable().from_dict(self.get('optimizer')).invoke()
        compile_args = self.tryget('compile-args', default={})
        compile_args.update({'optimizer': optimizer})
        model.compile(**compile_args)
        print('model compiled')
        
    def train(self, model):
        """train a model with fit_generator"""
        train_args = self.tryget('train-args', default={})
        train_args.update(kw2dict(
            generator=self.data_gens['train'],
            validation_data=self.data_gens['val'], 
            validation_steps=self.data_gens['val'].samples
        ))
        output_model_path = self.tryget('output-model-path', default=self.get('model-path'))
        ck_logger = ModelCheckpoint(filepath=output_model_path, monitor='val_acc', save_best_only=True)
        train_args.update(kw2dict(callbacks=[ck_logger]))
        hist = model.fit_generator(**train_args)
        
    def evaluate(self, model):
        """evaluate a mode with evaluate_generator"""
        eval_args = kw2dict(
            generator=self.data_gens['val'], steps=self.data_gens['val'].samples, verbose=1
        )
        return model.evaluate_generator(**eval_args)