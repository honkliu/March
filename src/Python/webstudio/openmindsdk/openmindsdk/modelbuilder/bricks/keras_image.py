'''
{
  "SourceName": "Keras-Builtin",
  "Url": "https://keras.io/applications/",
  "Description": "Keras Applications are deep learning models that are made available alongside pre-trained weights. These models can be used for prediction, feature extraction, and fine-tuning. Weights are downloaded automatically when instantiating a model. They are stored at ~/.keras/models/."
}
'''
from ..utils import ModelBrick, DataReaderBrick

from ...utils import Callable

from .utils_bricks import DownloaderBrick
from .keras_utils import KerasModelComponent

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import os


class KerasImageCategoricalFolderReader(DataReaderBrick):
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()
        self.set('role', 'data-reader')
        self.set('train-directory', 'train')
        self.set('val-directory', 'val')

        generator_name = 'keras.preprocessing.image.ImageDataGenerator'
        datagen = Callable(generator_name)
        directory_flow = Callable(generator_name + '.flow_from_directory', datagen.call().flow_from_directory)

        self.image_data_generator = datagen
        self.set('ImageDataGenerator', datagen.get('argdict'))
        augment_list = ["horizontal_flip", "vertical_flip", "rotation_range", "width_shift_range", "height_shift_range", "shear_range", "rescale"]
        self.add_fe_parameters(*["ImageDataGenerator::" + x for x in augment_list])
        self.add_fe_parameters('train-directory', 'val-directory')

    def get_data_ready(self):
        data_transforms = {
            'train': self.get('ImageDataGenerator'),
            'val': {'rescale': self.get('ImageDataGenerator', 'rescale')}
        }
        print(data_transforms)
        flow_args = {
            'target_size':self.get('x-shape')[1:3], 
            'batch_size':self.get('batch-size'), 
            'class_mode': 'categorical',
        }
        relative_path = {k: self.parent.node.local(self.get(k+'-directory')) for k in data_transforms.keys()}
        data_gens = {k: ImageDataGenerator(**v).flow_from_directory(relative_path[k], **flow_args) for k,v in data_transforms.items()}

        self.data_gen, self.valid_data_gen = data_gens['train'], data_gens['val']

    def traceback_run(self, state):
        super().traceback_run(state)
        self.get_data_ready()


class KerasImageFlowReader(DataReaderBrick):
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()
        self.set('role', 'data-reader')
        self.set('is-in-working-directory', True)

        generator_name = 'keras.preprocessing.image.ImageDataGenerator'
        datagen = Callable(generator_name)
        dategen_flow = Callable(generator_name + '.flow', datagen.call().flow)

        self.image_data_generator = datagen
        self.set('ImageDataGenerator', datagen.get('argdict'))
        self.set('ImageDataGenerator.flow', dategen_flow.get('argdict'))

        augment_list = ["horizontal_flip", "vertical_flip", "rotation_range", "width_shift_range", "height_shift_range", "shear_range", "rescale"]
        self.add_fe_parameters(*["ImageDataGenerator::" + x for x in augment_list])

    def get_X_Y(self, **argdict):
        return NotImplementedError

    def get_data_ready(self):
        train_X, train_Y, valid_X, valid_Y = self.get_X_Y()
        dategen_flow_arg = self.get('ImageDataGenerator.flow')
        dategen_flow_arg.update({'x':train_X, 'y':train_Y, 'batch_size':self.get('batch-size')})
        Callable.correct_args(dategen_flow_arg)
        self.data_gen = self.image_data_generator.call().flow(**dategen_flow_arg)

        if valid_X is not None:
            dategen_flow_arg['x'] = valid_X
            dategen_flow_arg['y'] = valid_Y
            self.valid_data_gen = self.image_data_generator.call().flow(**dategen_flow_arg)

    def explore_data(self):
        if self.data_gen is not None:
            self.x_data, self.y_data = self.data_gen.next()
        super().explore_data()


base_models = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'NASNetLarge', 'NASNetMobile', 'ResNet50', 'VGG16', 'VGG19', 'Xception']


class KerasImageBaseModel(KerasModelComponent):
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()
        self.nested_set(['basemodel', 'argdict', 'weights'], 'imagenet')
        self.nested_set(['basemodel', 'argdict', 'include_top'], False)
        self.set('trainable', False)
        self.set('require-input', True)
        self.add_fe_parameters('trainable')

    def get_group(self):
        return 'pretrained_networks'


def model_full_name(modelname):
    return 'keras.applications.%s' % (modelname)


for bmodel_name in base_models:
    dmodel_name = 'KerasImage' + bmodel_name
    globals()[dmodel_name] = type(dmodel_name, (KerasImageBaseModel,), {"BASEMODEL": model_full_name(bmodel_name), "SHOW_IN_GALLERY": True})
