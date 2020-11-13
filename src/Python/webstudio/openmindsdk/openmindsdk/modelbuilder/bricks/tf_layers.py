from .tf_utils import TensorflowModelComponent
from ...utils import dynamic_load_module

class TensorflowLayer(TensorflowModelComponent):
    FILENAME = __file__
    FRAMEWORK = 'Tensorflow'
    def __declare__(self):
        super().__declare__()
        self.set("trainable", True)

core_layers = dynamic_load_module("tensorflow.layers")

def layer_full_name(layer):
    return "tensorflow.layers.%s" % layer

for layer in core_layers:
    class_name = "TensorflowLayers" + layer
    globals()[class_name] = type(class_name, (TensorflowLayer,), {"BASEMODEL": layer_full_name(layer), "SHOW_IN_GALLERY": True})