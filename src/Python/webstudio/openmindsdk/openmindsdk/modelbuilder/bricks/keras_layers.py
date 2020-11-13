from .keras_utils import KerasModelComponent
from ...utils import dynamic_load_module, Configurable

core_layers = dynamic_load_module("keras.layers")

def layer_full_name(layer):
    return "keras.layers.%s" % layer

for layer in core_layers:
    class_name = "KerasLayers" + layer
    globals()[class_name] = type(
        class_name,
        (KerasModelComponent,),
        {"BASEMODEL": layer_full_name(layer), "SHOW_IN_GALLERY": True, "FILENAME": __file__}
    )