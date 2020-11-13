"""code templates accordint to task and framework"""
from ...utils import SourceCoder, FuncCoder

import codecs
from nbformat.v4 import new_notebook, new_code_cell
import nbformat


def get_keras_image_classifier(src_fn: SourceCoder, parameters: dict):
    "src_fn is instance of SourceCoder"
    func_name = 'train'
    cb = src_fn.add_codeblock(func_name, FuncCoder)
    cb.add_module('keras.preprocessing.image.ImageDataGenerator')
    cb.add_module('keras.optimizers.' + parameters['model_parameters']['optimizer'])
    cb.add_dependency('get_model')
    cb.add_dependency('get_data')
    optimizer = parameters['model_parameters']['optimizer'] + '('
    for key in parameters['model_parameters']['optimizer_params'].keys():
        optimizer = optimizer + key + '=' + parameters['model_parameters']['optimizer_params'][key] + ','
    if optimizer[-1] == ',':
        optimizer = optimizer[:-1]
    optimizer = optimizer + ')'
    func_body = '''model = get_model()
(x_train, y_train), (x_test, y_test) = get_data()
optimizer = %s
model.compile(optimizer=optimizer, loss='%s', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=%s, epochs=%s, validation_split=%s,shuffle=%s)
score = model.evaluate(x_test, y_test, batch_size=%s)
model.save('m.h5')
return hist.history, score
''' % (optimizer, parameters['model_parameters']['loss'], parameters['model_parameters']['batch_size'],
       parameters['model_parameters']['epochs'], parameters['model_parameters']['validation_split'],
       parameters['model_parameters']['shuffle'], parameters['model_parameters']['batch_size'])
    for line in func_body.split('\n'):
        cb.add_a_line(line)
    how_to_call = '''hist, score = train()
print('Test loss:', score[0])
print('Test accuracy:', score[1])'''
    return how_to_call


def get_keras_image_classifier_nni(src_fn: SourceCoder, parameters: dict):
    "src_fn is instance of SourceCoder"
    func_name = 'train'
    cb = src_fn.add_codeblock(func_name, FuncCoder)
    cb.add_module('nni')
    cb.add_module('keras.callbacks.Callback')
    cb.add_module('keras.optimizers.SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam')
    cb.add_dependency('get_model')
    cb.add_dependency('get_data')
    func_body = '''model = get_model()
(x_train, y_train), (x_test, y_test) = get_data()
optimizer = None
if params["optimizer"] == 'SGD':
    optimizer = SGD(momentum=params["momentum"], lr=params["lr"], decay=params['decay'])
elif params["optimizer"] == 'Adam':
    optimizer = Adam(lr=params["lr"], beta_1=params["beta_1"], beta_2=params["beta_2"],decay=params["decay"])
elif params["optimizer"] == 'RMSprop':
    optimizer = RMSprop(lr=params["lr"], rho=params["rho"], decay=params["decay"])
elif params["optimizer"] == 'Adagrad':
    optimizer = Adagrad(lr=params["lr"], decay=params["decay"])
elif params["optimizer"] == 'Adadelta':
    optimizer = Adadelta(lr=params["lr"], rho=params["rho"], decay=params["decay"])
elif params["optimizer"] == 'Adamax':
    optimizer = Adamax(lr=params["lr"], beta_1=params["beta_1"], beta_2=params["beta_2"],decay=params["decay"])
elif params["optimizer"] == 'Nadam':
    optimizer = Nadam(lr=params["lr"], beta_1=params["beta_1"], beta_2=params["beta_2"],schedule_decay=params["schedule_decay"])
model.compile(optimizer=optimizer, loss=params['loss'], metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'], validation_split=%s,shuffle=%s,callbacks=[SendMetrics()])
score = model.evaluate(x_test, y_test, batch_size=params['batch_size'])
model.save('m.h5')
nni.report_final_result(score[1])
return hist.history, score
''' % (parameters['model_parameters']['validation_split'],
       parameters['model_parameters']['shuffle'])
    for line in func_body.split('\n'):
        cb.add_a_line(line)
    how_to_call = '''received_params = nni.get_next_parameter()
params = generate_default_params()
params.update(received_params)
hist, score = train()
print('Test loss:', score[0])
print('Test accuracy:', score[1])'''
    return how_to_call


print_result_code = '''%matplotlib inline
import matplotlib.pyplot as plt
import pandas

history = result['attachments']

plt.subplot(121)
plt.plot(history['acc'], label='acc')
plt.plot(history['val_acc'], label='val_acc')
plt.legend()

plt.subplot(122)
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.legend()
'''

keras_visualization_code= '''import tensorflow as tf
from keras import backend as K

with tf.Session() as sess:
    K.set_session(sess)
    model = get_model()
show_graph(tf.get_default_graph().as_graph_def())
'''

tf_visualization_code = '''import tensorflow as tf

with tf.Session() as sess:
    model = get_model()
show_graph(tf.get_default_graph().as_graph_def())
'''

show_graph_code = '''import numpy as np
from IPython.display import HTML
# define the function to show the graph
def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))
'''

strip_consts_code = '''# define the function of strip_consts
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = b"<stripped %d bytes>"%size
    return strip_def
'''


def gen_model_runner(model_parameters: {}, src_fn: SourceCoder, builder):
    out_ = [x for x in builder.find_role('model-component-output') if not x.get('is-subgraph-output')]
    project_type = builder.stages[0].get('task-type')
    framework = out_[0].FRAMEWORK
    if project_type.lower() == 'image-classification' and framework.lower() == 'keras':
        assert len(out_) == 1
        #assert len(out_[0].model_inputs) == 1
        in_ = [builder.find(v) for v in out_[0].model_inputs.values()]
        if model_parameters['enable_nni'] == 'true':
            return get_keras_image_classifier_nni(src_fn, {
                'batch_input_shape': in_[0].get_x_shape(),
                'model_parameters': model_parameters
            })
        else:
            return get_keras_image_classifier(src_fn, {
                'batch_input_shape': in_[0].get_x_shape(),
                'model_parameters': model_parameters
            })
    if project_type.lower() == 'image-classification' and framework.lower() == 'tensorflow':
        return '# not implemented yet'
    return '# not implemented yet'


notebook_metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'Kernel Spec'
    }
}

def dump_notebook(notebook_path: str, model_path: str, framework: str='Keras', parameters: dict={}):
    cells = []

    with open(model_path) as model_file:
        lines = model_file.readlines()
    try:
        main_index = lines.index("if __name__ == '__main__':\n")
    except:
        main_index = len(lines)

    model_content = ''.join(lines[0: main_index])

    cells.append(new_code_cell(source = model_content))

    if main_index < len(lines):
        cells.append(new_code_cell(source = ''.join([line.strip(' ') for line in lines[main_index + 1:]])))

    framework = framework.lower()

    #cells.append(new_code_cell(source = parameters['run-task-code']))

    #cells.append(new_code_cell(source = print_result_code))

    cells.append(new_code_cell(source = strip_consts_code))

    cells.append(new_code_cell(source=show_graph_code))

    if framework in ['keras', 'tensorflow']:
        cells.append(new_code_cell(source = keras_visualization_code if framework == 'keras' else tf_visualization_code))

    notebook = new_notebook(cells = cells, metadata = notebook_metadata)

    with codecs.open(notebook_path, encoding = 'utf-8', mode = 'w') as notebook_file:
        nbformat.write(notebook, notebook_file, version = 4)
