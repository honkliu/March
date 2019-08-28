#@test {"skip": true}

# NOTE: If you are running a Jupyter notebook, and installing a locally built
# pip package, you may need to edit the following to point to the '.whl' file
# on your local filesystem.

# NOTE: The high-performance executor components used in this tutorial are not
# yet included in the released pip package; you may need to compile from source.
from __future__ import absolute_import, division, print_function

import nest_asyncio
nest_asyncio.apply()


import collections
import warnings
from six.moves import range
import numpy as np
import six
import tensorflow as tf

warnings.simplefilter('ignore')

tf.compat.v1.enable_v2_behavior()

import tensorflow_federated as tff

np.random.seed(0)

NUM_CLIENTS = 10

# NOTE: If the statement below fails, it means that you are
# using an older version of TFF without the high-performance
# executor stack. Call `tff.framework.set_default_executor()`
# instead to use the default reference runtime.
#if six.PY3:
#      tff.framework.set_default_executor(
#                    tff.framework.create_local_executor(NUM_CLIENTS))
#      tff.framework.set_default_executor()


if six.PY3:
  tff.framework.set_default_executor()
#      tff.framework.create_local_executor(NUM_CLIENTS))

tff.federated_computation(lambda: 'Hello, World!')()

#@test {"output": "ignore"}
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

emnist_train.output_types, emnist_train.output_shapes

example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])

example_element = iter(example_dataset).next()

example_element['label'].numpy()

NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500

def preprocess(dataset):

  def element_fn(element):
    return collections.OrderedDict([
        ('x', tf.reshape(element['pixels'], [-1])),
        ('y', tf.reshape(element['label'], [1])),
    ])

  return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
      SHUFFLE_BUFFER).batch(BATCH_SIZE)
#@test {"output": "ignore"}
preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = tf.nest.map_structure(
    lambda x: x.numpy(), iter(preprocessed_example_dataset).next())

sample_batch
def make_federated_data(client_data, client_ids):
  return [preprocess(client_data.create_tf_dataset_for_client(x))
          for x in client_ids]

#@test {"output": "ignore"}
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)

len(federated_train_data), federated_train_data[0]

def create_compiled_keras_model():
  model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])
  
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.02),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  return model

def model_fn():
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)

#@test {"output": "ignore"}
iterative_process = tff.learning.build_federated_averaging_process(model_fn)

#@test {"output": "ignore"}
str(iterative_process.initialize.type_signature)

state = iterative_process.initialize()

#@test {"timeout": 600, "output": "ignore"}
state, metrics = iterative_process.next(state, federated_train_data)
print('round  1, metrics={}'.format(metrics))

for round_num in range(2,5):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round  {:2d}, metrics={}'.format(round_num, metrics))

def create_mnist_variables():
    return MnistVariables(
        weights = tf.Variable(
            lambda:tf.zeros(dtype=tf.float32, shape=(784,10)),
            name = 'Weights',
            trainable = True),
        bias = tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(10)),
            name = 'bias',
            trainable = True),
        num_examples = tf.Variable(
            0.0, 
            name = 'num_examples',
            trainable = False),
        loss_sum = tf.Variable(
            0.0,
            name = 'loss_sum',
            trainale= False),
        accuracy_sum = tf.Variable(
            0.0,
            name = 'accuracy_sum',
            trainale= False))

def mnist_forward_pass(variables, batch):
    y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)

    predictions = tf.cast(tf.argmax(y, 1), tf.int32)

    flat_labels = tf.reshape(batch['y'], [-1])
    loss = -tf.reduce_mean(tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.log(y), reduction_indices=[1]))
    accuracy =tf.reduce_mean(tf.cast(tf.equal(predictions, flat_lables), tf.float32))

    num_examples = tf.cast(tf.size(batch['y']), tf.float32)
    variables.num_examples.assign_add(num_examples)
    variables.loss_sum.assign_add(loss * num_examples)
    variables.accuracy_sum.assign_add(accuracy * num_examples)
    
    return loss, predictions

def get_local_mnist_metrics(variables):
  return collections.OrderedDict([
      ('num_examples', variables.num_examples),
      ('loss', variables.loss_sum / variables.num_examples),
      ('accuracy', variables.accuracy_sum / variables.num_examples)
    ])

@tff.federated_computation
def aggregate_mnist_metrics_across_clients(metrics):
  return {
      'num_examples': tff.federated_sum(metrics.num_examples),
      'loss': tff.federated_mean(metrics.loss, metrics.num_examples),
      'accuracy': tff.federated_mean(metrics.accuracy, metrics.num_examples)
  }

class MnistModel(tff.learning.Model):

  def __init__(self):
    self._variables = create_mnist_variables()

  @property
  def trainable_variables(self):
    return [self._variables.weights, self._variables.bias]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def local_variables(self):
    return [
        self._variables.num_examples, self._variables.loss_sum,
        self._variables.accuracy_sum
    ]

  @property
  def input_spec(self):
    return collections.OrderedDict([('x', tf.TensorSpec([None, 784],
                                                        tf.float32)),
                                    ('y', tf.TensorSpec([None, 1], tf.int32))])

  @tf.function
  def forward_pass(self, batch, training=True):
    del training
    loss, predictions = mnist_forward_pass(self._variables, batch)
    return tff.learning.BatchOutput(loss=loss, predictions=predictions)

  @tf.function
  def report_local_outputs(self):
    return get_local_mnist_metrics(self._variables)

  @property
  def federated_output_computation(self):
    return aggregate_mnist_metrics_across_clients

class MnistTrainableModel(MnistModel, tff.learning.TrainableModel):

  @tf.function
  def train_on_batch(self, batch):
    output = self.forward_pass(batch)
    optimizer = tf.train.GradientDescentOptimizer(0.02)
    optimizer.minimize(output.loss, var_list=self.trainable_variables)
    return output

iterative_process = tff.learning.build_federated_averaging_process(
    MnistTrainableModel)

state = iterative_process.initialize()

 #@test {"timeout": 600, "output": "ignore"}
state, metrics = iterative_process.next(state, federated_train_data)
print('round  x1, metrics={}'.format(metrics))   