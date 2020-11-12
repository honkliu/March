Web Studio for deep learning
====
- [Web Studio for deep learning](#Web Studio-for-deep-learning)
- [Introduction](#introduction)
- [Quick Start](#quick-start)
  - [User Interface](#user-interface)
    - [Model tab](#model-tab)
- [Model editing](#model-editing)
  - [Overview](#overview)
  - [Example](#example)


# Introduction
**Web Studio** (_temporary_) is a web based, easy-to-use platform to run Machine Learning (esp. Deep Learning) tasks. With it, users can
+ Build a *deep neural network* model by dragging and dropping component blocks of Keras
+ Build a nlp model by dragging and dropping component blocks of NeuronBlocks
+ View and copy model code
+ Running on jupyter

# Quick Start
Web Studio is a `Flask` based web platform, it can run on the server and user just connect via `ip:port` or it can run locally.</br>

Here is the system structure of the studio. The frontend (repo: [WebStudio4DL]()) is based on GoJs. The blocks (operators) shown in the frontend is dumped from the backend (repo: [openmindsdk]()).


## User Interface
Here is a screenshot of Web Studio.

There are *tabs* like ***Model*** and ***Result*** on the navigation bar on the left. User can edit the models and context pipeline in the *Model* tab, and view the codes generated and performance metrics in the *Result* tab.

![alt text](/assets/doc/media/openmindplus-model.png)

### Model tab
The left top is the ***Library*** tool, where list all the available blocks that can be used to draw the model and its context.

The center of _Model_ tab is the drawing canvas. User cna drag blocks from _Library_, place and connect blocks here.

On the right of canvas is the ***Property*** tool, when selecting a block, the parameters of it can be discovered and edited here.


The former is the basic function of this studio, which is similar for both local running and running as a service. The latter function is also critical for providing a training service, because in this case, it is quite important to prepare the whole task environment and get things done.

# Model editing
## Overview
The basic function of this studio is
1. **Build** a neural network model by dragging and dropping basic *component blocks*, which includes
    + [X] Basic layers (e.g. *convolutional* or *fully connected* layers) supported by the framework
    + [X] Pre-trained networks and layers, e.g. `ResNet`, `MobileNet` trained on *ImageNet* amd word embedding layer based on public vocabulary (like `Globe`)
    + [ ] `NeuronBlocks` provided by peer team
    + [X] Any custom model that are stored and can be loaded into the framework
2. **View** the network structure and details from the drawing or code snippets
3. **Edit** a existing model (_new feature, under development_)

Considering the popularity of different frameworks, now we support [`Keras`](www.keras.io) as the first citizen, and try to support [`Tensorflow`](tensorflow.google.com) as far as possible.

After drawing the neural network graph, the studio will generate a `.py` file and a `Jupyter` notebook, both of which contain a function (`get_model`) returning the model. And the summary of the model will also be generated.

## Example
Here is an example of a CNN model on the CIFAR10 small images dataset.
```
ModelInput(224,224,3) -> Conv2D(224,224,32) -> Conv2D(222,222,32) -> MaxPooling2D(111,111,32) -> Dropout(111,111,32) -> 
-> Conv2D(111,111,64) -> Conv2D(109,109,64) -> MaxPooling2D(54,54,64) ->  Dropout(54,54,64) -> Flatten(186624)
-> Dense(512) -> Dropout(512) -> Dense(10) -> ModelOutput
```
_Note: use diagram other than screenshot for better illustration_

After building the models (and their context), just click _Preview_ to let the studio generate source codes, notebooks and model summaries. There will be pop-out message to notify when the generation is done. 
![alt text](/assets/doc/media/openmindplus-preview-notebook.png "model-tab")

For better explaining the code generated, below is an example fo generated `Keras` code.

```python
from keras.optimizers import SGD
from keras.layers import MaxPooling2D, Dense, Dropout, Flatten, Input, Conv2D
from keras.models import Model
from keras.utils import to_categorical
from keras.datasets.cifar10 import load_data

def get_model_0 (X):
    Conv2D_1__ = Conv2D (kernel_size=(3, 3), padding="same", filters=32, activation="relu")
    Conv2D_1_out_port_0 = Conv2D_1__ (X)
    Conv2D_2__ = Conv2D (kernel_size=(3, 3), filters=32, activation="relu")
    Conv2D_2_out_port_0 = Conv2D_2__ (Conv2D_1_out_port_0)
    MaxPooling2D_1__ = MaxPooling2D ()
    MaxPooling2D_1_out_port_0 = MaxPooling2D_1__ (Conv2D_2_out_port_0)
    Dropout_1__ = Dropout (rate=0.25)
    Dropout_1_out_port_0 = Dropout_1__ (MaxPooling2D_1_out_port_0)
    Conv2D_3__ = Conv2D (kernel_size=(3, 3), padding="same", filters=64, activation="relu")
    Conv2D_3_out_port_0 = Conv2D_3__ (Dropout_1_out_port_0)
    Conv2D_4__ = Conv2D (kernel_size=(3, 3), filters=64, activation="relu")
    Conv2D_4_out_port_0 = Conv2D_4__ (Conv2D_3_out_port_0)
    MaxPooling2D_2__ = MaxPooling2D ()
    MaxPooling2D_2_out_port_0 = MaxPooling2D_2__ (Conv2D_4_out_port_0)
    Dropout_2__ = Dropout (rate=0.25)
    Dropout_2_out_port_0 = Dropout_2__ (MaxPooling2D_2_out_port_0)
    Flatten_1__ = Flatten ()
    Flatten_1_out_port_0 = Flatten_1__ (Dropout_2_out_port_0)
    Dense_1__ = Dense (units=512, activation="relu")
    Dense_1_out_port_0 = Dense_1__ (Flatten_1_out_port_0)
    Dropout_3__ = Dropout (rate=0.5)
    Dropout_3_out_port_0 = Dropout_3__ (Dense_1_out_port_0)
    Dense_2__ = Dense (units=10, activation="softmax")
    Dense_2_out_port_0 = Dense_2__ (Dropout_3_out_port_0)
    Y = Dense_2_out_port_0
    return Y

def get_data ():
    # get public data
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape(x_train.shape[0], 32, 32, 3)
    x_test = x_test.reshape(x_test.shape[0], 32, 32, 3)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

def get_model ():
    X = Input(batch_shape=[None, 32, 32, 3])
    Y = get_model_0 (X=X)
    return Model([X], [Y])

def model_summary (filename):
    #generate summary to file
    model = get_model()
    with open(filename, 'w') as fn:
        model.summary(print_fn=lambda x: fn.write(x+"\n"))        
    
def train ():
    model = get_model()
    (x_train, y_train), (x_test, y_test) = get_data()
    optimizer = SGD(momentum=0.9,decay=0.,lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.2,shuffle=True)
    score = model.evaluate(x_test, y_test, batch_size=32)
    model.save('m.h5')
    return hist.history, score
    
if __name__ == '__main__':
    hist, score = train()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

