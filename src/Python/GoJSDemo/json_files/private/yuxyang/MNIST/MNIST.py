import nni
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.layers import Conv2D, Input, MaxPooling2D, Dropout, Flatten, Dense
from keras.datasets.mnist import load_data
from keras.utils import to_categorical
from keras.callbacks import Callback
from keras.models import Model

def get_model_0 (X):
    Conv2D_1__ = Conv2D (kernel_size=[3, 3], filters=32)
    Conv2D_1_out_port_0 = Conv2D_1__ (X)
    Conv2D_2__ = Conv2D (kernel_size=[3, 3], filters=32)
    Conv2D_2_out_port_0 = Conv2D_2__ (Conv2D_1_out_port_0)
    MaxPooling2D_1__ = MaxPooling2D ()
    MaxPooling2D_1_out_port_0 = MaxPooling2D_1__ (Conv2D_2_out_port_0)
    Flatten_1__ = Flatten ()
    Flatten_1_out_port_0 = Flatten_1__ (MaxPooling2D_1_out_port_0)
    Dense_1__ = Dense (units=128, activation="relu")
    Dense_1_out_port_0 = Dense_1__ (Flatten_1_out_port_0)
    Dropout_1__ = Dropout (rate=0.2)
    Dropout_1_out_port_0 = Dropout_1__ (Dense_1_out_port_0)
    Dense_2__ = Dense (units=10, activation="softmax")
    Dense_2_out_port_0 = Dense_2__ (Dropout_1_out_port_0)
    Y = Dense_2_out_port_0
    return Y


class SendMetrics (Callback):
    #Keras callback to send metrics to NNI framework
    def on_epoch_end(self, epoch, logs):
        #Run on end of each epoch
        nni.report_intermediate_result(logs["val_acc"])
    


def generate_default_params ():
    return {'loss': 'categorical_crossentropy', 'rho': 0.9, 'optimizer': 'SGD', 'beta_1': 0.9, 'lr': 0.001, 'schedule_decay': 0.004, 'beta_2': 0.999, 'optimizer_params': {'momentum': '0.99', 'decay': '0.', 'lr': '0.001'}, 'batch_size': 32, 'epochs': 5, 'momentum': 0.99, 'decay': 0.0}


def get_data ():
    # get public data
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def get_model ():
    X = Input(batch_shape=[None, 28, 28, 1])
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
    hist = model.fit(x_train, y_train, batch_size=params['batch_size'], epochs=params['epochs'], validation_split=0.2,shuffle=True,callbacks=[SendMetrics()])
    score = model.evaluate(x_test, y_test, batch_size=params['batch_size'])
    model.save('m.h5')
    nni.report_final_result(score[1])
    return hist.history, score
    


if __name__ == '__main__':
    received_params = nni.get_next_parameter()
    params = generate_default_params()
    params.update(received_params)
    hist, score = train()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

