from keras.layers import Flatten, Dropout, Dense, MaxPooling2D, Conv2D, Input
from keras.models import Model
from keras.utils import to_categorical
from keras.datasets.cifar10 import load_data
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

def get_model_0 (X):
    Conv2D_1__ = Conv2D (filters=32, kernel_size=(3, 3), padding="same", activation="relu")
    Conv2D_1_out_port_0 = Conv2D_1__ (X)
    Conv2D_2__ = Conv2D (filters=32, kernel_size=(3, 3), activation="relu")
    Conv2D_2_out_port_0 = Conv2D_2__ (Conv2D_1_out_port_0)
    MaxPooling2D_1__ = MaxPooling2D ()
    MaxPooling2D_1_out_port_0 = MaxPooling2D_1__ (Conv2D_2_out_port_0)
    Dropout_1__ = Dropout (rate=0.25)
    Dropout_1_out_port_0 = Dropout_1__ (MaxPooling2D_1_out_port_0)
    Conv2D_3__ = Conv2D (filters=64, kernel_size=(3, 3), padding="same", activation="relu")
    Conv2D_3_out_port_0 = Conv2D_3__ (Dropout_1_out_port_0)
    Conv2D_4__ = Conv2D (filters=64, kernel_size=(3, 3), activation="relu")
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
    optimizer = SGD(lr=0.001,momentum=0.9,decay=0.)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.2,shuffle=True)
    score = model.evaluate(x_test, y_test, batch_size=32)
    model.save('m.h5')
    return hist.history, score
    


if __name__ == '__main__':
    hist, score = train()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

