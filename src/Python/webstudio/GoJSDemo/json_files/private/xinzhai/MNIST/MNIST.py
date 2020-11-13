from keras.preprocessing.image import ImageDataGenerator
from keras.datasets.mnist import load_data
from keras.layers import Input, Conv2D, Dense, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.models import Model

def get_model_0 (X):
    Conv2D_1__ = Conv2D (filters=32, kernel_size=[3, 3])
    Conv2D_1_out_port_0 = Conv2D_1__ (X)
    Conv2D_2__ = Conv2D (filters=32, kernel_size=[3, 3])
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
    optimizer = SGD(momentum=0.99,lr=0.001,decay=0.)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.2,shuffle=True)
    score = model.evaluate(x_test, y_test, batch_size=32)
    model.save('m.h5')
    return hist.history, score
    


if __name__ == '__main__':
    hist, score = train()
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

