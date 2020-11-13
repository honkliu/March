from keras.layers import Reshape, MaxPooling3D, Dropout, AveragePooling3D, BatchNormalization, LSTM, Dense, Conv3D, Input
from keras.models import Model
from keras.utils import to_categorical
from keras.datasets.cifar10 import load_data
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

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


def get_hidden_layer (X):
    Dense_1__ = Dense (units=256, activation="relu")
    Dense_1_out_port_0 = Dense_1__ (X)
    BatchNormalization_7__ = BatchNormalization ()
    BatchNormalization_7_out_port_0 = BatchNormalization_7__ (Dense_1_out_port_0)
    Dropout_4__ = Dropout (rate=0.5)
    Dropout_4_out_port_0 = Dropout_4__ (BatchNormalization_7_out_port_0)
    Y = Dropout_4_out_port_0
    return Y


def get_model_0 (X):
    AveragePooling3D_1__ = AveragePooling3D (pool_size=(1, 3, 3))
    AveragePooling3D_1_out_port_0 = AveragePooling3D_1__ (X)
    Conv3D_1__ = Conv3D (filters=64, kernel_size=(1, 3, 3), padding="same", activation="tanh")
    Conv3D_1_out_port_0 = Conv3D_1__ (AveragePooling3D_1_out_port_0)
    BatchNormalization_2__ = BatchNormalization ()
    BatchNormalization_2_out_port_0 = BatchNormalization_2__ (Conv3D_1_out_port_0)
    Conv3D_2__ = Conv3D (filters=48, kernel_size=(1, 3, 3), padding="same", activation="relu")
    Conv3D_2_out_port_0 = Conv3D_2__ (BatchNormalization_2_out_port_0)
    BatchNormalization_3__ = BatchNormalization ()
    BatchNormalization_3_out_port_0 = BatchNormalization_3__ (Conv3D_2_out_port_0)
    MaxPooling3D_1__ = MaxPooling3D (pool_size=(1, 3, 3))
    MaxPooling3D_1_out_port_0 = MaxPooling3D_1__ (BatchNormalization_3_out_port_0)
    Dropout_1__ = Dropout (rate=0.5)
    Dropout_1_out_port_0 = Dropout_1__ (MaxPooling3D_1_out_port_0)
    Conv3D_3__ = Conv3D (filters=32, kernel_size=(1, 3, 3), padding="same", activation="tanh")
    Conv3D_3_out_port_0 = Conv3D_3__ (Dropout_1_out_port_0)
    BatchNormalization_4__ = BatchNormalization ()
    BatchNormalization_4_out_port_0 = BatchNormalization_4__ (Conv3D_3_out_port_0)
    MaxPooling3D_2__ = MaxPooling3D (pool_size=(1, 3, 3))
    MaxPooling3D_2_out_port_0 = MaxPooling3D_2__ (BatchNormalization_4_out_port_0)
    Dropout_2__ = Dropout (rate=0.5)
    Dropout_2_out_port_0 = Dropout_2__ (MaxPooling3D_2_out_port_0)
    Reshape_1__ = Reshape (target_shape=(20, 512))
    Reshape_1_out_port_0 = Reshape_1__ (Dropout_2_out_port_0)
    LSTM_1__ = LSTM (units=256, activation="relu", dropout=0.3, recurrent_dropout=0.3, return_sequences=True)
    LSTM_1_out_port_0 = LSTM_1__ (Reshape_1_out_port_0)
    LSTM_2__ = LSTM (units=512, dropout=0.3, recurrent_dropout=0.3)
    LSTM_2_out_port_0 = LSTM_2__ (LSTM_1_out_port_0)
    Dropout_3__ = Dropout (rate=0.5)
    Dropout_3_out_port_0 = Dropout_3__ (LSTM_2_out_port_0)
    SubGraphSiSo_1_out_port_0 = get_hidden_layer (Dropout_3_out_port_0)
    SubGraphSiSo_2_out_port_0 = get_hidden_layer (SubGraphSiSo_1_out_port_0)
    Dense_4__ = Dense (units=1, activation="softmax")
    Dense_4_out_port_0 = Dense_4__ (SubGraphSiSo_2_out_port_0)
    Y = Dense_4_out_port_0
    return Y


def get_model ():
    X = Input(batch_shape=[None, 20, 128, 128, 3])
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

