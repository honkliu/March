from keras.preprocessing.image import ImageDataGenerator
from keras.layers import AveragePooling3D, Dropout, Dense, LSTM, BatchNormalization, Reshape, Input, MaxPooling3D, Conv3D
from keras.optimizers import SGD
from keras.models import Model

def get_hidden_layer (X):
    Dense_1__ = Dense (units=256, activation="relu")
    Dense_1_out_port_0 = Dense_1__ (X)
    BatchNormalization_7__ = BatchNormalization ()
    BatchNormalization_7_out_port_0 = BatchNormalization_7__ (Dense_1_out_port_0)
    Dropout_4__ = Dropout (rate=0.5)
    Dropout_4_out_port_0 = Dropout_4__ (BatchNormalization_7_out_port_0)
    Y = Dropout_4_out_port_0
    return Y


def get_data ():
    #add your code to get data, return (x_train, y_train), (x_test, y_test)
    pass


def get_model_0 (X):
    AveragePooling3D_1__ = AveragePooling3D (pool_size=(1, 3, 3))
    AveragePooling3D_1_out_port_0 = AveragePooling3D_1__ (X)
    Conv3D_1__ = Conv3D (filters=64, padding="same", kernel_size=(1, 3, 3), activation="tanh")
    Conv3D_1_out_port_0 = Conv3D_1__ (AveragePooling3D_1_out_port_0)
    BatchNormalization_2__ = BatchNormalization ()
    BatchNormalization_2_out_port_0 = BatchNormalization_2__ (Conv3D_1_out_port_0)
    Conv3D_2__ = Conv3D (filters=48, padding="same", kernel_size=(1, 3, 3), activation="relu")
    Conv3D_2_out_port_0 = Conv3D_2__ (BatchNormalization_2_out_port_0)
    BatchNormalization_3__ = BatchNormalization ()
    BatchNormalization_3_out_port_0 = BatchNormalization_3__ (Conv3D_2_out_port_0)
    MaxPooling3D_1__ = MaxPooling3D (pool_size=(1, 3, 3))
    MaxPooling3D_1_out_port_0 = MaxPooling3D_1__ (BatchNormalization_3_out_port_0)
    Dropout_1__ = Dropout (rate=0.5)
    Dropout_1_out_port_0 = Dropout_1__ (MaxPooling3D_1_out_port_0)
    Conv3D_3__ = Conv3D (filters=32, padding="same", kernel_size=(1, 3, 3), activation="tanh")
    Conv3D_3_out_port_0 = Conv3D_3__ (Dropout_1_out_port_0)
    BatchNormalization_4__ = BatchNormalization ()
    BatchNormalization_4_out_port_0 = BatchNormalization_4__ (Conv3D_3_out_port_0)
    MaxPooling3D_2__ = MaxPooling3D (pool_size=(1, 3, 3))
    MaxPooling3D_2_out_port_0 = MaxPooling3D_2__ (BatchNormalization_4_out_port_0)
    Dropout_2__ = Dropout (rate=0.5)
    Dropout_2_out_port_0 = Dropout_2__ (MaxPooling3D_2_out_port_0)
    Reshape_1__ = Reshape (target_shape=(20, 512))
    Reshape_1_out_port_0 = Reshape_1__ (Dropout_2_out_port_0)
    LSTM_1__ = LSTM (return_sequences=True, dropout=0.3, units=256, activation="relu", recurrent_dropout=0.3)
    LSTM_1_out_port_0 = LSTM_1__ (Reshape_1_out_port_0)
    LSTM_2__ = LSTM (dropout=0.3, units=512, recurrent_dropout=0.3)
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
    optimizer = SGD(momentum=0.9,lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.2,shuffle=True)
    score = model.evaluate(x_test, y_test, batch_size=32)
    model.save('m.h5')
    return hist.history, score
    

