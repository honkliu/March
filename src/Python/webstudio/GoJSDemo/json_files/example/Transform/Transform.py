from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dot, Conv1D, Dense, BatchNormalization, Add, Embedding, Softmax, Input
from keras.optimizers import SGD
from keras.models import Model

def get_data ():
    #add your code to get data, return (x_train, y_train), (x_test, y_test)
    pass


def get_multi_head_attention (X0, X1):
    Dot_3__ = Dot (axes=(2, 2))
    Dot_3_out_port_0 = Dot_3__ ([X0,X1])
    Softmax_3__ = Softmax ()
    Softmax_3_out_port_0 = Softmax_3__ (Dot_3_out_port_0)
    Dot_4__ = Dot (axes=(2, 1))
    Dot_4_out_port_0 = Dot_4__ ([Softmax_3_out_port_0,X1])
    Add_2__ = Add ()
    Add_2_out_port_0 = Add_2__ ([X0,Dot_4_out_port_0])
    BatchNormalization_2__ = BatchNormalization ()
    BatchNormalization_2_out_port_0 = BatchNormalization_2__ (Add_2_out_port_0)
    Y = BatchNormalization_2_out_port_0
    return Y


def get_feed_forward (X):
    Conv1D_1__ = Conv1D (kernel_size=1, filters=2048, activation="relu")
    Conv1D_1_out_port_0 = Conv1D_1__ (X)
    Conv1D_2__ = Conv1D (kernel_size=1, filters=512)
    Conv1D_2_out_port_0 = Conv1D_2__ (Conv1D_1_out_port_0)
    Add_1__ = Add ()
    Add_1_out_port_0 = Add_1__ ([Conv1D_2_out_port_0,X])
    BatchNormalization_1__ = BatchNormalization ()
    BatchNormalization_1_out_port_0 = BatchNormalization_1__ (Add_1_out_port_0)
    Y = BatchNormalization_1_out_port_0
    return Y


def get_encode (X):
    multi_head_attention_1_out_port_0 = get_multi_head_attention (X,X)
    feed_forward_1_out_port_0 = get_feed_forward (multi_head_attention_1_out_port_0)
    Y = feed_forward_1_out_port_0
    return Y


def get_decode (X0, X1):
    multi_head_attention_2_out_port_0 = get_multi_head_attention (X0,X1)
    multi_head_attention_3_out_port_0 = get_multi_head_attention (multi_head_attention_2_out_port_0,X1)
    feed_forward_2_out_port_0 = get_feed_forward (multi_head_attention_3_out_port_0)
    Y = feed_forward_2_out_port_0
    return Y


def get_model_0 (X0, X1):
    Embedding_2__ = Embedding (input_dim=32, output_dim=512)
    Embedding_2_out_port_0 = Embedding_2__ (X1)
    Embedding_1__ = Embedding (input_dim=32, output_dim=512)
    Embedding_1_out_port_0 = Embedding_1__ (X0)
    encode_1_out_port_0 = get_encode (Embedding_1_out_port_0)
    decode_1_out_port_0 = get_decode (Embedding_2_out_port_0,encode_1_out_port_0)
    encode_2_out_port_0 = get_encode (encode_1_out_port_0)
    decode_2_out_port_0 = get_decode (decode_1_out_port_0,encode_2_out_port_0)
    encode_3_out_port_0 = get_encode (encode_2_out_port_0)
    decode_3_out_port_0 = get_decode (decode_2_out_port_0,encode_3_out_port_0)
    encode_4_out_port_0 = get_encode (encode_3_out_port_0)
    decode_4_out_port_0 = get_decode (decode_3_out_port_0,encode_4_out_port_0)
    encode_5_out_port_0 = get_encode (encode_4_out_port_0)
    decode_5_out_port_0 = get_decode (decode_4_out_port_0,encode_5_out_port_0)
    encode_6_out_port_0 = get_encode (encode_5_out_port_0)
    decode_6_out_port_0 = get_decode (decode_5_out_port_0,encode_6_out_port_0)
    Dense_4__ = Dense (units=512)
    Dense_4_out_port_0 = Dense_4__ (decode_6_out_port_0)
    Softmax_2__ = Softmax ()
    Softmax_2_out_port_0 = Softmax_2__ (Dense_4_out_port_0)
    Y = Softmax_2_out_port_0
    return Y


def get_model ():
    X1 = Input(batch_shape=[None, 32])
    X0 = Input(batch_shape=[None, 32])
    Y = get_model_0 (X1=X1, X0=X0)
    return Model([X1, X0], [Y])


def model_summary (filename):
    #generate summary to file
    model = get_model()
    with open(filename, 'w') as fn:
        model.summary(print_fn=lambda x: fn.write(x+"\n"))        
    


def train ():
    model = get_model()
    (x_train, y_train), (x_test, y_test) = get_data()
    optimizer = SGD(momentum=0.9,lr=0.001,decay=0.)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.2,shuffle=True)
    score = model.evaluate(x_test, y_test, batch_size=32)
    model.save('m.h5')
    return hist.history, score
    

