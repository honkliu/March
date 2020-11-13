import os
import gzip
import pickle
import keras
import numpy as np

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.utils import Sequence

from bricks.keras_image import KerasImageFlowReader
from utils import ModelBrick, DataReaderBrick, Callable, train_validation_split_folder, dynamic_load, TextSequence

class KerasMnistReader(KerasImageFlowReader):
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()
        self.set('download_filename', '')
        self.add_fe_parameters('download_filename')
         
    def get_X_Y(self):
        filename= self.get('download_filename')
        file_whole_path = self.parent.node.local(filename)

        with gzip.open(file_whole_path, 'rb') as f:
            unpick = pickle._Unpickler(f)
            unpick.encoding = 'latin1'
            train, validation, test = unpick.load()

        data_all = np.concatenate([train[0], validation[0], test[0]])
        label_all = np.concatenate([train[1], validation[1], test[1]])

        img_height, img_width, channel_count, categories_num = 28, 28, 1, 10
        data_all = data_all.reshape(data_all.shape[0], img_height, img_width, channel_count)
        data_all = data_all.astype('float32') / 255.0
        label_all = keras.utils.to_categorical(label_all, categories_num)

        val_split_ratio = self.get('validation-split')
        if val_split_ratio is not None and val_split_ratio > 0.0:
            assert(val_split_ratio >= 0.0 and val_split_ratio <= 1.0)
            train_count = int(len(data_all) * (1 - val_split_ratio))
            return data_all[:train_count, ...], label_all[:train_count, ...], data_all[train_count+1:, ...], label_all[train_count+1:, ...]

        return data_all, label_all, None, None

class KerasCifar10Reader(KerasImageFlowReader):
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()
        self.set('folder', '')
        self.add_fe_parameters('folder')

    def get_X_Y(self):
        folder = self.get('folder')
        if os.path.isdir(self.parent.node.local(folder)):
            folder = self.parent.node.local(folder)
        
        batch_files_count = 5
        train_datas = []
        labels = []
        for i in range(batch_files_count):
            file_whole_path = os.path.join(folder, 'data_batch_' + str(i+1))
            data_batch, label_batch = self.load_batch_data(file_whole_path)
            train_datas.append(data_batch)
            labels.append(label_batch)

        data_all = np.concatenate(train_datas)
        label_all = np.concatenate(labels)

        val_split_ratio = self.get('validation-split')
        if val_split_ratio is not None and val_split_ratio > 0.0:
            assert(val_split_ratio >= 0.0 and val_split_ratio <= 1.0)
            train_count = int(len(data_all) * (1 - val_split_ratio))
            return data_all[:train_count, ...], label_all[:train_count, ...], data_all[train_count+1:, ...], label_all[train_count+1:, ...]

        return data_all, label_all, None, None

    def load_batch_data(self, file_path):
        with open(file_path, mode='rb') as file:
            data = pickle.load(file, encoding='bytes')
            images = data[b'data']
            labels = np.array(data[b'labels'])

            img_float = np.array(images, dtype=float) / 255.0
            channel_count, img_height, img_width, categories_num = 3, 32, 32, 10
            images = img_float.reshape(-1, channel_count, img_height, img_width)
            images = images.transpose(0, 2, 3, 1)
        
            labels = keras.utils.to_categorical(labels, categories_num)

        return images, labels


class TextSequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        np.random.seed(113)
        indices = np.arange(len(self.x))
        np.random.shuffle(indices)
        self.x = self.x[indices]
        self.y = self.y[indices]



class KerasTextDatasetReader(DataReaderBrick):
    '''
    load imdb / reuters datasets from keras.dataset package
    '''
    FILENAME = __file__

    def __declare__(self):
        super().__declare__()
        available_datasets = ['imdb', 'reuters']
        generator_name = 'keras.datasets.imdb.load_data'
        self.datagen = Callable(generator_name)
        self.set('datagen', self.datagen.get('argdict'))
        
        # load_data(path='imdb.npz', num_words=None, skip_top=0, maxlen=None, seed=113, start_char=1, oov_char=2, index_from=3, **kwargs)
        self.set('sen_len', None)
        self.add_fe_parameters(*["datagen::" + x for x in ['num_words', 'skip_top', 'maxlen']])
        self.add_fe_parameters('sen_len')
        self.add_fe_parameters("dataset", type_="select", enum_values = available_datasets)


    def get_data_ready(self):
        (x_train, y_train), (x_test, y_test) = self.datagen.call()
        x_train = sequence.pad_sequences(x_train, maxlen=self.get('sen_len'))
        x_test = sequence.pad_sequences(x_test, maxlen=self.get('sen_len'))
        self.data_gen = TextSequence(x_train, y_train, self.get('batch-size'))
        self.valid_data_gen = TextSequence(x_test, y_test, self.get('batch-size'))
    
    def explore_data(self):
        if self.data_gen is not None:
            self.x_data, self.y_data = self.data_gen[0]
        super().explore_data()

        

            
    
    
