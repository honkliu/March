'''
{
  "SourceName": "Keras-Builtin",
  "Url": "https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html",
  "Description": "Keras helps solving a text classification problem using pre-trained word embeddings and a convolutional neural network."
}
'''
import os
import numpy as np

from keras.models import Sequential
from keras.layers import Embedding

from ...utils import download_and_unpack, ModelBrick

class KerasBrick_TextEmbedding(ModelBrick):
    FILENAME = __file__
    def __declare__(self):
        super().__declare__()
        self.batch_set(['embedding_vector', 'word_vector_sources', 'embedding_dim'])
        # add embedding vectors
        self.nested_set(['word_vector_sources', 'glove.6B'], {
            'source_url':'http://nlp.stanford.edu/data/glove.6B.zip', 
            'dest_file_name':'glove.6B.zip', 'dest_dir_name':None, 
            'vector_files': ['glove.6B.50d.txt', 'glove.6B.100d.txt', 'glove.6B.200d.txt','glove.6B.300d.txt']
            })
        all_vectors = []
        for k in self.get('word_vector_sources'):
            all_vectors.extend(self.get('word_vector_sources', k, 'vector_files'))
        self.mark_enum('embedding_vector', all_vectors)
        # 
        self.embedding_matrix = None
        self.reader = None

    def find_word_vector_source(self):
        for k,v in self.get('word_vector_sources').items():
            if k[0] + k[-1] == '<>':
                continue
            if self.get('embedding_vector') in v['vector_files']:
                return k
        raise Exception('not found word vector file %s' % self.get('embedding_vector'))

    def get_embedding_matrix(self):
        if self.embedding_matrix is not None:
            return self.embedding_matrix
        # read embedding index
        vecsrc = self.find_word_vector_source()
        self.word_dir = download_and_unpack(self.get('word_vector_sources', vecsrc), self.get('workdir'))
        filename = os.path.join(self.word_dir, self.get('embedding_vector'))
        embeddings_index = {}
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        # prepare embedding matrix
        word_index = self.reader.word_index()
        max_num_words = self.reader.get('max_num_words')
        num_words = min(max_num_words, len(word_index) + 1)
        embedding_matrix = np.zeros((num_words, self.get('embedding_dim')))
        for word, i in word_index.items():
            if i >= max_num_words:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix
        return self.embedding_matrix

    def generate_model(self, model=None):
        assert(self.reader is not None)
        embedding_matrix = self.get_embedding_matrix()
        embedding_layer = Embedding(embedding_matrix.shape[0],
                            embedding_matrix.shape[1],
                            weights=[embedding_matrix],
                            input_length=self.get('x_shape', role='data-reader')[1],
                            trainable=False)
        if model is None:
            model = Sequential()
        model.add(embedding_layer)
        self.model = model
        return model

brick_names = ['KerasBrick_TextEmbedding']
def list_bricks():
    d = dict()
    for m in brick_names:
        mdl = globals()[m]()
        d[mdl.get('name')] = mdl.list_config()
    return d