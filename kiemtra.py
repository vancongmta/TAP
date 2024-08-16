#coding=utf-8

import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import multiprocessing
import re
from gensim.corpora.dictionary import Dictionary

K.clear_session()

np.random.seed(1337)  # For reproducibility
vocab_dim = 256
maxlen = 350
batch_size = 512
n_epoch = 60
input_length = 350
cpu_count = multiprocessing.cpu_count()

labels = ["safe", "CWE-78", "CWE-79", "CWE-89", "CWE-90", "CWE-91", "CWE-95", "CWE-98", "CWE-601", "CWE-862"]

def create_dictionaries(model=None, combined=None):
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.key_to_index.keys(), allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}
        w2vec = {word: model.wv[word] for word in w2indx.keys()}

        def parse_dataset(combined):
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except KeyError:
                        new_txt.append(0)  # Use 0 for unknown words
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec, combined
    else:
        print('No data provided...')

def input_transform(string):
    words = string.split()
    words = [words]  # Wrap in a list to create a batch of size 1
    model = Word2Vec.load('./Word2vec_model.pkl')
    _, _, combined = create_dictionaries(model, words)
    return combined

def lstm_predict():
    global labels
    print('Loading model...')
    with open('./lstm_model.json', 'r', encoding='utf-8') as f:
        json_string = f.read()
    model = model_from_json(json_string)

    print('Loading weights...')
    model.load_weights('./lstm.h5')

    with open('./traindata_x.txt', 'r', encoding='utf-8') as f:
        strings = f.readlines()
    with open('./traindata_y.txt', 'r', encoding='utf-8') as f:
        y = [list(map(int, re.findall(r'\d+', label))) for label in f.readlines()]

    right = 0
    false = 0
    prevalue = ''
    preresult = ''

    for i, string in enumerate(strings):
        data = input_transform(string)
        data[data >= model.layers[0].input_dim] = 0
        result = np.argmax(model.predict(data), axis=1)[0]
        value = model.predict(data)[0]
        prevalue += (','.join(str(i) for i in value) + '\n')
        preresult += (labels[result] + ' ' + labels[y[i].index(1)] + '\n')
        
        if result == y[i].index(1):
            right += 1
        else:
            false += 1

    print('Correct: %d Incorrect: %d' % (right, false))
    with open('./predict_result1.txt', 'w', encoding='utf-8') as f:
        f.write(preresult)
    with open('./predict_value1.txt', 'w', encoding='utf-8') as f:
        f.write(prevalue)
    print('Prediction results saved to predict_result1.txt and predict_value1.txt')

if __name__ == '__main__':
    lstm_predict()
