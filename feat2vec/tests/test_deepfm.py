# file: test_deepfm.py
# provide a test example for Fmachines
import numpy as np
import pandas as pd
import math
import os
import logging
import re
import tensorflow as tf

from feat2vec import DeepFM

from keras.callbacks import EarlyStopping
from keras.preprocessing import text, sequence
from keras.layers import Input, GlobalMaxPool1D, Dense, Embedding
from keras.layers.convolutional import Convolution1D
from keras.constraints import non_neg
embed_dim = 10

from unittest import TestCase


class TestDeepFM(TestCase):
    def test_all_pairwise(self):

        feature_names = ["skill_id", "skill_match", "skills", "education", "experience", "summary", "titles",
                         "professional", "norm_major", "norm_dpt"]

        fm = DeepFM(model_features=[["skill_id", ],
                                    ["skill_match"],
                                    [1, 2],
                                    [1, 2],
                                    [1, 2],
                                    [1, 2],
                                    [1, 2],
                                    [1, 2],
                                    [1, 2],
                                    [1, 2]],
                    feature_dimensions=[100,
                                        1,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100],
                    realval=[False, True, False, False, False, False, False, False, False, False],
                    mask_zero=True,
                    feature_names=feature_names,
                    obj="ns")

        model = fm.build_model(10,
                               dropout_layer=0.5,
                               deep_out=True,
                               deep_out_bias=False,
                               deep_kernel_constraint=non_neg())

        model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())
        model.fit( x=[np.array([0]), # skill_id
                      np.array([0]), #skill_match
                      np.array([[51, 2]]), # skills
                      np.array([[0, 0]]), # education
                      np.array([[25, 1]]), # experience
                      np.array([[0, 0]]), # summary
                      np.array([[17, 1]]), # titles
                      np.array([[1, 1]]), #professional
                      np.array([[1, 1]]), # norm_major
                      np.array([[0, 0]])], #norm_department,
                  y=np.array([0]))
        try:
            from keras.utils import plot_model
            plot_model(model, to_file="all_pairwise.png")
        except:
            pass


    def test_some_pairwise(self):

        feature_names = ["skill_id", "skill_match", "skills", "education", "experience", "summary", "titles",
                         "professional", "norm_major", "norm_dpt"]

        fm = DeepFM(model_features=[["skill_id", ],
                                    ["skill_match"],
                                    [10],
                                    [10],
                                    [10],
                                    [10],
                                    [10],
                                    [10],
                                    [10],
                                    [10]],
                    feature_dimensions=[100,
                                        1,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100,
                                        100],
                    realval=[False, True, False, False, False, False, False, False, False, False],
                    mask_zero=True,
                    feature_names=feature_names,
                    obj="nce")

        model = fm.build_model(10,
                               dropout_layer=0.5,
                               deep_out=True,
                               deep_out_bias=False,
                               deep_weight_groups=[0] + ([1] * (len(feature_names) - 1)),
                               deep_kernel_constraint=non_neg())

        try:
            from keras.utils import plot_model
            plot_model(model, to_file="some_pairwise.png")
        except:
            pass




    def a_simple_test(self):
        fm1 = DeepFM(model_features=[["feat1"],
                                     ["subfeat2a", "subfeat2b"],
                                     ["subfeat3a", "subfeat3b"]],
                     feature_dimensions=[10, 20, 30],
                     mask_zero=True,
                     feature_names=["feat1", "feat2", "feat3"],
                     # feature_names=feature_names,
                     obj="ns")

        k = fm1.build_model(10, deep_out=False)


        self.assertTrue(k.inputs[0].name.startswith("feat1"))




    def test_deepfm(self):
        np.random.seed(1)
        # generate some data
        samplesize = 100000
        testdata = pd.DataFrame({'cat1': np.random.randint(0, 9, size=samplesize),
                                 'cat2': np.random.randint(0, 2, size=samplesize),
                                 'real1': np.random.uniform(0, 1, size=samplesize),
                                 'offset_': np.ones(samplesize)
                                 })

        testdata['latenty'] = (
        testdata.cat1 - 2 * math.pi * testdata.cat2 + testdata.real1 - (testdata.cat1 == 7) * np.exp(
            1) + np.random.normal(size=samplesize))
        # convert to binary indicator
        testdata['y'] = (testdata['latenty'] > 0).astype('int')

        # now apply Deep-Out FM
        features = [['cat1'], ['cat2'], ['real1']]
        feature_dim = [len(testdata['cat1'].unique()), len(testdata['cat2'].unique()), 1]
        realvals = [False, False, True]



        fm_obj = DeepFM(model_features=features,
                        feature_dimensions=feature_dim,
                        realval=realvals)


        print feature_dim
        print features
        fm = fm_obj.build_model(embed_dim,
                                l2_bias=0.0, l2_factors=0.0, l2_deep=0.0, deep_out=True,
                                deep_out_bias=True, deep_out_activation='linear')
        print fm.summary()

        train = testdata.iloc[:90000, :]
        test = testdata.iloc[90000:, :]
        earlyend = EarlyStopping()
        fm.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())
        inputs = [train['cat1'], train['cat2'], train['real1']]
        fm.fit(x=inputs, y=train['y'], batch_size=1000, epochs=2,
               verbose=1, callbacks=[earlyend], validation_split=.1, shuffle=True)


        # Deep FM with features

    def test_deepin_fm(self):
        try:
            import nltk
            nltk.download('movie_reviews')
            from nltk.corpus import movie_reviews

        except ImportError:
            self.skipTest("NLTK is not not installed.  Reinstall with option 'tests'")

            # download some text data, process it, and create some feature extraction layer to plug in
        print "processing text..."
        samplesize = 2000
        reviews = []
        labels = []
        for rf in movie_reviews.fileids():
            review = movie_reviews.open(rf).read()
            reviews.append(review)
            labels.append(rf.find('pos/') != -1)

        textdata = pd.DataFrame({'text': reviews, 'pos': labels, 'offset_': np.ones(len(reviews))})
        # pre-process text (do same thing Ralph does leave only consecutive alphabetical characters
        textdata['cleantext'] = textdata['text'].map(lambda x: (" ".join(re.findall('[A-Za-z]+', x))).encode('utf8'))
        tokens = [i.lower().split(" ") for i in textdata['cleantext']]
        textdata['len'] = [len(t) for t in tokens]
        textdata.len.describe()
        textdata['cat1'] = np.random.randint(0, 9, size=samplesize)
        textdata['cat2'] = np.random.randint(0, 2, size=samplesize)
        textdata['real1'] = np.random.uniform(0, 1, size=samplesize)
        textdata['latenty'] = (
        textdata.cat1 - 2 * math.pi * textdata.cat2 + textdata.real1 - math.exp(1) * textdata.pos.astype('float') +
        textdata.real1 * textdata.pos.astype('float') + np.random.normal(size=samplesize))
        # convert to binary indicator
        textdata['y'] = (textdata['latenty'] > 0).astype('int')

        # sequence length cutoff is going to be 75th percentile
        cutoff = int(textdata.len.describe()['75%'])
        tokens = [r[0:min(len(r), cutoff)] for r in tokens]

        # build vocab
        vocab = set()
        counter = 0
        for r in tokens:
            for w in r:
                if w not in vocab:
                    vocab.add(w)

        vocabsize = len(vocab)
        vocab_indices = {}
        index = 1
        for v in vocab:
            vocab_indices[v] = index
            index += 1

        tokens_indexed = []
        for r in tokens:
            tokens_indexed.append([vocab_indices[w] for w in r])

        sequence_mat = sequence.pad_sequences(tokens_indexed, maxlen=cutoff, value=0, padding='post', truncating='post')

        # build the feature extraction layer
        # do a CNN mimicing ralph's architecture (but of significantly lower dimensionality)
        word_seq = Input(batch_shape=(None, sequence_mat.shape[1]), name='wordind_seq')
        word_embeddings = Embedding(input_dim=vocabsize + 1, output_dim=1, input_length=cutoff, mask_zero=False)(
            word_seq)
        word_conv = Convolution1D(filters=10, kernel_size=3, activation='relu', use_bias=True)(word_embeddings)
        pooler = GlobalMaxPool1D()(word_conv)
        word_dense_layer = Dense(units=10, activation='relu')(pooler)
        word_final_layer = Dense(units=embed_dim, name='textfeats')(word_dense_layer)

        # collect relevant valuesfor deepFM model
        features = [['cat1'], ['cat2'], ['real1'], ['offset_'], ['textseq']]
        feature_dim = [len(textdata['cat1'].unique()), len(textdata['cat2'].unique()), 1, 1, embed_dim]
        deep_inputs = [word_seq]
        deep_feature = [word_final_layer]
        deepin = [False, False, False, False, True]
        bias_only = [False, False, False, True, False]
        realvalued = [False, False, True, False, None]  # doesn't matter what we assign to the deep feature, so just say None
        inputs = [textdata['cat1'], textdata['cat2'], textdata['real1'], pd.Categorical(textdata['offset_']).codes,
                  sequence_mat]

        # build deep-in FM
        difm_obj = DeepFM(features,
                          feature_dim,
                          realval=realvalued,
                          deepin_feature=deepin,
                          deepin_inputs=deep_inputs, deepin_layers=deep_feature)

        tf.set_random_seed(1)
        np.random.seed(1)
        difm = difm_obj.build_model(embed_dim,
                                    deep_out=False,
                                    bias_only=bias_only,
                                    dropout_input=0,
                                    dropout_layer=0)
        print difm.summary()
        earlyend = EarlyStopping(monitor='val_loss')
        difm.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())

        difm.fit(x=inputs, y=textdata['y'], batch_size=100, epochs=2,
                 verbose=1, callbacks=[earlyend], validation_split=.1, shuffle=True)

        # now add a deep-out layer for the interactions
        tf.set_random_seed(1)
        np.random.seed(1)
        diofm = difm_obj.build_model(embed_dim, deep_out=True)
        # print diofm.summary()
        earlyend = EarlyStopping(monitor='val_loss')
        diofm.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())

        diofm.fit(x=inputs, y=textdata['y'], batch_size=100, epochs=100,
                  verbose=1, callbacks=[earlyend], validation_split=.1, shuffle=True)

        # significant improvement in performance in deepFM layer
