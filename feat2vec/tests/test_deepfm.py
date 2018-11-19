# file: test_deepfm.py
# provide a test example for Fmachines

from feat2vec import DeepFM, Feat2VecModel
import numpy as np
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.layers import Input, GlobalMaxPool1D, Dense, Embedding, Lambda, Reshape, concatenate
from keras.constraints import  non_neg
from keras.layers.convolutional import Convolution1D
import keras.backend as K
import tensorflow as tf
import pandas as pd
import math
import re

from unittest import TestCase


class TestDeepFM(TestCase):
    def test_rai_faster(self):
        dimensions = 10
        EMBEDING_DIM = 5



        feature_names = ["principal", "f1", "f2", "f3", "f4", "f5"]



        # Principal id

        principal_input = Input(batch_shape=(None, 1), name="principal")
        int = Embedding(input_dim=EMBEDING_DIM, output_dim=dimensions, name="embedding_principal" )(principal_input)
        principal_embedding = int # Reshape((dimensions,1))(int)

        # DEFINE FM MACHINE:
        feature_specification = []
        for feat in feature_names:
            if feat == "principal":
                feature_specification.append({"name": "principal",
                                              "type": {"input": principal_input,
                                                       "output": principal_embedding}})

            elif feat == "f1":
                feature_specification.append({"name": feat,
                                              "type": "real"})
            else:
                feature_specification.append({"name": feat,
                                              "type": "discrete",
                                              "len": 10,
                                              "vocab": 100
                                              })
        fm = Feat2VecModel(features=feature_specification,
                           mask_zero=True,
                           obj='ns')

        groups = []
        groups.append( [("principal", feature_names)] )
        groups.append( [("f1" , feature_names)] )
        print groups

        keras_model_collapsed = fm.build_model(dimensions,
                                               collapsed_type=2,
                                               deep_out=True,
                                               deep_out_bias=False,
                                               deep_weight_groups=groups,
                                               dropout_layer=0.5,
                                               dropout_input=0.1
                                               )

        keras_model_notcollapsed = fm.build_model(dimensions,
                                                  collapsed_type=None,
                                                  deep_out=True,
                                                  deep_out_bias=False,
                                                  deep_weight_groups=groups,
                                                  dropout_layer=0.5,
                                                  dropout_input=0.1
                                                  )


        #f5 = keras_model_notcollapsed.get_layer("dropout_grp_000")
        #assert f5.rate > 0.
        #f5.rate = 0.
        try:
            from keras.utils import plot_model
            plot_model(keras_model_collapsed, to_file="rai_faster_collapsed.png")
            plot_model(keras_model_notcollapsed, to_file="rai_faster_notcollapsed.png")
        except:
            pass

    def test_rai(self):
        dimensions = 10
        dup_dimension = 20
        EMBEDING_DIM = 5
        SEC_EMBEDDING = 5


        feature_names = ["principal", "f1", "f2", "f3", "f4", "f5"]

        # DEFINE CUSTOM LAYERS
        skream_rotator = Dense(units=dimensions, activation="linear", use_bias=False,name="rotator")

        # Popularity
        f1_input = Input(batch_shape=(None, 1), name="f1")
        popularity_embed_inter = Embedding(input_dim=EMBEDING_DIM, output_dim=dup_dimension, name="embedding_dup",
                                           mask_zero=True)(f1_input)
        popularity_unmasker = Lambda(lambda x: x, name='unmasker_dup')(popularity_embed_inter)
        popularity_embed1 = Dense(units=dimensions, activation="linear", use_bias=False, name="resized_embedding")(
            popularity_unmasker)
        f1_embed = Reshape((dimensions,))(popularity_embed1)

        # Principal id

        principal_input = Input(batch_shape=(None, 1), name="principal")
        principal_embedding = Embedding(input_dim=EMBEDING_DIM, output_dim=SEC_EMBEDDING, name="embedding_principal",
                                    mask_zero=True)(principal_input)
        principal_makser = Lambda(lambda x: x, name='unmasker_principal')(principal_embedding)
        principal_reshape = Reshape((SEC_EMBEDDING,))(principal_makser)

        merged_principal = concatenate([principal_reshape, f1_embed])
        rotated_principal = Reshape((dimensions,), name="rotated_principal")(skream_rotator(merged_principal))

        # F2
        f2_input = Input(batch_shape=(None, 20), name="f2")
        f2_temp = Embedding(input_dim=5, input_length=20,output_dim=SEC_EMBEDDING, mask_zero=True, name="embedding_f2")(f2_input)
        avg_f2 = Reshape((SEC_EMBEDDING,))(Lambda(lambda x: K.sum(x, axis=1, keepdims=True), name="avg_f2_embedding")(f2_temp))

        merged_f2 = concatenate([avg_f2, f1_embed])
        f2_embed = Reshape((dimensions,), name="rotated_f2")(skream_rotator(merged_f2))

        # DEFINE FM MACHINE:
        feature_specification = []
        for feat in feature_names:
            if feat == "principal":
                feature_specification.append({"name": "principal",
                                              "type": {"input": principal_input,
                                                       "output": rotated_principal}})
            elif feat == "f1":
                feature_specification.append({"name": "f1",
                                              "type": {"input": f1_input,
                                                       "output": f1_embed}})
            elif feat == "f2":
                feature_specification.append({"name": feat,
                                              "type": {"input": f2_input,
                                                       "output": f2_embed}})
            elif feat == "f3":
                feature_specification.append({"name": feat,
                                              "type": "real"})
            else:
                feature_specification.append({"name": feat,
                                              "type": "discrete",
                                              "len": 10,
                                              "vocab": 100
                                              })
        fm = Feat2VecModel(features=feature_specification,
                           mask_zero=True,
                           obj='ns')

        groups = []
        groups.append( zip(["principal"] * (len(feature_names)-1), feature_names[1:]) )
        groups.append( zip(["f1"] * (len(feature_names)-2), feature_names[2:]) )
        groups.append([ ("f1", "principal") ] )
        print groups
        keras_model = fm.build_model(dimensions,
                                     deep_out=True,
                                     deep_out_bias=False,
                                     deep_weight_groups=groups,
                                     dropout_layer=0.5,
                                     dropout_input=0.1
                                     )

        #f5 = keras_model.get_layer("dropout_embedding_f5")
        #assert f5.rate > 0.
        #f5.rate = 0.
        try:
            from keras.utils import plot_model
            plot_model(keras_model, to_file="rai.png")
        except:
            pass

    def test_easy(self):
        f2vm = Feat2VecModel([{"name": "disc1",
                               "type": "discrete",
                               "vocab": 10}, # disc1 has a vocabulary of 10 words
                              {"name": "cont1",
                               "type": "real"}, #cont1 is a real number
                              {"name": "disc100",
                               "type": "discrete",
                               "len": 100,
                               "vocab": 10}, # disc100 is a sequence of 100 words, with a vocabulary of 10 words
                              {"name": "cont100",
                               "len": 100,
                               "type": "real"}, #cont100 is a sequence of 100 numbers
                             ])
        keras_model = f2vm.build_model(embedding_dimensions=5)
        keras_model.compile(loss='binary_crossentropy', optimizer=tf.train.AdamOptimizer())



    def test_easy_in(self):
        import keras.layers
        # define custom layer:
        input_layer = keras.layers.Input( shape=(1,) )
        custom_layer = keras.layers.Embedding(input_dim=1, output_dim=5)(input_layer) #keeping it simple here
        # define feat2vec model
        f2vm = Feat2VecModel([{"name": "disc1", # 'disc1 is a (single) discrete feature with a vocabulary of 10 words
                               "type": "discrete",
                               "vocab": 10},
                              {"name": "cont1", # 'cont1' is a sequence of 20 real numbers
                               "type": "real",
                               "len":  20},
                              {"name": "custom", # 'custom' is a custom-type feature
                               "type": {"input": input_layer,
                                        "output": custom_layer}}
                             ])
        keras_model = f2vm.build_model(embedding_dimensions=5) # this returns a Keras network
        keras_model.compile(loss='binary_crossentropy', optimizer=tf.train.AdamOptimizer())




    def test_all_pairwise(self):

        feature_names = ["f1", "f2", "f3", "f5", "f6", "f7", "f8",
                         "f9", "f10", "f11"]

        fm = DeepFM(model_features=[["f1", ],
                                    ["f2"],
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
        model.fit( x=[np.array([0]),
                      np.array([0]),
                      np.array([[51, 2]]),
                      np.array([[0, 0]]),
                      np.array([[25, 1]]),
                      np.array([[0, 0]]),
                      np.array([[17, 1]]),
                      np.array([[1, 1]]),
                      np.array([[1, 1]]),
                      np.array([[0, 0]])],
                  y=np.array([0]))
        try:
            from keras.utils import plot_model
            plot_model(model, to_file="all_pairwise.png")
        except:
            pass


    def test_some_pairwise(self):

        feature_names = ["f1", "f2", "f3", "f5", "f6", "f7", "f8",
                         "f9", "f10", "f11"]

        fm = DeepFM(model_features=[["f1" ],
                                    ["f2"],
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
        groups = zip(["f1"] * (len(feature_names)-1), feature_names[1:])
        print groups
        model = fm.build_model(10,
                               dropout_layer=0.5,
                               deep_out=True,
                               deep_out_bias=False,
                               deep_weight_groups=[groups],
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
        embed_dim = 10
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
        embed_dim = 10

        word_seq = Input(batch_shape=(None, sequence_mat.shape[1]), name='wordind_seq')
        word_embeddings = Embedding(input_dim=vocabsize + 1, output_dim=1, input_length=cutoff, mask_zero=False)(
            word_seq)
        word_conv = Convolution1D(filters=10, kernel_size=3, activation='relu', use_bias=True)(word_embeddings)
        pooler = GlobalMaxPool1D()(word_conv)
        word_dense_layer = Dense(units=10, activation='relu')(pooler)
        word_final_layer = Dense(units=embed_dim, name='textfeats')(word_dense_layer)
        word_final_layer = Reshape( (1, 10))(word_final_layer)

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

        try:
            from keras.utils import plot_model
            plot_model(difm, to_file="difm.png")
        except:
            pass

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
