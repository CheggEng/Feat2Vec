#file: test_deepfm.py
#provide a test example for Fmachines
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import keras
import re
from matplotlib import pyplot as plt
import feat2vec.deepfm
reload(feat2vec.deepfm)
from feat2vec.deepfm import DeepFM

from keras.callbacks import EarlyStopping
from keras.preprocessing import text,sequence
from keras.layers import Input, GlobalMaxPool1D,Dense,Embedding
from keras.layers.convolutional import Convolution1D
from keras.layers.advanced_activations import PReLU
import nltk
nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
embed_dim = 10

############################
####step 1: plain jane deep FM (with real and discrete features)
############################
np.random.seed(1)
#generate some data
samplesize = 100000
testdata = pd.DataFrame({ 'cat1': np.random.randint(0,9,size=samplesize),
						  'cat2': np.random.randint(0,2,size=samplesize),
						  'real1': np.random.uniform(0,1,size=samplesize),
						  'offset_':np.ones(samplesize)
	})

testdata['latenty'] = (testdata.cat1 - 2*math.pi*testdata.cat2  + testdata.real1 - (testdata.cat1==7)*np.exp(1) +  np.random.normal(size=samplesize) )
#convert to binary indicator
testdata['y'] = (testdata['latenty'] > 0 ).astype('int')



#now apply Deep-Out FM
features = ['cat1','cat2','real1']
feature_dim = [len(testdata['cat1'].unique()),len(testdata['cat2'].unique()),1]
realvals = [False,False,True]


fm_obj = DeepFM(feature_dim, embed_dim,
                 feature_names=features, realval=realvals)


fm=fm_obj.build_model(l2_bias=0.0, l2_factors=0.0, l2_deep=0.0, deep_out=True,
    			    deep_out_bias=True, deep_out_activation = 'linear',
    			    deep_weight_groups = ['cat','cat','real'])
print fm.summary()


train = testdata.iloc[:90000,:]
test = testdata.iloc[90000:,:]
earlyend = EarlyStopping()
fm.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())
inputs = [train['cat1'],train['cat2'],train['real1']]
fm.fit(x=inputs,y=train['y'],batch_size=1000,epochs=100,
	verbose=1,callbacks =[earlyend],validation_split = .1,shuffle=True)


######################################
#example 2: show off deep-in features
######################################

#download some text data, process it, and create some feature extraction layer to plug in
print "processing text..."
samplesize = 2000
reviews = []
labels = []
for rf in movie_reviews.fileids():
	review = movie_reviews.open(rf).read()
	reviews.append(review)
	labels.append(rf.find('pos/')!=-1)

textdata = pd.DataFrame({'text':reviews, 'pos':labels,'offset_':np.ones(len(reviews))})
#pre-process text (do same thing Ralph does leave only consecutive alphabetical characters
textdata['cleantext'] = textdata['text'].map(lambda x: (" ".join(re.findall('[A-Za-z]+',x))).encode('utf8'))
tokens = [i.lower().split(" ") for i in textdata['cleantext']]
textdata['len'] = [len(t) for t in tokens]
textdata.len.describe()
textdata['cat1']=np.random.randint(0,9,size=samplesize)
textdata['cat2']= np.random.randint(0,2,size=samplesize)
textdata['real1']= np.random.uniform(0,1,size=samplesize)
textdata['latenty'] = (textdata.cat1 - 2*math.pi*textdata.cat2  + textdata.real1 - math.exp(1)*textdata.pos.astype('float')  +
					   textdata.real1*textdata.pos.astype('float')  + np.random.normal(size=samplesize) )
#convert to binary indicator
textdata['y'] = (textdata['latenty'] > 0 ).astype('int')

# sequence length cutoff is going to be 75th percentile
cutoff = int(textdata.len.describe()['75%'])
tokens = [r[0:min(len(r),cutoff)] for r in tokens]

#build vocab
vocab = set()
counter=0
for r in tokens:
	for w in r:
		if w not in vocab:
			vocab.add(w)

vocabsize = len(vocab)
vocab_indices= {}
index = 1
for v in vocab:
	vocab_indices[v] = index
	index+=1

tokens_indexed = []
for r in tokens:
	tokens_indexed.append([vocab_indices[w] for w in r])

sequence_mat = sequence.pad_sequences(tokens_indexed,maxlen=cutoff,value=0,padding='post',truncating='post')

#build the feature extraction layer
#do a CNN mimicing ralph's architecture (but of significantly lower dimensionality)
word_seq = Input(batch_shape=(None, sequence_mat.shape[1]), name='wordind_seq')
word_embeddings = Embedding( input_dim = vocabsize+1 ,output_dim = 1,input_length=cutoff,mask_zero=False)(word_seq)
word_conv = Convolution1D(filters=10,kernel_size=3,activation='relu',use_bias=True)(word_embeddings)
pooler=GlobalMaxPool1D()(word_conv)
word_dense_layer=Dense(units=10,activation='relu')(pooler)
word_final_layer=Dense(units=embed_dim,name='textfeats')(word_dense_layer)

#collect relevant valuesfor deepFM model
features = ['cat1','cat2','real1','offset_', 'textseq']
feature_dim = [len(textdata['cat1'].unique()),len(textdata['cat2'].unique()),1,1,embed_dim]
deep_inputs = [word_seq]
deep_feature = [word_final_layer]
deepin = [False,False,False,False,True]
bias_only=[False,False,False,True,False]
realvalued = [False,False,True,False,None] #doesn't matter what we assign to the deep feature, so just say None
inputs = [textdata['cat1'],textdata['cat2'],textdata['real1'],pd.Categorical(textdata['offset_']).codes,
		sequence_mat]


#build deep-in FM
difm_obj = DeepFM(feature_dim, embed_dim,
                 feature_names=features, realval=realvalued,
                 deepin_feature = deepin,
                 deepin_inputs=deep_inputs,deepin_layers=deep_feature)

tf.set_random_seed(1)
np.random.seed(1)
difm=difm_obj.build_model(deep_out=False,
                    bias_only = bias_only,
    			    dropout_input=0,
                    dropout_layer=0)
print difm.summary()
earlyend = EarlyStopping(monitor='val_loss')
difm.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())

difm.fit(x=inputs,y=textdata['y'],batch_size=100,epochs=100,
	verbose=1,callbacks =[earlyend],validation_split = .1,shuffle=True)


#now add a deep-out layer for the interactions
tf.set_random_seed(1)
np.random.seed(1)
diofm=difm_obj.build_model(deep_out=True)
#print diofm.summary()
earlyend = EarlyStopping(monitor='val_loss')
diofm.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())

diofm.fit(x=inputs,y=textdata['y'],batch_size=100,epochs=100,
	verbose=1,callbacks =[earlyend],validation_split = .1,shuffle=True)
#significant improvement in performance in deepFM layer
