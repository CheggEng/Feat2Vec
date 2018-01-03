import pandas as pd
import numpy as np
import gzip
import json
import cPickle
import os
import sys
sys.path.append('feat2vec/')
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
import feat2vec
from feat2vec.feat2vec import Feat2Vec



### Load DF
datadir = '/home/luis/Data/Yelp/dataset/'
print "Loading DF..."
with open(os.path.join(datadir,'yelp_train_data.p'),'r') as f:
    traindf=cPickle.load(f)
with open(os.path.join(datadir,'yelp_textfilter.p'),'r') as f:
    review_filter = cPickle.load(f)

vocab_map = {}
for c in ['business_id','user_id','stars']:
    vocab_map[c] = dict([(cat,i) for i,cat in enumerate(traindf[c].cat.categories)])
    traindf[c] = traindf[c].cat.codes

#move the star var to 0-1
# for c in ['stars']:
#     vocab_map[c] = {'max':np.max(traindf[c]),'min':np.min(traindf[c])}
#     print c,vocab_map[c]
#     traindf[c] = ( traindf[c] - np.min(traindf[c]) ) / (np.max(traindf[c]) - np.min(traindf[c]) )

vocab_map['textseq'] = review_filter.word_index
#traindf = traindf.iloc[0:100000,:]

### Hyperparameters
batch_size=2000
feature_alpha=0.5
sampling_alpha=.75
negative_samples=5
dim = 50
text_embedding_size=25
text_max_len = 250

### Filter text
#review_pd['textseq'] = review_filter.texts_to_sequences(review_pd.text.values.tolist())
#textmat= keras.preprocessing.sequence.pad_sequences(review_pd['textseq'].values.tolist(),maxlen=text_max_len,value=0,padding='post',truncating='post')
#review_pd['textseq'] = [textmat[i,:] for i in range(len(review_pd))]
#num_words = len(review_filter.word_index) +1
#print num_words
def process_keras_text(df):
    #seq = review_filter.texts_to_sequences(df.textseq.values.tolist())
    mat = keras.preprocessing.sequence.pad_sequences(df.textseq.values.tolist(),maxlen=text_max_len,value=0,padding='post',truncating='post')
    return mat

print "creating sequences of text"
traindf['textseq'] = review_filter.texts_to_sequences(traindf.text.values.tolist())
traindf['textseq'] = traindf['textseq'].map(lambda x: tuple(x))
#### Build NLP keras layer
text_input = Input(batch_shape=(None, text_max_len), name='wordind_seq')
word_embed_layer=keras.layers.Embedding(input_dim=review_filter.num_words, output_dim=text_embedding_size,
    input_length=text_max_len, mask_zero=False, name='word_embedding_layer')(text_input)
conv_layer=keras.layers.convolutional.Convolution1D(filters=100,
    kernel_size=3,activation='relu',name='conv')(word_embed_layer)
maxpool_layer=keras.layers.GlobalMaxPool1D()(conv_layer)
text_embed_layer=keras.layers.Dense(dim,name='embedding_textseq',activation='linear')(maxpool_layer)
#build deep Ratings layer
# star_input = Input(batch_shape=(None,1),name='input_rating')
# star_deep = keras.layers.Dense(dim,name='intermed_rating',activation='sigmoid')(star_input)
# star_embed =  keras.layers.Dense(dim,name='embedding_stars',activation='linear')(star_deep)

### define F2V params
model_features = [['business_id'],['user_id'],['stars'],['textseq']]
is_deepin_feature=[False,False,False,True]
realvalued=[False,False,False,False]
model_feature_names = ['business_id','user_id','stars', 'textseq']
feature_dimensions = [ len(vocab_map['business_id'].keys()),
                       len(vocab_map['user_id'].keys()),
                       len(vocab_map['stars'].keys()),None]
sampling_features = [f for f in model_features]
custom_formats=[None,None,None,process_keras_text]
#deep_input_layers=[star_input,text_input]
#deepin_layers = [star_embed,text_embed_layer]
deep_input_layers=[text_input]
deepin_layers = [text_embed_layer]

print traindf.head()

### Create F2V object
reload(feat2vec.feat2vec)
reload(feat2vec.deepfm)
reload(feat2vec.implicitsampler)
import feat2vec
from feat2vec.feat2vec import Feat2Vec
f2v = Feat2Vec(df=traindf,model_feature_names=model_feature_names,
    feature_dimensions=feature_dimensions,
    model_features=model_features,
    realvalued=realvalued,
    sampling_features=sampling_features,
    embedding_dim=dim,
    dropout=0.,
    deepin_feature = is_deepin_feature,
    deepin_inputs=deep_input_layers,deepin_layers=deepin_layers,
    feature_alpha=feature_alpha,sampling_alpha=sampling_alpha,
    custom_formats=custom_formats,
    negative_samples=negative_samples,
    sampling_bias=0,batch_size=batch_size)

print f2v.model.summary()
earlyend = EarlyStopping(patience=0,monitor='val_loss')
chkpoint = ModelCheckpoint(filepath=os.path.join(datadir,'checkpoint.h5'),monitor='val_loss',
    verbose=1,save_best_only=True,save_weights_only=True)
callbacks=[earlyend,chkpoint]
f2v.fit_model(epochs=25,validation_split=1./9.,
    callbacks = callbacks)
f2v.model.load_weights(os.path.join(datadir,'checkpoint.h5'))


#save learned embeddings from model
embeddings = f2v.get_embeddings(vocab_map)
print embeddings
embeddings.to_csv(os.path.join(datadir,'yelp_embeddings.tsv'),sep='\t',index=False)


#save the vocab_maps to a pickle file to infer categories later
with open(os.path.join(datadir,'f2v_vocab_map.p'),'w') as f:
    cPickle.dump(vocab_map,f)

#dump the trained model as well.
f2v.model.save(os.path.join(datadir,'yelp_f2v_model.h5'))
