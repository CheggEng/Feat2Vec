#file: feat2vec_imdb.py
#run feat2vec on imdb movie data
import pandas as pd
import numpy as np
import gzip
import cPickle
import os
import matplotlib.pyplot as plt
import feat2vec
import keras
reload(keras)
from keras.callbacks import EarlyStopping
from keras.layers import Input,Dense
from feat2vec.feat2vec import Feat2Vec
datadir = '/home/luis/Data/IMDB/'
#datadir = '/media/luis/hdd3/Data/IMDB/'
batch_size=1000
feature_alpha=.25
sampling_alpha=.5
negative_samples=5
dim = 50
np.random.seed(9)
deepin_features = [ ['runtimeMinutes'], ['averageRating','mi_rating'],['numVotes','mi_rating']]

#load data
with open(os.path.join(datadir,'imdb_train_movie_data.p'),'r') as f:
    df = cPickle.load(f)

print df.head()
vocab_maps = {} #we will store our maps here from categories/identifiers  to integers
#map categories to integers
#again, mask zero


for c in ['tconst','startYear']:
    df[c] = pd.Categorical(df[c],categories = [r'\N'] + [v for v in pd.unique(df[c]) if v != r'\N'])
    print df[c].head()
    vocab_dict = dict([(cat,i) for i,cat in enumerate(df[c].cat.categories)])
    vocab_maps[c] = vocab_dict
    df[c] = df[c].cat.codes

#normalize continuous vars to be 0 to 1
df.dtypes
df.loc[df['runtimeMinutes']==r'\N','runtimeMinutes'] = '0'
df['runtimeMinutes'] = pd.to_numeric(df['runtimeMinutes'])
for c in ['runtimeMinutes', 'averageRating','numVotes']:
    vocab_maps[c] = {'max':np.max(df[c]),'min':np.min(df[c])}
    print c,vocab_maps[c]
    df[c] = ( df[c] - np.min(df[c]) ) / (np.max(df[c]) - np.min(df[c]) )

#map lists to integer sequences so keras can work with them
#limit to 5 writers, directors, 10 cast members

seqlengths = {'genres':3,'writers':5,'directors':5,'principalCast':10,'titleSeq':5}
for c in seqlengths.keys():
    print 'transferring {} to integer sequence '.format(c)
    #generate the vocab
    vocab_dict = {r'\N':0} #set null entries to zero so we can mask them
    vocab = set([r'\N']) #temporary set used for faster parsing
    nextInteger = 1
    for i in df[c]:
        for w in i:
            if w not in vocab:
                vocab.add(w)
                vocab_dict[w] = nextInteger
            	nextInteger+=1
    vocab_maps[c] = vocab_dict
    print "Vocab size:" , len(vocab)
    #now create the integer sequences
    for idx in range(seqlengths[c]):
        df['{c}_{ind}'.format(c=c,ind=idx+1)] = df[c].map(lambda x: vocab_dict[x[idx]] if len(x) >=idx+1 else 0)
    #print vocab_dict
    #print df[[c] + ['{c}_{ind}'.format(c=c,ind=idx+1) for idx in range(seqlengths[c])]].head()

##create deepin features (relu layers)
deep_input_layers=[]
deep_embed_layers=[]
for c in deepin_features:
    deep_input  = Input(batch_shape=(None, len(c)), name='input_{}'.format(c[0]))
    deep_input_layers.append(deep_input)
    deeplayer = Dense(units=dim,activation='relu',use_bias=False,name='intermed_{}'.format(c[0]))(deep_input)
    deeplayer = Dense(units=dim,activation='relu',use_bias=False,name='embed_{}'.format(c[0]))(deeplayer)
    deep_embed_layers.append(deeplayer)



#normalize realvalued space to be in [0,1]
#df['numVotes'] = df['numVotes']/np.max(df['numVotes'])
#df['averageRating'] /= 10.
#define feature space
titlecols = [c for c in df.columns if c.startswith('titleSeq_')]
genrecols = [c for c in df.columns if c.startswith('genres_')]
castcols = [c for c in df.columns if c.startswith('principalCast_')]
directorcols = [c for c in df.columns if c.startswith('directors_')]
writercols = [c for c in df.columns if c.startswith('writers_')]
model_features = [titlecols,['startYear'],['isAdult'],['runtimeMinutes'],['averageRating','mi_rating'],['numVotes','mi_rating'],
                  genrecols,castcols,directorcols,writercols]
is_deepin_feature=[False,False,False,True,True,True,False,False,False,False]
model_feature_names = ['titleSeq','startYear','isAdult','runtimeMinutes', 'averageRating','numVotes','genres','principalCast','directors','writers']

feature_dimensions = [ len(vocab_maps['titleSeq'].keys()),
                       len(vocab_maps['startYear'].keys()),
                       1,None,None,None,
                      len(vocab_maps['genres'].keys()),
                      len(vocab_maps['principalCast'].keys()),
                      len(vocab_maps['directors'].keys()),
                      len(vocab_maps['writers'].keys())]
sampling_features =  [titlecols,['startYear'],['isAdult'],['runtimeMinutes'],['averageRating','numVotes','mi_rating'],
                      genrecols,castcols,directorcols,writercols]

#Calculate the step1 probs manually due to deepin
param_counts=np.zeros(len(sampling_features))
param_counts[sampling_features.index(titlecols)] = len(vocab_maps['titleSeq'])*dim
param_counts[sampling_features.index(genrecols)] = len(vocab_maps['genres'])*dim
param_counts[sampling_features.index(castcols)] = len(vocab_maps['principalCast'])*dim
param_counts[sampling_features.index(directorcols)] = len(vocab_maps['directors'])*dim
param_counts[sampling_features.index(writercols)] = len(vocab_maps['writers'])*dim
param_counts[sampling_features.index(['startYear'])] = len(vocab_maps['startYear'])*dim
param_counts[sampling_features.index(['isAdult'])] = dim
param_counts[sampling_features.index(['runtimeMinutes'])] = 1*dim + dim*dim
param_counts[sampling_features.index(['averageRating','numVotes','mi_rating'])] = 2*(1*dim + dim*dim)
print param_counts
init_probs = np.power(param_counts,sampling_alpha)
init_probs /= np.sum(init_probs)
print init_probs

#define some hyperparameters
earlyend = EarlyStopping(patience=0,monitor='val_loss')
callbacks=[earlyend]
reload(feat2vec.feat2vec)
reload(feat2vec.deepfm)
import feat2vec
from feat2vec.feat2vec import Feat2Vec
f2v = Feat2Vec(df=df,model_feature_names=model_feature_names,
    feature_dimensions=feature_dimensions,
    model_features=model_features,
    sampling_features=sampling_features,
    embedding_dim=dim,
    dropout=0.,
    mask_zero=True,
    deepin_feature = is_deepin_feature,
    deepin_inputs=deep_input_layers,deepin_layers=deep_embed_layers,
    step1_probs = init_probs,
    feature_alpha=feature_alpha,sampling_alpha=sampling_alpha,
    negative_samples=negative_samples,  sampling_bias=0,batch_size=batch_size)

print f2v.model.summary()
history = f2v.fit_model(epochs=25,validation_split=.1,
    callbacks = callbacks)
#dir(f2v.model.history)
opt_epoch = len(f2v.model.history.epoch)
print "Optimal Epochs: ", opt_epoch
f2v = Feat2Vec(df=df,model_feature_names=model_feature_names,
    feature_dimensions=feature_dimensions,
    model_features=model_features,
    sampling_features=sampling_features,
    embedding_dim=dim,
    dropout=0.,
    mask_zero=True,
    deepin_feature = is_deepin_feature,
    deepin_inputs=deep_input_layers,deepin_layers=deep_embed_layers,
    step1_probs = init_probs,
    feature_alpha=feature_alpha,sampling_alpha=sampling_alpha,
    negative_samples=negative_samples,  sampling_bias=0,batch_size=batch_size)

f2v.fit_model(epochs=opt_epoch,validation_split=None)
#opt_epochs =
f2v.model.save(os.path.join(datadir,'f2v_imdb.h5'))

embeddings = f2v.get_embeddings(vocab_maps)
embeddings.to_csv(os.path.join(datadir,'imdb_movie_embeddings.tsv'),sep='\t',index=False)

#save the vocab_maps to a pickle file to infer categories later
with open(os.path.join(datadir,'f2v_vocab_map.p'),'w') as f:
    cPickle.dump(vocab_maps,f)
