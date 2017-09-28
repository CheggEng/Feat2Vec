#file: feat2vec_imdb.py
#run feat2vec on imdb movie data
import pandas as pd
import numpy as np
import gzip
import cPickle
import os
import matplotlib.pyplot as plt
import feat2vec
from feat2vec.feat2vec import Feat2Vec
datadir = '/home/luis/Data/IMDB/'
#datadir = '/media/luis/hdd3/Data/IMDB/'

#load data
with open(os.path.join(datadir,'imdb_movie_data.p'),'r') as f:
    df = cPickle.load(f)

print df.head()

vocab_maps = {} #we will store our maps here from categories/identifiers  to integers
#map categories to integers
#again, mask zero
for c in ['tconst','startYear','runtimeMinutes']:
    df[c] = pd.Categorical(df[c],categories = [r'\N'] + [v for v in pd.unique(df[c]) if v != r'\N'])
    print df[c]
    vocab_dict = dict([(cat,i) for i,cat in enumerate(df[c].cat.categories)])
    vocab_maps[c] = vocab_dict
    df[c] = df[c].cat.codes

#map lists to integer sequences so keras can work with them
#limit to 5 writers, directors, 10 cast members

seqlengths = {'genres':3,'writers':5,'directors':5,'principalCast':10}
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


#normalize realvalued space to be in [0,1]
#df['numVotes'] = df['numVotes']/np.max(df['numVotes'])
#df['averageRating'] /= 10.
#define feature space
model_features = [['tconst'],['startYear'],['isAdult'],['averageRating','mi_rating']]
model_feature_names = ['startYear','isAdult','title','rating']
feature_dimensions = [np.max(df['tconst'])+1,
                      np.max(df['startYear'])+1,
                      1,2]
sampling_features =  [['tconst'],['startYear'],['isAdult'],['averageRating','mi_rating']]
#define some hyperparameters
batch_size=5000
feature_alpha=.25
sampling_alpha=.5
negative_samples=5
dim = 100
reload(feat2vec.feat2vec)
import feat2vec
from feat2vec.feat2vec import Feat2Vec
f2v = Feat2Vec(df=df,model_feature_names=model_feature_names,
    feature_dimensions=feature_dimensions,
    model_features=model_features,
    sampling_features=sampling_features,
    embedding_dim=dim,
    feature_alpha=feature_alpha,sampling_alpha=sampling_alpha,
    negative_samples=negative_samples,  sampling_bias=0,batch_size=batch_size)
print f2v.model.summary()

f2v.fit_model(epochs=1,validation_split=.5)
