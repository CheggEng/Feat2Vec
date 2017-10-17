#file: feat2vec_imdb.py
#run feat2vec on imdb movie data
import pandas as pd
import numpy as np
import gzip
import cPickle
import os
import matplotlib.pyplot as plt
import feat2vec
import gensim
datadir = '/home/luis/Data/IMDB/'
#datadir = '/media/luis/hdd3/Data/IMDB/'
batch_size=1000
negative_samples=5
dim = 50
np.random.seed(9)
recreate_docs = True
#load data
with open(os.path.join(datadir,'imdb_train_movie_data.p'),'r') as f:
    df = cPickle.load(f)


#split the data
validation_split =.1
validation_index = np.random.choice(df[df['dirOccurences']>1].index,size=int(len(df)*validation_split),replace=False)
train_index = [x for x in df.index if x not in validation_index]

valdf = df.loc[validation_index,:]
traindf= df.loc[train_index,:]
#output df as a document for training and testing
exclude_tokens = set(['isAdult_0'])
#
