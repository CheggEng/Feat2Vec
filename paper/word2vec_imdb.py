#file: word2vec_imdb.py
#run word2vec on imdb movie data
import pandas as pd
import numpy as np
import gzip
import cPickle
import os,sys
import matplotlib.pyplot as plt
import feat2vec
import gensim
import itertools
from gensim.models.word2vec import Word2Vec
#datadir = '/home/luis/Data/IMDB/'
datadir = '/media/luis/hdd3/Data/IMDB/'
batch_size=1000
negative_samples=5
dim = 50
np.random.seed(9)
recreate_docs = True
skipgram=False
#load data
with open(os.path.join(datadir,'imdb_train_movie_data.p'),'r') as f:
    df = cPickle.load(f)


#drop irrelevant columns not used for training
df.drop(['tconst','endYear','primaryTitle','originalTitle'],axis=1,inplace=True)
print df.head()

#split the data
validation_split =.1
validation_index = np.random.choice(df.index,size=int(len(df)*validation_split),replace=False)
train_index = [x for x in df.index if x not in validation_index]

valdf = df.loc[validation_index,:]
traindf= df.loc[train_index,:]
#create df as a document for training and testing

exclude_tokens = set(['isAdult_0','mi_rating_0'])
sentence_fcn = lambda x: ' '.join([w for w in x if w not in exclude_tokens])

#create document version of df
docs = []
seqlengths = {'genres':3,'writers':5,'directors':5,'principalCast':10,'titleSeq':5}
for i,df in enumerate([traindf,valdf,df]):
    #export the train DF to a datalist
    newdf = df.copy()
    for c in df.columns:
        print c
        if c in seqlengths.keys():
            null_key = r' {}_\N'.format(c)
            tag_fcn = lambda x: ' '.join([c  + "_" + str(w) for w in x]) + null_key*(seqlengths[c]-len(x))
            trunc_tag_fcn = lambda x: ' '.join([c  + "_" + str(w) for w in x[0:seqlengths[c]] ])
            def seq_tag(x):
                if len(x) <= seqlengths[c]:
                    return tag_fcn(x)
                else:
                    return trunc_tag_fcn(x)
            newdf[c] = newdf[c].map(seq_tag)
        else:
            tag_fcn = lambda x: c  + "_" + str(x)
            newdf[c] = newdf[c].map(tag_fcn)
    print "generating sentences..."
    docs.append([sentence_fcn(r) for r in newdf.values.tolist()])
    docs[i] = [s.split(' ') for s in docs[i]]
    print newdf.head(1)
    print docs[i][0]



#define some impt fcns
def gen_pairs(doc):
    '''
    returns all unordered unique pairs in a list
    '''
    return list(itertools.combinations(doc, 2))
def doc_similarity(doc,model):
    '''
    calculates average cosine similarity between all pairs in a document
    '''
    pairs = gen_pairs(doc)
    return np.mean([model.similarity(p[0],p[1]) if p[0] in model and p[1] in model else 0 for p in pairs])

def dev_loss(doc_list,model,num_samples=None):
    '''
    estimates the loss (-E[similarity]) for a document list
    '''
    if num_samples is None:
        doc_inds = range(len(doc_list))
    else:
        doc_inds = np.random.choice(range(len(doc_list)),size=num_samples,replace=False)
    dev_sims=[]
    for iter,ind in enumerate(doc_inds):
        if iter%100==0:
            sys.stdout.write("\r Evaluated {s}/{ns} Docs".format(s=iter,ns=len(doc_inds)))
            sys.stdout.flush()
        dev_sims.append(doc_similarity(doc_list[ind],model))
    sys.stdout.write("\n")
    return -np.mean(dev_sims)


traindocs = docs[0]
valdocs = docs[1]
fulldocs=docs[2]

num_epochs = 10
#window = len(df.columns) + np.sum([seqlengths[x]-1 for x in seqlengths.keys()])
window = np.max([len(x) for x in fulldocs])
print window

#begin training
w2v = Word2Vec(size=dim, window=window,
 min_count=0, max_vocab_size=None, sample=0.,
 seed=9, workers=3,
 alpha=0.025,min_alpha=0.0001,
 sg=skipgram, hs=0, negative=negative_samples, cbow_mean=0,
 iter=1, sorted_vocab=1, batch_words=1000, compute_loss=False)
print w2v.iter
print "Building vocab..."
w2v.build_vocab(fulldocs)
print "Beginning Training..."
dev_losses=[]
dev_losses.append(0)
for i in range(num_epochs):
    w2v.train(traindocs,epochs=w2v.iter,total_examples=len(traindocs))
    #evaluate
    loss = dev_loss(valdocs,w2v)
    dev_losses.append(loss)
    print "Epoch {i}: {l}".format(i=i,l=loss)
    if loss > dev_losses[len(dev_losses)-2]:
        print "Exiting at Epoch {} due to decrease in Validation Similarity!!!".format(i)
        break



opt_epoch = np.argmin(dev_losses)

print "Now Completing full training on full corpus ({} epochs)...".format(opt_epoch)
w2v = Word2Vec(sentences=fulldocs,size=dim, window=window,
 min_count=0, max_vocab_size=None, sample=0.,
 seed=9, workers=3,
 alpha=0.025,min_alpha=0.0001,
 sg=skipgram, hs=0, negative=negative_samples, cbow_mean=0,
 iter=opt_epoch, sorted_vocab=1, batch_words=1000, compute_loss=False)

print "exporting embeddings..."
w2v.save(os.path.join(datadir,'w2v_gensim_model'))
w2v.wv.save_word2vec_format(os.path.join(datadir,'w2v_vectors.txt'))
