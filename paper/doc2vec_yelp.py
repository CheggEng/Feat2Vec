#file: word2vec_imdb.py
#run word2vec on imdb movie data
import pandas as pd
import numpy as np
import gzip
import random
import cPickle
import os,sys
import matplotlib.pyplot as plt
import feat2vec
import gensim
import itertools
import yaml
import keras
import keras.backend as K
from keras.layers import Input, Embedding, Lambda, Activation,Dot, Flatten, Dense, Add
from keras.models import Model
from keras.preprocessing.sequence import skipgrams, make_sampling_table
from gensim.models.word2vec import Word2Vec
constants = yaml.load(open("paper/constants.yaml", 'rU'))
glovedir=constants['wordembeddir']
yelpdir = constants['yelpdir']
negative_samples=5
dim = 100
recreate_docs = False

################
## Doc2Vec in Keras
################

with open(os.path.join(constants['yelpdir'],'dataset/yelp_train_data.p'),'r') as f:
    df=cPickle.load(f)
with open(os.path.join(constants['yelpdir'],'w2v/vocab_map.p'),'r') as f:
    vocab_map = cPickle.load(f)
with open(os.path.join(constants['yelpdir'],'dataset/yelp_textfilter.p'),'r') as f:
    review_filter = cPickle.load(f)


vocabsize = review_filter.num_words
tag_size_dict={}
for c in ['business_id','user_id','stars','funny']:
    tag_size_dict[c] = len(vocab_map[c])
    df[c] = df[c].cat.codes

df.drop(['numOccurences','numBiz','numUser'],axis=1,inplace=True)



#shuffle df
np.random.seed(9)
df = df.sample(frac=1,replace=False)

from keras.preprocessing import sequence
def d2v_generator(df,vocab_size,context_window=5,ns=1, sampling_table=None, shuffle=True,batch_size=1):
    while True:
        if shuffle:
            shuffled_df = df.sample(frac=1,replace=False)
        else:
            shuffled_df = df
        context_length = context_window*2
        label_pattern = [1] + [0]*ns
        for batch in (df.iloc[pos:pos + batch_size] for pos in xrange(0, len(df), batch_size)):
            labels=[]
            contexts=[]
            targets=[]
            users=[]
            bizs = []
            stars=[]
            funnys=[]
            for idx,r in batch.iterrows():
                #print r
                #build the word-context pairs
                words = r['textseq']
                numwords = len(words)
                numsamples = numwords*(ns+1)
                doc_context= []
                targets.append(np.array(words))
                if sampling_table:
                    neg_words = np.random.choice(range(1,vocab_size+1),size=ns*num_words,p=sampling_table)
                else:
                    neg_words = np.random.randint(1,vocab_size,size=ns*numwords)
                targets.append(neg_words)
                for wdx,w in enumerate(words):
                    ctxt = [words[i]   for i in range( wdx-context_window, wdx+context_window+1) if (i!=wdx) and (0 <= i < numwords)]
                    doc_context+= [ctxt]
                doc_context*=(ns+1)
                contexts+=doc_context
                    #labels.append(1)
                    #labels += [0]*ns
                labels.append(np.ones(numwords))
                labels.append(np.zeros(numwords*ns))

                bizs.append(np.repeat(r['business_id'],numsamples))
                users.append(np.repeat(r['user_id'],numsamples))
                stars.append(np.repeat(r['stars'],numsamples))
                funnys.append(np.repeat(r['funny'],numsamples))
            contexts = sequence.pad_sequences(contexts, maxlen=context_length)
            #targets = np.array(targets)
            #labels = np.array(labels)
            bizs = np.concatenate(bizs)
            stars = np.concatenate(stars)
            funnys = np.concatenate(funnys)
            users = np.concatenate(users)
            labels = np.concatenate(labels)
            targets = np.concatenate(targets)
            yield [ [targets,contexts,bizs,users,stars,funnys ], labels]

#wdx=2
#context_window=2
#range( wdx-context_window,wdx) + range(wdx+1,wdx+context_window+1)



context_window=5
dim=50
ns=5
batch_size=10


np.random.seed(9)

from time import time
gen = d2v_generator(df,vocab_size =vocabsize,context_window=context_window,shuffle=True,ns=ns,batch_size=batch_size)

from keras.layers import Concatenate, Average
def build_d2v_dm(dim,vocabsize,tag_size_dict,context_window):
    word = Input(shape=(1,),name='word')
    context =Input(shape=(context_window*2,),name='context')
    tag_inputs=[]
    for t in tag_size_dict.keys():
        tag_inp = Input(shape=(1,),name='%s' % t)
        tag_inputs.append(tag_inp)

    embedding = Embedding(vocabsize,dim,name='embeddings')
    wordvec = embedding(word)
    contextwordvec = embedding(context)
    tag_embeddings = []
    for i,t in enumerate(tag_size_dict.keys()):
        tagvec = Embedding(tag_size_dict[t],dim,name='%s_embedding' % t)(tag_inputs[i])
        tag_embeddings.append(tagvec)

    contextvec = Concatenate(axis=1,name='contextmat')(tag_embeddings+[contextwordvec])
    contextvec = Lambda(lambda x: K.mean(x,axis=1,keepdims=True),name='context_mean')(contextvec)
    #print contextwordvec.shape

    score = Dot(2,name='score')([wordvec,contextvec])
    prob = Flatten()(Dense(1,activation='sigmoid',name='prob',kernel_constraint=keras.constraints.non_neg())(score))
    d2v_dm = Model([word,context]+tag_inputs,prob)
    return d2v_dm

d2v_dm = build_d2v_dm(dim,vocabsize,tag_size_dict,context_window=context_window)
#d2v_dm.get_layer('embeddings').set_weights([glovevecs])
print d2v_dm.summary()

d2v_dm.compile('adam',loss='binary_crossentropy')
from keras.callbacks import ModelCheckpoint
chk = ModelCheckpoint(os.path.join(yelpdir,'w2v/d2v_dm_model.h5'),save_weights_only=True)
d2v_dm.fit_generator(gen,epochs=3,steps_per_epoch=len(df)/batch_size,workers=4,max_queue_size=100,use_multiprocessing=False,callbacks=[chk])
d2v_dm.get_layer('prob').get_weights()
word_vecs = d2v_dm.get_layer('embeddings').get_weights()[0]
word_vecs.shape

reverse_dict = {j:i for i,j in review_filter.word_index.iteritems()}
widx =[None] + [reverse_dict[i] for i in range(1,word_vecs.shape[0])]
len(widx)
word_vecs = pd.DataFrame(word_vecs)
word_vecs['feature'] = 'word'
word_vecs['values'] = widx

allvecs=[word_vecs]
for r in tag_size_dict:
    vec = d2v_dm.get_layer('%s_embedding' % r).get_weights()[0]
    reverse_dict = {j:i for i,j in vocab_map[r].iteritems()}
    eidx = [reverse_dict[i] for i in range(tag_size_dict[r])]
    vec = pd.DataFrame(vec)
    vec['feature'] = r
    vec['values'] = eidx
    allvecs.append(vec)

allvecdf = pd.concat(allvecs)
allvecdf = allvecdf.set_index(['feature','values']).sort_index()
allvecdf.to_csv(os.path.join(yelpdir,'w2v/d2v_dm_embeddings.tsv'),sep='\t',encoding='utf-8')

#########################
####
#w2v in gensim
