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
from keras.layers import Concatenate, Average

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
    prob = Flatten()(Dense(1,activation='sigmoid',name='prob')(score))
    d2v_dm = Model([word,context]+tag_inputs,prob)
    return d2v_dm
