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
from keras.preprocessing import sequence
from keras.preprocessing.sequence import skipgrams, make_sampling_table, pad_sequences
from keras.callbacks import ModelCheckpoint
from gensim.models.word2vec import Word2Vec
constants = yaml.load(open("paper/constants.yaml", 'rU'))
glovedir=constants['wordembeddir']
yelpdir = constants['yelpdir']
batch_size=1000
negative_samples=5
dim = 100
recreate_docs = False


#load data
with open(os.path.join(yelpdir,'dataset/yelp_textfilter.p'),'r') as f:
    review_filter = cPickle.load(f)
review_filter.num_words
review_filter.word_counts


#import keras
#model = keras.models.load_model(os.path.join(yelpdir,'model/yelp_f2v_model.h5'))
#w=model.get_layer('word_embedding_layer').get_weights()[0]

if recreate_docs:
    print "Loading Data..."
    print "Loading DF..."
    with open(os.path.join(datadir,'yelp_train_data.p'),'r') as f:
        df=cPickle.load(f)
    df.drop(['numOccurences','numBiz','numUser'],axis=1,inplace=True)
    with open(os.path.join(datadir,'yelp_textfilter.p'),'r') as f:
        review_filter = cPickle.load(f)

    vocab_map = {}
    for c in ['business_id','user_id','stars','funny']:
        vocab_map[c] = dict([(cat,i) for i,cat in enumerate(df[c].cat.categories)])
        df[c] = df[c].cat.codes

    vocab_map['textseq'] = review_filter.word_index
    print df.head()


    with open(os.path.join('/media/luis/hdd3/Data/Yelp/w2v/vocab_map.p'),'w') as f:
        cPickle.dump(vocab_map,f)


    idx_word_dict = {j:i for i,j in review_filter.word_index.iteritems()}
    #create document version of df
    full_docs=[]
    sentence_fcn = lambda x: ' '.join([w for w in x])
    onelist = lambda x: ' '.join([w for w in x]).split(' ')
    doc_batch_size=1000
    seqlengths = {'textseq':250}
    print len(df)/doc_batch_size
    write_option='w'
    for i,batch in enumerate([df[idx:idx+doc_batch_size] for idx in xrange(0,len(df),doc_batch_size)]):
        #export the train DF to a datalist

        newdf = batch.copy()
        sys.stdout.write('\r %g' % i)
        sys.stdout.flush()
        for c in df.columns:
            if c =='textseq':
                null_key = r' \\N'.format(c)
                tag_fcn = lambda x: ' '.join([idx_word_dict[w] for w in x]) + null_key*(seqlengths[c]-len(x))
                trunc_tag_fcn = lambda x: ' '.join([idx_word_dict[w] for w in x[0:seqlengths[c]] ])
                def seq_tag(x):
                    if len(x) <= seqlengths[c]:
                        return tag_fcn(x)
                    else:
                        return trunc_tag_fcn(x)
                def no_pad_seq_tag(x):
                    if len(x) <= seqlengths[c]:
                        return ' '.join([idx_word_dict[w] for w in x])
                    else:
                        return trunc_tag_fcn(x)
                newdf[c] = newdf[c].map(no_pad_seq_tag)
            else:
                tag_fcn = lambda x: c  + "_" + str(x)
                newdf[c] = newdf[c].map(tag_fcn)
        batch_docs = [sentence_fcn(r) for r in newdf.values.tolist()]
        #full_docs += [onelist(r) for r in newdf.values.tolist()]
        with open(os.path.join('/media/luis/hdd3/Data/Yelp/w2v/yelp_docs.txt'),write_option) as f:
            for l in batch_docs:
                f.write(l.encode('ascii','ignore') + '\n')
        write_option='a'

full_docs = []
with open(os.path.join('/media/luis/hdd3/Data/Yelp/w2v/yelp_docs.txt'),'r') as f:
    full_docs = f.readlines()




# for i in range(len(full_docs)):
#     if i % 10000==0:
#         sys.stdout.write('\r %g ' % i)
#         sys.stdout.flush()
#     full_docs[i] = full_docs[i].rstrip('\n').split(' ')

#with open(os.path.join('/media/luis/hdd3/Data/Yelp/w2v/w2v_ready.p'),'w') as f:
#    cPickle.dump(full_docs,f)


with open(os.path.join('/media/luis/hdd3/Data/Yelp/w2v/vocab_map.p'),'r') as f:
    vocab_map = cPickle.load(f)



#split the data
np.random.seed(9)
random.shuffle(full_docs)


#keras

#build the vocab:
word_counts={}
for i,d in enumerate(full_docs):
    if i % 1000==0:
        sys.stdout.write('\r %g' % i )
        sys.stdout.flush()
    for w in d.rstrip('\n').split():
        if w in word_counts:
            word_counts[w]+=1
        else:
            word_counts[w]=1

words,counts = zip(*word_counts.iteritems())
ranks  = np.argsort(-np.array(counts)).argsort()+1



word_dict = {words[i]:ranks[i] for i in range(len(word_counts))}

int_counts = {word_dict[i]:j for i,j in word_counts.iteritems()}

#window=255
window=5
dim=50
batch_size=50
ns=5
vocab_size=len(counts)

vocab_size
sample_probs = np.array([int_counts[i]**(3./4) for i in range(1,vocab_size+1)])
sample_probs /= np.sum(sample_probs)
sample_probs


#make docs
int_docs = []
for i,d in enumerate(full_docs):
    if i % 10000==0:
        sys.stdout.write('\r %g' % i)
        sys.stdout.flush()
    #int_seq = [word_dict[w] for w in d.rstrip('\n').split() if word_counts[w] >=5]
    int_seq = [word_dict[w] for w in d.rstrip('\n').split()]
    int_docs.append(int_seq)


#vocab_size=np.array(counts>=5).sum()
#sample_probs = make_sampling_table(vocab_size+1)

def w2v_sg_generator(doclist):
    while True:
        for d in doclist:
            coup,lab=skipgrams(int_seq,vocabulary_size=vocab_size,window_size=5,negative_samples=1,sampling_table=sample_probs)
            #coup,lab=skipgrams(int_seq,vocabulary_size=vocab_size,window_size=5,negative_samples=1)
            coup = zip(*coup)
            words=np.array(coup[0])
            context=np.array(coup[1])
            yield [words,context],lab

gen=w2v_sg_generator(int_docs)

def build_w2v_sg(vocabsize,dim):
    word = Input(shape=(1,),name='word')
    context =Input(shape=(1,),name='context')
    embedding = Embedding(vocabsize,dim,name='embeddings')

    wordvec = embedding(word)
    contextvec = embedding(context)
    score = Dot(2)([wordvec,contextvec])
    prob = Flatten()(Activation('sigmoid')(score))
    w2v_cbow = Model([word,context],prob)
    return w2v_cbow

#now build W2V model
w2v_sg = build_w2v_sg(vocab_size,dim)
print w2v_sg.summary()
#import hyperdash
#hd_exp = hyperdash.Experiment('W2v SG yelp')
w2v_sg.compile('adam',loss='binary_crossentropy')``
from keras.callbacks import ModelCheckpoint
chk = ModelCheckpoint(os.path.join(yelpdir,'w2v/d2v_model_check.h5'),save_best_only=True)
w2v_sg.fit_generator(gen,epochs=3,steps_per_epoch=len(full_docs),workers=4,max_queue_size=100,use_multiprocessing=False,callbacks=[chk])
w2v_sg.save(os.path.join(yelpdir,'w2v/w2v_sg.h5'))


w2v_sg = keras.models.load_model(os.path.join(yelpdir,'w2v/w2v_sg.h5'))
w2v_embeddings = w2v_sg.get_layer('embeddings').get_weights()[0]
reverse_int_dict = {j:i for i,j in word_dict.iteritems() if j<=vocab_size}
is_tag = lambda w: (len(w.split('_'))>=2)  & (w.startswith('business_id') or w.startswith('user_id') or w.startswith('funny') or w.startswith('stars'))
w2v_idx = [reverse_int_dict[i].rsplit('_',1) if is_tag(reverse_int_dict[i]) else ('word',reverse_int_dict[i]) for i in range(1,vocab_size+1)]
w2v_idx = pd.MultiIndex.from_tuples(w2v_idx,names=['feature','values'])
print w2v_embeddings.shape,len(w2v_idx)
w2v_embeddings_df = pd.DataFrame(w2v_embeddings,index=w2v_idx)
w2v_embeddings_df= w2v_embeddings_df.sort_index()
w2v_embeddings_df
w2v_embeddings_df.to_csv(os.path.join(yelpdir,'w2v/w2v_embeddings.tsv'),sep='\t')

################
## CBOW in Keras
################

def w2v_cbow_generator(docs,vocab_size,context_window=5,ns=1, sampling_table=None,batch_size=1):
    context_length = context_window*2
    label_pattern = [1] + [0]*ns
    ns_choices = np.array(range(1,vocab_size+1))
    while True:
        for batch in (docs[pos:pos + batch_size] for pos in xrange(0, len(docs), batch_size)):
            labels=[]
            contexts=[]
            targets=[]
            for words in batch:
                #print r
                #build the word-context pairs
                numwords = len(words)
                numsamples = numwords*(ns+1)

                doc_context= []
                targets.append(np.array(words))
                if sampling_table is not None:
                    neg_words = np.random.choice(ns_choices,size=ns*numwords,p=sampling_table)
                else:
                    neg_words = np.random.randint(1,vocab_size,size=ns*numwords)
                targets.append(neg_words)
                for wdx,w in enumerate(words):
                    ctxt = [words[i]   for i in range( wdx-context_window, wdx+context_window+1) if (i!=wdx) and (0 <= i < numwords)]
                    doc_context.append(ctxt)
                doc_context *=(ns+1)
                contexts+=doc_context
                labels.append(np.ones(numwords))
                labels.append(np.zeros(numwords*ns))
            contexts = sequence.pad_sequences(contexts, maxlen=context_length)
            labels = np.concatenate(labels)
            targets = np.concatenate(targets)
            yield [ [targets,contexts], labels]



def build_w2v_cbow(dim,vocabsize,context_window):

    word = Input(shape=(1,),name='word')
    context =Input(shape=(context_window*2,),name='context')

    embedding = Embedding(vocabsize,dim,mask_zero=False,name='embeddings')

    wordvec = embedding(word)
    contextvec = embedding(context)

    contextvec = Lambda(lambda x: K.mean(x,axis=1,keepdims=True),name='context_vec')(contextvec)

    score = Dot(2,name='score')([wordvec,contextvec])
    prob = Flatten()(Activation('sigmoid',name='prob')(score))
    w2v_cbow = Model([word,context],prob)
    return w2v_cbow


gen = w2v_cbow_generator(int_docs,vocab_size,context_window=window,ns=5,batch_size=batch_size,sampling_table=None)
s=gen.next()


w2v_cbow = build_w2v_cbow(dim,vocab_size,window)
#print w2v_cbow.summary()
w2v_cbow.compile('adam',loss='binary_crossentropy')

chk = ModelCheckpoint(os.path.join(yelpdir,'w2v/w2v_cbow_model_check.h5'))
w2v_cbow.fit_generator(gen,epochs=3,steps_per_epoch=np.ceil(len(full_docs)/float(batch_size)),workers=4,max_queue_size=100,use_multiprocessing=False,callbacks=[chk])
w2v_cbow.save(os.path.join(yelpdir,'w2v/w2v_cbow.h5'))
w2v_embeddings = w2v_cbow.get_layer('embeddings').get_weights()[0]


reverse_int_dict = {j:i for i,j in word_dict.iteritems() if j<=vocab_size}
is_tag = lambda w: (len(w.split('_'))>=2)  & (w.startswith('business_id') or w.startswith('user_id') or w.startswith('funny') or w.startswith('stars'))
w2v_idx = [reverse_int_dict[i].rsplit('_',1) if is_tag(reverse_int_dict[i]) else ('word',reverse_int_dict[i]) for i in range(1,vocab_size+1)]




len(vocab_map['user_id'])
len([w for w in word_dict if w.startswith('user_id_')])
for t in ['user_id','business_id','stars','funny']:
    print t
    reverse_dict = {j:i for i,j in vocab_map[t].iteritems()}
    w2v_idx = [ [i,reverse_dict[int(j)]] if i==t else [i,j] for i,j in w2v_idx]




w2v_idx = pd.MultiIndex.from_tuples(w2v_idx,names=['feature','values'])
print w2v_embeddings.shape,len(w2v_idx)
w2v_embeddings_df = pd.DataFrame(w2v_embeddings,index=w2v_idx)
w2v_embeddings_df= w2v_embeddings_df.sort_index()
w2v_embeddings_df
w2v_embeddings_df.to_csv(os.path.join(yelpdir,'w2v/w2v_cbow_embeddings.tsv'),sep='\t')
