# evaluate_vec_similarity.py
# eval the performance of w2v and f2v as reccomender systems
# for the following question:
# given a cast, who is the most likely director?
# use pretrained vectors from both on a left out dataset
import numpy  as np
import pandas as pd
import sys,os
sys.path.append('feat2vec/')
import cPickle
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import feat2vec
datadir=''
outputdir= 'paper/output/'
#load both sets of vectors
print "Loading w2v/f2v embeddings..."
import sys
redo_w2v = False
if redo_w2v:
    w2v = KeyedVectors.load_word2vec_format(os.path.join(datadir,'w2v_vectors.txt'), binary=False)

f2v = pd.read_csv(os.path.join(datadir,'imdb_movie_embeddings.tsv'),sep='\t')
f2v = f2v.set_index(['feature','values'])

print f2v.head()

print "Loading df..."
with open(os.path.join(datadir,'imdb_test_movie_data.p'),'r') as f:
    testdf = cPickle.load(f)

print testdf.head()

#create document list vsn of test dataframe
print "cleaning df..."
seqlengths = {'genres':3,'writers':5,'directors':5,'principalCast':10,'titleSeq':5}
if redo_w2v:
    tempdf = testdf.copy()
    exclude_tokens = set(['isAdult_0','mi_rating_0'])
    sentence_fcn = lambda x: ' '.join([w for w in x if w not in exclude_tokens])

    for c in testdf.columns:
        #print c
        if c in seqlengths.keys():
            null_key = r' {}_\N'.format(c)
            tag_fcn = lambda x: ' '.join([c  + "_" + str(w) for w in x]) + null_key*(seqlengths[c]-len(x))
            trunc_tag_fcn = lambda x: ' '.join([c  + "_" + str(w) for w in x[0:seqlengths[c]] ])
            def seq_tag(x):
                if len(x) <= seqlengths[c]:
                    return tag_fcn(x)
                else:
                    return trunc_tag_fcn(x)
            tempdf[c] = tempdf[c].map(seq_tag)
        else:
            tag_fcn = lambda x: c  + "_" + str(x)
            tempdf[c] = tempdf[c].map(tag_fcn)
    test_docs = [sentence_fcn(r) for r in tempdf.values.tolist()]
    test_docs = [s.split(' ') for s in test_docs]


#edit the testdf to be correct seq lengths
print testdf['directors'].head()
for c in testdf.columns:
    null_key = r'\N'
    if c in seqlengths.keys():
        print c
        tag_fcn = lambda x: x  + [null_key]*(seqlengths[c]-len(x))
        trunc_tag_fcn = lambda x:  x[0:seqlengths[c]]
        def seq_tag(x):
            if len(x) <= seqlengths[c]:
                return tag_fcn(x)
            else:
                return trunc_tag_fcn(x)
        testdf[c] = testdf[c].map(seq_tag)

print testdf['directors'].head()
testdf = testdf.reset_index(drop=True)

def rank_byfeature_w2v(doc_list,target_feature,source_feature):
    '''
    Rank a set of features associated with a given vetor.
    source_feature is the feature in docs you want to use as an input to calculate the vector. this is the "labeled info"
    target_feature is the feature type you want to rank against
    '''
    target_tokens = [i for i in w2v.vocab if i.startswith(target_feature + '_')]
    target_vecs = np.concatenate([np.expand_dims(w2v[i],axis=0) for i in target_tokens],axis=0)
    print target_vecs.shape
    rankings=[]
    for i,doc in enumerate(doc_list):
        #print doc
        #print  [ w for w in doc if w.startswith(target_feature + '_') ]
        targetid = ( w for w in doc if w.startswith(target_feature + '_') ).next()
        if targetid not in w2v.vocab:
            print "skipping {} since OOV".format(targetid)
            continue
        y = np.zeros(len(target_tokens))
        y[target_tokens.index(targetid)] = 1
        sourcevec = np.sum([w2v[w] for w in doc if w.startswith(source_feature+'_') if w in w2v.vocab],axis=0)[:,np.newaxis]
        #print sourcevec.shape
        scores = cosine_similarity(sourcevec.T,target_vecs)[0]
        temp = np.argsort(-scores)
        ranks = np.empty(len(scores), int)
        ranks[temp] = np.arange(len(scores))
        rank=ranks[target_tokens.index(targetid)]
        sys.stdout.write("\r Ranking: {s}/{l}".format(s=i,l=len(doc_list)))
        sys.stdout.flush()
        rankings.append(rank)
    return rankings

if redo_w2v:
    w2v_ranks = rank_byfeature_w2v(test_docs,target_feature = 'directors',source_feature='principalCast')
else:
    with open(os.path.join(datadir,'w2v_test_ranks.p'),'r') as f:
        w2v_ranks=cPickle.load(f)

#w2v_ranks_old = rank_byfeature_w2v(test_docs[0:10],target_feature = 'directors',source_feature='principalCast')
#print w2v_ranks[0:10]
#print w2v_ranks_old





def rank_byfeature_f2v(df,target_feature,source_feature,batch_size=100):
    '''
    Rank a set of features associated with a given vetor.
    feature is the var name
    feature_weight_name is the name of the embedding layer the embeddings come from.
    '''
    target_tokens = f2v.loc[target_feature].index.tolist()
    target_vecs = np.array(f2v.loc[target_feature])
    source_vecs = f2v.loc[source_feature]
    #target_vecs = np.array(f2v[f2v['feature']==target_feature,embedding_cols])
    print target_vecs.shape
    rankings=[]
    def sum_vec(row):
        return  np.sum([source_vecs.loc[w] for w in row[source_feature] if w in source_vecs.index ],axis=0)[:,np.newaxis]
    index=0
    for batch in [df.iloc[pos:pos + batch_size] for pos in xrange(0, len(df), batch_size)]:
        batch_vecs = np.concatenate([sum_vec(row) for i,row in batch.iterrows()],axis=1)
        #print batch_vecs.shape
        scores = cosine_similarity(batch_vecs.T,target_vecs)
        #print scores.shape
        tempranks = np.argsort(-scores,axis=1)
        #print tempranks.shape
        for i in range(len(batch)):
            row  = batch.iloc[i]
            targetid = row[target_feature][0]
            ranks = np.empty(len(target_tokens), int)
            ranks[tempranks[i,:]] = np.arange(len(target_tokens))
            rank=ranks[target_tokens.index(targetid)]
            rankings.append(rank)
        index+=batch_size
        sys.stdout.write("\r Ranking: {s}/{l}".format(s=index,l=len(df)))
        sys.stdout.flush()
    return rankings

f2v_ranks = rank_byfeature_f2v(testdf,target_feature='directors',source_feature='principalCast',batch_size=1000)




#descriptives
maxrank =  len(f2v.loc['directors'])

#hists
plt.hist(w2v_ranks,alpha=1.,label='W2V',bins=100,histtype='step')
plt.hist(f2v_ranks,alpha=1.,label='F2V',bins=100,histtype='step')
plt.xlim([0, maxrank])
plt.xlabel('Rankings')
plt.ylabel('Counts')
plt.legend()
plt.savefig(os.path.join(outputdir,'rankhist.pdf'))
plt.show()

#cdf
plt.hist(w2v_ranks,alpha=1.,label='W2V',bins=range(maxrank),cumulative=True,normed=1,histtype='step')
plt.hist(f2v_ranks,alpha=1.,label='F2V',bins=range(maxrank),cumulative=True,normed=1,histtype='step')
plt.xlim([0, maxrank])
plt.xlabel('Rankings')
plt.ylabel('CDF')
plt.legend(loc=2)
plt.savefig(os.path.join(outputdir,'rankcdf.pdf'))
plt.show()


#cdf mediumzoomed
plt.hist(w2v_ranks,alpha=1.,label='W2V',bins=range(maxrank),cumulative=True,normed=1,histtype='step')
plt.hist(f2v_ranks,alpha=1.,label='F2V',bins=range(maxrank),cumulative=True,normed=1,histtype='step')
plt.xlim([0, maxrank/10])
plt.ylim([0.,.6])
plt.xlabel('Rankings')
plt.ylabel('CDF')
plt.legend(loc=2)
plt.savefig(os.path.join(outputdir,'rankcdf_10pct.pdf'))
plt.show()

#cdf super zoom
plt.hist(w2v_ranks,alpha=1.,label='W2V',bins=range(maxrank),cumulative=True,normed=1,histtype='step')
plt.hist(f2v_ranks,alpha=1.,label='F2V',bins=range(maxrank),cumulative=True,normed=1,histtype='step')
plt.xlim([0, 50])
plt.ylim([0.,.2])
plt.xlabel('Rankings')
plt.ylabel('CDF')
plt.legend(loc=2)
plt.savefig(os.path.join(outputdir,'rankcdf_top50.pdf'))
plt.show()

np.min(w2v_ranks)
#summary statistics
def MPR(ranks):
    return np.mean(ranks)/maxrank

def MRR(ranks):
    return np.mean(1/(np.array(ranks,dtype='float')+1))

def MDNR(ranks):
    return np.median(ranks)
def Top1Precision(ranks):
    return np.mean([r==0 for r in ranks])
with open(os.path.join(outputdir,'summstats.txt'),'w') as f:
    for stat in [MPR, MRR, MDNR,Top1Precision]:
        print "*"*30
        print stat.__name__
        print "F2V:", stat(f2v_ranks)
        print "W2V:",  stat(w2v_ranks)
        f.write( "*"*30 + "\n")
        f.write( stat.__name__ + '\n')
        f.write( "F2V:{}\n".format( stat(f2v_ranks) ) )
        f.write( "W2V:{}\n".format( stat(w2v_ranks)    )   )


with open(os.path.join(datadir,'f2v_test_ranks.p'),'w') as f:
    cPickle.dump(f2v_ranks,f)


with open(os.path.join(datadir,'w2v_test_ranks.p'),'w') as f:
    cPickle.dump(w2v_ranks,f)
