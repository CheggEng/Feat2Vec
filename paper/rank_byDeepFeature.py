#rank_byDeepFeature.py
#rank some movies or directors by
import numpy  as np
import pandas as pd
import sys,os
import cPickle
import matplotlib.pyplot as plt
from gensim.models.word2vec import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import feat2vec
datadir = '/home/luis/Data/IMDB/'
datadir = '/media/luis/hdd3/Data/IMDB/'
f2vdir = os.path.join(datadir,'alpha_75_25')
#datadir = ''
outputdir= 'paper/output/alpha_75_25/'
#load both sets of vectors
print "Loading w2v/f2v embeddings..."
w2v = KeyedVectors.load_word2vec_format(os.path.join(datadir,'w2v_vectors.txt'), binary=False)

f2v = pd.read_csv(os.path.join(f2vdir,'imdb_movie_embeddings.tsv'),sep='\t')
f2v = f2v.set_index(['feature','values'])


#print f2v.loc['titleSeq'].loc['maya']
print "Loading df..."
with open(os.path.join(datadir,'imdb_test_movie_data.p'),'r') as f:
    testdf = cPickle.load(f)


print testdf.head()

#create document list vsn of test dataframe
print "cleaning df..."
tempdf = testdf.copy()
exclude_tokens = set(['isAdult_0','mi_rating_0'])
sentence_fcn = lambda x: ' '.join([w for w in x if w not in exclude_tokens])
seqlengths = {'genres':3,'writers':5,'directors':5,'principalCast':10,'titleSeq':5}
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

#average rating by first director in testdf

avgRatings = testdf[['averageRating','directors','mi_rating']]
avgRatings['directors'] = avgRatings['directors'].map(lambda x: x[0])
avgRatings = avgRatings[avgRatings['mi_rating']==0]
avgRatings = avgRatings.groupby('directors').agg({'averageRating':['count','mean']})

avgRatings.head()


print f2v.loc['averageRating'].sort_index().index
print pd.Series([w for w in w2v.vocab if w.startswith('averageRating_')]).sort_values()


directortokens = [w for w in w2v.vocab if w.startswith('directors_')]
director_w2v_vecs = np.concatenate([np.expand_dims(w2v[i],axis=0) for i in directortokens],axis=0)
director_f2v_vecs = f2v.loc['directors']
print len(director_w2v_vecs),len(director_f2v_vecs)
ratings = [3.0,7.0,9.0]



for r in ratings:
    print "*"*30
    print "Rating:",r
    print "*"*30
    w2v_rating = w2v['averageRating_{}'.format(r)]
    w2v_rating.shape = (1,50)

    f2v_rating = np.array(f2v.loc['averageRating'].loc['({}, 0.0)'.format(r)])
    f2v_rating.shape = (1,50)
    print f2v_rating
    w2v_scores = cosine_similarity(w2v_rating,director_w2v_vecs)[0]
    f2v_scores = cosine_similarity(f2v_rating,np.array(director_f2v_vecs))[0]
    #print pd.Series(f2v_scores).describe()
    w2v_ranks = np.argsort(-w2v_scores)
    f2v_ranks = np.argsort(-f2v_scores)
    #print f2v_ranks

    #F2V part
    maxrank = 3
    curRank=0
    idx=-1
    f2v_avg_ratings = []
    while curRank < maxrank:
        idx+=1
        dir_idx = f2v_ranks[idx]
        dname = director_f2v_vecs.index[dir_idx]
        if director_f2v_vecs.index[dir_idx] not in avgRatings.index:
            #print director_f2v_vecs.index[dir_idx], "not in index!"
            continue
        else:
            print "Rank {}: ".format(curRank+1), "(Index {})".format(idx)
            print "Score : ", f2v_scores[dir_idx]
            print avgRatings.loc[ director_f2v_vecs.index[dir_idx] ]
            curRank += 1
            f2v_avg_ratings.append(avgRatings.loc[ director_f2v_vecs.index[dir_idx] ]['averageRating']['mean'])
            print "-"*20

    print "Overall Average F2V Rating:",np.mean(f2v_avg_ratings)
    print "-"*30
    # curRank=0
    # idx=-1
    # w2v_avg_ratings = []
    # while curRank < maxrank:
    #     idx+=1
    #     dir_idx = w2v_ranks[idx]
    #     dname = directortokens[dir_idx][len('directors_'):]
    #     if dname not in avgRatings.index:
    #         #print director_f2v_vecs.index[dir_idx], "not in index!"
    #         continue
    #     else:
    #         print "Rank {}: ".format(curRank+1), "(Index {})".format(idx)
    #         #print "Score : ", w2v_scores[dir_idx]
    #         print avgRatings.loc[  dname ]
    #         curRank += 1
    #         w2v_avg_ratings.append(avgRatings.loc[dname]['averageRating']['mean'])
    #         print "-"*20

    print "Overall Average W2V Rating:",np.mean(w2v_avg_ratings)


f2v_rating1 = np.array(f2v.loc['averageRating'].loc['({}, 0.0)'.format(1.0)])
f2v_rating2 = np.array(f2v.loc['averageRating'].loc['({}, 0.0)'.format(9.0)])
np.dot(f2v_rating1,f2v_rating2)/(np.linalg.norm(f2v_rating1)*np.linalg.norm(f2v_rating2))
