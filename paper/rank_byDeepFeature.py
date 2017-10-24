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

#datadir = ''
outputdir= 'paper/output/alpha_75_25/'
#load both sets of vectors
print "Loading w2v/f2v embeddings..."
w2v = KeyedVectors.load_word2vec_format(os.path.join(datadir,'w2v_vectors.txt'), binary=False)



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

#calculate average rating by first director in testdf
testdf['firstDirector'] = testdf['directors'].map(lambda x: x[0])
testdf = testdf[testdf['mi_rating']==0] #remove missing rating

avgRatings = testdf[['averageRating','directors']]
avgRatings['directors'] = avgRatings['directors'].map(lambda x: x[0])
avgRatings = avgRatings.groupby('directors').agg({'averageRating':['count','mean']})
avgRatings.head()





alphas = [0,25,50,75,100]
testerrs = []
weighted_testerrs = []
for a in alphas:
    print "alpha:{}".format(a)
    f2vdir = os.path.join(datadir,'alpha_75_{}'.format(a))
    f2v = pd.read_csv(os.path.join(f2vdir,'imdb_movie_embeddings.tsv'),sep='\t')
    f2v = f2v.set_index(['feature','values'])
    f2v_directors = f2v.loc['directors']
    f2v_ratings = f2v.loc['averageRating']
    f2v_ratings = f2v_ratings[f2v_ratings.index != '(0.0, 1.0)']#remove mi_rating cat
    #get first rank
    scores = cosine_similarity(f2v_ratings,f2v_directors)
    bestRatings  = np.argmax(scores,axis=0)
    #get rating
    f2v_directors['matchedRating'] = f2v_ratings.index.values[bestRatings]
    avgRatings['PredictedRating'] = avgRatings.index.map(lambda x: eval(f2v_directors.loc[x,'matchedRating'])[0])
    testdf['PredictedRating'] = testdf['firstDirector'].map(lambda x: avgRatings.loc[x,'PredictedRating'])
    RMSE = np.sqrt( np.mean((avgRatings['averageRating']['mean'] - avgRatings['PredictedRating'])**2) )
    WRMSE= np.sqrt( np.mean( (testdf['PredictedRating'] - testdf['averageRating'])**2) )
    testerrs.append(RMSE)
    weighted_testerrs.append(WRMSE)
    print RMSE,WRMSE

#W2V as a comparison
directortokens = [w for w in w2v.vocab if w.startswith('directors_')]
w2v_directors = np.concatenate([np.expand_dims(w2v[i],axis=0) for i in directortokens],axis=0)
w2v_directors =  pd.DataFrame(w2v_directors,index=directortokens)

w2v_rating_tokens = [w for w in w2v.vocab if w.startswith('averageRating_') and w != 'averageRating_0.0']
w2v_ratings = np.concatenate([w2v[w].reshape((1,50)) for w in w2v_rating_tokens],axis=0)
w2v_ratings = pd.DataFrame(w2v_ratings,index=w2v_rating_tokens)
print w2v_ratings.shape,w2v_directors.shape
w2vscores = cosine_similarity(w2v_ratings,w2v_directors)
bestRatings  = np.argmax(w2vscores,axis=0)
#print bestRatings.shape
w2vscores[bestRatings[0],0], directortokens[0],w2v_ratings.index[bestRatings[0]]
w2v_directors['matchedRating'] = w2v_ratings.index.values[bestRatings]
w2v_directors['matchedRating'] = w2v_directors['matchedRating'].map(lambda x: float(x[len('averageRating_'):]))
print w2v_directors['matchedRating']
print w2v_directors.head()
w2v_directors.index = w2v_directors.index.map(lambda x: x[len('directors_'):])
avgRatings['W2VPredictedRating'] = avgRatings.index.map(lambda x: w2v_directors.loc[x,'matchedRating'])
testdf['W2VPredictedRating'] = testdf['firstDirector'].map(lambda x: avgRatings.loc[x,'W2VPredictedRating'])
W2V_RMSE = np.sqrt( np.mean((avgRatings['averageRating']['mean'] - avgRatings['W2VPredictedRating'])**2) )
W2V_WRMSE= np.sqrt( np.mean( (testdf.loc[testdf['mi_rating']==0,'W2VPredictedRating'] - testdf.loc[testdf['mi_rating']==0,'averageRating'])**2) )
print W2V_RMSE,W2V_WRMSE


unif_RMSE = np.sqrt(np.mean( avgRatings['averageRating']['mean'] - 5.)**2 + 100./12.)
unif_WRMSE = np.sqrt( np.mean( (testdf['averageRating'] - 5.)**2 + 100./12.) )
print unif_RMSE,unif_WRMSE


#plt.plot([a/100. for a in alphas],testerrs,label='Feat2Vec')
plt.plot([a/100. for a in alphas],weighted_testerrs,label='Feat2Vec',marker='o',lw=3)
plt.axhline(y=W2V_WRMSE,xmin=0.,xmax=1.,color='g',ls='--',label='Word2Vec',lw=3)
plt.axhline(y=unif_WRMSE,xmin=0.,xmax=1.,color='black',ls='--',label='Random Uniform',lw=3)


plt.xlabel(r'$\alpha_1$')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('paper/output/ratingsRankingRMSE.pdf')
plt.show()

print np.mean( avgRatings['averageRating']['mean'] - 5.)**2
#get rankings for every director

#
#
# for r in ratings:
#     print "*"*30
#     print "Rating:",r
#     print "*"*30
#     w2v_rating = w2v['averageRating_{}'.format(r)]
#     w2v_rating.shape = (1,50)
#
#     f2v_rating = np.array(f2v.loc['averageRating'].loc['({}, 0.0)'.format(r)])
#     f2v_rating.shape = (1,50)
#     print f2v_rating
#     w2v_scores = cosine_similarity(w2v_rating,director_w2v_vecs)[0]
#     f2v_scores = cosine_similarity(f2v_rating,np.array(director_f2v_vecs))[0]
#     #print pd.Series(f2v_scores).describe()
#     w2v_ranks = np.argsort(-w2v_scores)
#     f2v_ranks = np.argsort(-f2v_scores)
#     #print f2v_ranks
#
#     #F2V part
#     maxrank = 3
#     curRank=0
#     idx=-1
#     f2v_avg_ratings = []
#     while curRank < maxrank:
#         idx+=1
#         dir_idx = f2v_ranks[idx]
#         dname = director_f2v_vecs.index[dir_idx]
#         if director_f2v_vecs.index[dir_idx] not in avgRatings.index:
#             #print director_f2v_vecs.index[dir_idx], "not in index!"
#             continue
#         else:
#             print "Rank {}: ".format(curRank+1), "(Index {})".format(idx)
#             print "Score : ", f2v_scores[dir_idx]
#             print avgRatings.loc[ director_f2v_vecs.index[dir_idx] ]
#             curRank += 1
#             f2v_avg_ratings.append(avgRatings.loc[ director_f2v_vecs.index[dir_idx] ]['averageRating']['mean'])
#             print "-"*20
#
#     print "Overall Average F2V Rating:",np.mean(f2v_avg_ratings)
#     print "-"*30
#     # curRank=0
#     # idx=-1
#     # w2v_avg_ratings = []
#     # while curRank < maxrank:
#     #     idx+=1
#     #     dir_idx = w2v_ranks[idx]
#     #     dname = directortokens[dir_idx][len('directors_'):]
#     #     if dname not in avgRatings.index:
#     #         #print director_f2v_vecs.index[dir_idx], "not in index!"
#     #         continue
#     #     else:
#     #         print "Rank {}: ".format(curRank+1), "(Index {})".format(idx)
#     #         #print "Score : ", w2v_scores[dir_idx]
#     #         print avgRatings.loc[  dname ]
#     #         curRank += 1
#     #         w2v_avg_ratings.append(avgRatings.loc[dname]['averageRating']['mean'])
#     #         print "-"*20
#
#     print "Overall Average W2V Rating:",np.mean(w2v_avg_ratings)
#
#
# f2v_rating1 = np.array(f2v.loc['averageRating'].loc['({}, 0.0)'.format(1.0)])
# f2v_rating2 = np.array(f2v.loc['averageRating'].loc['({}, 0.0)'.format(9.0)])
# np.dot(f2v_rating1,f2v_rating2)/(np.linalg.norm(f2v_rating1)*np.linalg.norm(f2v_rating2))
