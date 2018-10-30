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
sys.path.append('feat2vec/')
import feat2vec
#datadir = '/home/luis/Data/IMDB'
datadir = '/media/luis/hdd3/Data/IMDB/'
outputdir= 'paper/output/'
#load both sets of vectors
print "Loading w2v embeddings..."
w2v = KeyedVectors.load_word2vec_format(os.path.join(datadir,'w2v_vectors.txt'), binary=False)


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




#F2V evaluation by alpha1
alphas = [0,25,50,75,100]
#alphas = [75]
testerrs = []
for a in alphas:
    print "alpha:{}".format(a)
    f2vdir = os.path.join(datadir,'alpha_75_{}'.format(a))
    #f2vdir=datadir
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
    RMSE= np.sqrt( np.mean( (testdf['PredictedRating'] - testdf['averageRating'])**2) )
    testerrs.append(RMSE)
    print RMSE, #WRMSE

#W2V as a comparison
directortokens = [w for w in w2v.vocab if w.startswith('directors_')]
w2v_directors = np.concatenate([np.expand_dims(w2v[i],axis=0) for i in directortokens],axis=0)
w2v_directors =  pd.DataFrame(w2v_directors,index=directortokens)

w2v_rating_tokens = [w for w in w2v.vocab if w.startswith('averageRating_') and w != 'averageRating_0.0']
w2v_ratings = np.concatenate([w2v[w].reshape((1,50)) for w in w2v_rating_tokens],axis=0)
w2v_ratings = pd.DataFrame(w2v_ratings,index=w2v_rating_tokens)
w2vscores = cosine_similarity(w2v_ratings,w2v_directors)
bestRatings  = np.argmax(w2vscores,axis=0)
w2vscores[bestRatings[0],0], directortokens[0],w2v_ratings.index[bestRatings[0]]
w2v_directors['matchedRating'] = w2v_ratings.index.values[bestRatings]
w2v_directors['matchedRating'] = w2v_directors['matchedRating'].map(lambda x: float(x[len('averageRating_'):]))

w2v_directors.index = w2v_directors.index.map(lambda x: x[len('directors_'):])
avgRatings['W2VPredictedRating'] = avgRatings.index.map(lambda x: w2v_directors.loc[x,'matchedRating'])
testdf['W2VPredictedRating'] = testdf['firstDirector'].map(lambda x: avgRatings.loc[x,'W2VPredictedRating'])
W2V_RMSE= np.sqrt( np.mean( (testdf.loc[testdf['mi_rating']==0,'W2VPredictedRating'] - testdf.loc[testdf['mi_rating']==0,'averageRating'])**2) )
print W2V_RMSE

#create uniform RMSE
unif_RMSE = np.sqrt( np.mean( (testdf['averageRating'] - 5.)**2 + 100./12.) )
print unif_RMSE,unif_RMSE


#plt.plot([a/100. for a in alphas],testerrs,label='Feat2Vec')
plt.plot([a/100. for a in alphas],testerrs,label='Feat2Vec',marker='o',lw=3)
plt.axhline(y=W2V_RMSE,xmin=0.,xmax=1.,color='g',ls='--',label='CBOW',lw=3)
plt.axhline(y=unif_RMSE,xmin=0.,xmax=1.,color='black',ls='--',label='Random Uniform',lw=3)


plt.xlabel(r'$\alpha_1$')
plt.ylabel('RMSE')
plt.legend()
plt.savefig('paper/output/ratingsRankingRMSE.pdf')
plt.show()
