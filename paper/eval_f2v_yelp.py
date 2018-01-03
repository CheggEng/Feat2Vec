#eval_f2v_yelp.py
#using embeddings, try to predict ratings of test dataset; use MSE as metric
import keras
import cPickle
import os
import numpy  as np
import pandas as pd
import keras
from keras.models import Model
import sys
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity

datadir = '/home/luis/Data/Yelp/dataset/'
embedding_dim=50
#load everything up
with open(os.path.join(datadir,'f2v_vocab_map.p'),'r') as f:
    vocab_map = cPickle.load(f)

with open(os.path.join(datadir,'yelp_test_data.p'),'r') as f:
    testdf = cPickle.load(f)

with open(os.path.join(datadir,'yelp_textfilter.p'),'r') as f:
    review_filter = cPickle.load(f)

#load trained model/embeddings
f2v_model = keras.models.load_model(os.path.join(datadir,'yelp_f2v_model.h5'))
f2v_model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())





print f2v_model.summary()

f2v = pd.read_csv(os.path.join(datadir,'yelp_embeddings.tsv'),sep='\t')
f2v = f2v.set_index(['feature','values'])

#extract text model.
CS = pd.DataFrame(cosine_similarity(f2v.loc['business_id'],f2v.loc['stars']))
CS.columns = list(f2v.loc['stars'].index)
CS.shape
CS.rank(axis=1).describe()
print CS

textModel = Model(inputs=f2v_model.get_layer('wordind_seq').input,outputs=f2v_model.get_layer('embedding_textseq').output)
print textModel.summary()
textModel.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())

#starModel = Model(inputs=f2v_model.get_layer('input_rating').input,outputs=f2v_model.get_layer('embedding_stars').output)
#print starModel.summary()
#starModel.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())

#star_embeddings = starModel.predict([(np.arange(1.,5.01,.01) -1.)/4.])
#star_embeddings = pd.DataFrame(star_embeddings, index =np.arange(1.,5.01,.01) )
#print star_embeddings

testdf['textseq'] = review_filter.texts_to_sequences(testdf.text.values.tolist())
testdf['textseq'] = testdf['textseq'].map(lambda x: tuple(x))

def process_keras_text(df):
    text_max_len=250
    mat = keras.preprocessing.sequence.pad_sequences(df.textseq.values.tolist(),maxlen=text_max_len,value=0,padding='post',truncating='post')
    return mat

print "loading text embeddings"
textmat = process_keras_text(testdf)
textmat.shape
text_embeddings = textModel.predict([textmat])
text_embeddings.shape

pd.DataFrame(text_embeddings).describe()
#pd.DataFrame(star_embeddings).describe()


embed_cols = ['dim_{}'.format(i) for i in range(1,embedding_dim+1)]
print 'loading user embeddings'
user_embeddings = f2v.loc['user_id'].to_dict('index')
user_embeddings = dict([ (k,np.array([user_embeddings[k][c] for c in embed_cols])) for k in user_embeddings.keys()   ])
user_embeddings = np.concatenate([user_embeddings[u][:,np.newaxis].T for u in testdf['user_id']],axis=0)

print 'loading business embeddings'
biz_embeddings = f2v.loc['business_id'].to_dict('index')
biz_embeddings = dict([ (k,np.array([biz_embeddings[k][c] for c in embed_cols])) for k in biz_embeddings.keys()   ])
biz_embeddings = np.concatenate([biz_embeddings[u][:,np.newaxis].T for u in testdf['business_id']],axis=0)

star_embeddings = f2v.loc['stars']
print star_embeddings.shape
print star_embeddings
text_embeddings.shape
len(text_embeddings)
###attempt a cosine simlarity exercise by
def pred_by_text_f2v(batch_size=1000):
    predicted_ratings = []
    index=0
    for batch in [text_embeddings[pos:(pos + batch_size),:] for pos in xrange(0, len(text_embeddings), batch_size)]:
        scores = cosine_similarity(batch,star_embeddings)
        #print scores
        neighbor = np.argmax(scores,axis=1)
        neighbor = map(lambda x: float(star_embeddings.index[x]),neighbor)
        index+=batch_size
        sys.stdout.write("\r batch: {s}/{l}".format(s=index,l=len(text_embeddings)))
        sys.stdout.flush()
        predicted_ratings.append(neighbor)
    predicted_ratings = np.concatenate(predicted_ratings,axis=0)
    return predicted_ratings

def pred_by_all_f2v(batch_size=1000):
    predicted_ratings = []
    index=0
    for pos in xrange(0, len(text_embeddings), batch_size):
        batch = testdf.iloc[pos:(pos + batch_size),:]
        text_vecs = text_embeddings[pos:(pos + batch_size),:]
        biz_vecs = biz_embeddings[pos:(pos + batch_size),:]
        user_vecs = user_embeddings[pos:(pos + batch_size),:]
        avg_vecs = (text_vecs + biz_vecs + user_vecs) / 3
        avg_vecs = biz_vecs
        scores = cosine_similarity(avg_vecs,star_embeddings)
        #print scores
        neighbor = np.argmax(scores,axis=1)
        neighbor = map(lambda x: float(star_embeddings.index[x]),neighbor)
        index+=batch_size
        sys.stdout.write("\r batch: {s}/{l}".format(s=index,l=len(text_embeddings)))
        sys.stdout.flush()
        predicted_ratings.append(neighbor)
    predicted_ratings = np.concatenate(predicted_ratings,axis=0)
    return predicted_ratings

pred_ratings = pred_by_text_f2v()
testdf['stars'] = testdf['stars'].map(lambda x: float(str(x))).astype(int)
testdf['stars'].value_counts()
MSE = np.mean( (pred_ratings - testdf['stars'])**2)
print "*"*40
print pd.Series(pred_ratings).value_counts()
print MSE


pred_ratings = pred_by_all_f2v()

MSE = np.mean( (pred_ratings - testdf['stars'])**2)
print "\n",pd.Series(pred_ratings).value_counts()
print MSE
