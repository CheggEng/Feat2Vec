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
import yaml
from sklearn.metrics.pairwise import cosine_similarity
constants = yaml.load(open("paper/constants.yaml", 'rU'))
embedding_dim=50
#load everything up
with open(os.path.join(constants['yelpdir'],'dataset/f2v_vocab_map.p'),'r') as f:
    vocab_map = cPickle.load(f)

with open(os.path.join(constants['yelpdir'],'dataset/yelp_test_data.p'),'r') as f:
    testdf = cPickle.load(f)

with open(os.path.join(constants['yelpdir'],'dataset/yelp_textfilter.p'),'r') as f:
    review_filter = cPickle.load(f)

#load trained model/embeddings
f2v_model = keras.models.load_model(os.path.join(constants['yelpdir'],'model/yelp_f2v_model.h5'))
f2v_model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())

w2v_vecs = pd.read_csv(os.path.join(constants['yelpdir'],'w2v/w2v_cbow_embeddings.tsv'),sep='\t',index_col=['feature','values'])
d2v_vecs = pd.read_csv(os.path.join(constants['yelpdir'],'w2v/d2v_dm_embeddings.tsv'),sep='\t',index_col=['feature','values'])



print f2v_model.summary()
import h5py
#f=h5py.File(os.path.join(constants['yelpdir'],'w2v/d2v_dm_model.h5'))
#print {k:np.array(f['prob']['prob_1'][k]) for k in f['prob']['prob_1'].keys()}
#f.close()
f2v = pd.read_csv(os.path.join(constants['yelpdir'],'model/yelp_embeddings.tsv'),sep='\t')
f2v = f2v.set_index(['feature','values'])



# CS = pd.DataFrame(cosine_similarity(f2v.loc['business_id'],f2v.loc['stars']))
# CS.columns = list(f2v.loc['stars'].index)
# CS.shape
# CS.rank(axis=1).describe()
# print CS

#pd.DataFrame(star_embeddings).describe()
embed_cols = ['dim_{}'.format(i) for i in range(1,embedding_dim+1)]
f2v_testvecs={}
w2v_testvecs={}
d2v_testvecs={}
for v in ['funny','stars','user_id','business_id']:
    print v
    vec = f2v.loc[v].to_dict('index')
    vec.keys()
    vec = dict([ (k,np.array([vec[k][c] for c in embed_cols])) for k in vec.keys()   ])
    vec = np.concatenate([vec[str(u)][:,np.newaxis].T for u in testdf[v]],axis=0)
    f2v_testvecs[v] = vec

w2v_cols = [str(i) for i in range(embedding_dim)]
for v in ['funny','stars','user_id','business_id']:
    print v
    vec = w2v_vecs.loc[v].to_dict('index')
    vec = dict([ (k,np.array([vec[k][c] for c in w2v_cols])) for k in vec.keys()   ])
    #vec = np.concatenate([vec[str(u)][:,np.newaxis].T for u in testdf[v]],axis=0)
    key_bank = set(vec.keys())
    vec = [vec[str(u)][:,np.newaxis].T if str(u) in key_bank else np.zeros((1,embedding_dim)) for u in testdf[v]]
    vec = np.concatenate(vec,axis=0)
    print (np.sum(np.abs(vec),axis=1)==0.).mean()
    w2v_testvecs[v] = vec

for v in ['funny','stars','user_id','business_id']:
    print v
    vec = d2v_vecs.loc[v].to_dict('index')
    vec = dict([ (k,np.array([vec[k][c] for c in w2v_cols])) for k in vec.keys()   ])
    #vec = np.concatenate([vec[str(u)][:,np.newaxis].T for u in testdf[v]],axis=0)
    key_bank = set(vec.keys())
    vec = [vec[str(u)][:,np.newaxis].T if str(u) in key_bank else np.zeros((1,embedding_dim)) for u in testdf[v]]
    vec = np.concatenate(vec,axis=0)
    print (np.sum(np.abs(vec),axis=1)==0.).mean()
    d2v_testvecs[v] = vec


#extract text model.
textModel = Model(inputs=f2v_model.get_layer('wordind_seq').input,outputs=f2v_model.get_layer('embedding_textseq').output)
print textModel.summary()
textModel.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())

def process_keras_text(df):
    text_max_len=250
    mat = keras.preprocessing.sequence.pad_sequences(df.textseq.values.tolist(),maxlen=text_max_len,value=0,padding='post',truncating='post')
    return mat

print "loading text embeddings"
textmat = process_keras_text(testdf)
text_embeddings = textModel.predict([textmat])
f2v_testvecs['text'] = text_embeddings





word_bank =  set(vocab_map['textseq'].keys())
w2v_word_vecs = w2v_vecs.loc['word'].to_dict('index')
w2v_word_vecs = {vocab_map['textseq'][k]:np.array([w2v_word_vecs[k][str(c)] for c in range(50) ])  for k in w2v_word_vecs.keys() if k in word_bank}
w2v_word_bank = set(w2v_word_vecs.keys())
w2v_text_embeddings=[]
w2v_text_count = []
for i,doc in enumerate(testdf['textseq'].values):
    if  i % 1000==0:
        sys.stdout.write("\r {s}/{l}".format(s=i,l=len(testdf)))
        sys.stdout.flush()

    count = np.sum([w in w2v_word_bank for w in doc])
    if count > 0:
        doc_vec = np.mean(np.array([w2v_word_vecs[w] for w in doc if w in w2v_word_bank ]),axis=0)
    else:
        doc_vec = np.zeros((50,))
    w2v_text_count.append(count)
    w2v_text_embeddings.append(doc_vec[np.newaxis,:])

w2v_text_embeddings = np.concatenate(w2v_text_embeddings,axis=0)
print (np.sum(np.abs(w2v_text_embeddings),axis=1)==0.).mean()
w2v_testvecs['text'] = w2v_text_embeddings




d2v_word_vecs = d2v_vecs.loc['word'].to_dict('index')
d2v_word_vecs = {vocab_map['textseq'][k]:np.array([d2v_word_vecs[k][str(c)] for c in range(50) ])  for k in d2v_word_vecs.keys() if k in word_bank}
d2v_word_bank = set(d2v_word_vecs.keys())
d2v_text_embeddings=[]
d2v_text_count = []
for i,doc in enumerate(testdf['textseq'].values):
    if  i % 1000==0:
        sys.stdout.write("\r {s}/{l}".format(s=i,l=len(testdf)))
        sys.stdout.flush()

    count = np.sum([w in d2v_word_bank for w in doc])
    if count > 0:
        doc_vec = np.mean(np.array([d2v_word_vecs[w] for w in doc if w in d2v_word_bank ]),axis=0)
    else:
        doc_vec = np.zeros((50,))
    d2v_text_count.append(count)
    d2v_text_embeddings.append(doc_vec[np.newaxis,:])
d2v_text_embeddings = np.concatenate(d2v_text_embeddings,axis=0)
print '\n',(np.sum(np.abs(d2v_text_embeddings),axis=1)==0.).mean()
d2v_testvecs['text'] = d2v_text_embeddings




#create a context vector for CBOW
context_vecs=[]
num_words = 4 # num of words to scrape into given window size of 4
for i,doc in enumerate(testdf['textseq'].values):
    if  i % 1000==0:
        sys.stdout.write("\r {s}/{l}".format(s=i,l=len(testdf)))
        sys.stdout.flush()
    #count = np.sum([w in w2v_word_bank for w in doc])
    filter_doc = [w for w in doc if w in w2v_word_bank ]
    if len(filter_doc) >=num_words:
        doc_vec = np.sum(np.array([w2v_word_vecs[w] for w in filter_doc[:num_words] if w in w2v_word_bank ]),axis=0)
    else:
        doc_vec = np.zeros((50,))
    context_vecs.append(doc_vec[np.newaxis,:])

context_vecs =  np.concatenate(context_vecs,axis=0)

context_vecs += w2v_testvecs['business_id'] + w2v_testvecs['funny'] + w2v_testvecs['user_id'] + w2v_testvecs['funny']
context_vecs/=8.
w2v_testvecs['context_mean'] =context_vecs


###attempt a cosine simlarity exercise by
def pred_by_feature(feat_embeddings,star_embeddings,batch_size=1000):
    predicted_ratings = []
    index=0
    for batch in [feat_embeddings[pos:(pos + batch_size),:] for pos in xrange(0, len(feat_embeddings), batch_size)]:
        scores = cosine_similarity(batch,star_embeddings)
        #print scores
        neighbor = np.argmax(scores,axis=1)
        neighbor = map(lambda x: float(star_embeddings.index[x]),neighbor)
        index+=batch_size
        sys.stdout.write("\r batch: {s}/{l}".format(s=index,l=len(feat_embeddings)))
        sys.stdout.flush()
        predicted_ratings.append(neighbor)
    predicted_ratings = np.concatenate(predicted_ratings,axis=0)
    return predicted_ratings

def pred_by_all(text_embeddings,funny_embeddings,biz_embeddings,user_embeddings,star_embeddings,batch_size=1000):
    predicted_ratings = []
    index=0
    for pos in xrange(0, len(text_embeddings), batch_size):
        batch = testdf.iloc[pos:(pos + batch_size),:]
        text_vecs = text_embeddings[pos:(pos + batch_size),:]
        biz_vecs = biz_embeddings[pos:(pos + batch_size),:]
        user_vecs = user_embeddings[pos:(pos + batch_size),:]
        funny_vecs = funny_embeddings[pos:(pos + batch_size),:]
        avg_vecs = (text_vecs + biz_vecs + funny_vecs + user_vecs) / 4
        #if np.isnan(biz_vecs).sum()>0:
        #    avg_vecs = (text_vecs + user_vecs+ funny_vecs) / 2
        #avg_vecs = biz_vecs
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

testdf['stars'] = testdf['stars'].map(lambda x: float(str(x))).astype(int)
testdf['stars'].value_counts()

#evalmodes=['funny','text','business_id','user_id','all']
evalmodes=['text']
evals={}
#models = ['f2v','d2v','w2v']
models = ['d2v']
for v in models:
    print '\n',v
    evals[v]={}
    if v=='d2v':
        star_embeddings = d2v_vecs.loc['stars']
        edict = d2v_testvecs
    elif v=='f2v':
        star_embeddings = f2v.loc['stars']
        edict = f2v_testvecs
    elif v=='w2v':
        star_embeddings = w2v_vecs.loc['stars']
        edict = w2v_testvecs
    for m in evalmodes:
        if m=='all' and v!='w2v':
            pred_ratings = pred_by_all(edict['text'],edict['funny'],edict['business_id'],edict['user_id'],star_embeddings)
        elif m=='all' and v=='w2v':
            pred_ratings = pred_by_feature(edict['context_mean'],star_embeddings)
        else:
            pred_ratings = pred_by_feature(edict[m],star_embeddings)
        error_rate = (pred_ratings!=testdf['stars']).mean()
        MSE = np.mean( (pred_ratings - testdf['stars'])**2)
        conmat = pd.crosstab(testdf['stars'],pred_ratings)
        evals[v][m] = {'error':error_rate,'mse':MSE,'confusion':conmat}


evaldf=pd.DataFrame(index=pd.MultiIndex.from_product([models,evalmodes ] ),columns=['error','mse'])

for s in ['error','mse']:
    print '*'*20,s,'*'*20
    for e in evalmodes:
        for m in models:
            sv= evals[m][e][s]
            evaldf[s].loc[m].loc[e] = sv
            print m,e,sv

evaldf.to_csv('paper/output/yelp/eval_stats.csv')

import matplotlib.pyplot as plt
import itertools
#export confusion matrix
#from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
normalize=True
for t in evalmodes:
    if t=='text':
        sub='Text Nearest Neighbor'
    elif t=='all':
        sub='Context Mean Nearest Neighbor'
    elif t=='business_id':
        sub='Business Nearest Neighbor'
    elif t=='user_id':
        sub='User Nearest Neighbor'
    elif t=='funny':
        sub='Funny Nearest Neighbor'
    for m in models:
        mat = np.copy(evals[m][t]['confusion'])
        mat
        if normalize:
            mat = mat.astype(float)/mat.sum(axis=1,keepdims=True)*100.
        plt.imshow(mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.suptitle("Confusion matrix for %s " % m)
        plt.title(sub)
        fmt = '.2f' if normalize else 'd'
        thresh = mat.max() *3/ 4.
        for i, j in itertools.product(range(mat.shape[0]), range(mat.shape[1])):
            plt.text(j, i, format(mat[i, j], fmt) + r'%',
                     horizontalalignment="center",
                     color="white" if mat[i, j] > thresh else "black")
        plt.xticks(range(5),range(1,6))
        plt.yticks(range(5),range(1,6))
        plt.ylabel('True Rating')
        plt.xlabel('Predicted Rating')
        #plt.tight_layout()
        plt.savefig('paper/output/yelp/confusionmat_%s_%s.pdf' % (m,t))
        plt.show()


#calc MSE
mse =0.
for s in range(1,6):
    mse+= .2*np.mean((testdf['stars']-s)**2)
accrate = 0.
for s in range(1,6):
    accrate+= .2*np.mean(testdf['stars']==s)
erate = 1.-accrate
accrate
#use empirical distro
ecdf= testdf['stars'].value_counts(normalize=True).sort_index()
empmse =0.
for s in range(1,6):
    empmse+= ecdf.loc[s]*np.mean((testdf['stars']-s)**2)
empaccrate = 0.
for s in range(1,6):
    empaccrate+= ecdf.loc[s]*np.mean(testdf['stars']==s)
emperate = 1.-empaccrate
#print empmse,emperate
#3.6520727302729012 0.7341351361054282
evals[]

mse
plotevals=['user_id','business_id','funny','text','all']
labels=['User','Business','Funny','Text','Context']
colors=['r','b','g','purple']
for s in ['error','mse']:
    for i,m in enumerate(models):
        metrics = [evals[m][e][s] for e in plotevals]
        plt.scatter(range(len(evalmodes)),metrics,s=12,color=colors[i])
        plt.plot(range(len(evalmodes)),metrics,label=m,color=colors[i])
    if s=='error':
        plt.plot(range(len(evalmodes)),[erate]*len(evalmodes),'--',color='black',label='Random Uniform')
        plt.plot(range(len(evalmodes)),[emperate]*len(evalmodes),'--',color='brown',label='Random Empirical')
        plt.ylabel('Error Rate')
    elif s=='mse':
        plt.plot(range(len(evalmodes)),[mse]*len(evalmodes),'--',color='black',label='Random Uniform')
        plt.plot(range(len(evalmodes)),[empmse]*len(evalmodes),'--',color='brown',label='Random Empirical')
        plt.ylabel('MSE')
    plt.legend()
    plt.xlabel('Source Embedding')
    plt.xticks(range(len(evalmodes)),labels)
    plt.savefig('paper/output/yelp/eval_%s.pdf' % s)
    plt.show()

star_embeddings = f2v.loc['stars']
print star_embeddings.shape
print star_embeddings
f2v_text_pred_ratings = pred_by_text(text_embeddings,star_embeddings)

f2v_text_error_rate = (f2v_text_pred_ratings!=testdf['stars']).mean()
f2v_text_MSE = np.mean( (f2v_text_pred_ratings - testdf['stars'])**2)

print "*"*40
print pd.crosstab(testdf['stars'],f2v_text_pred_ratings,normalize='index')
print f2v_text_MSE,f2v_text_error_rate


f2v_all_pred_ratings = pred_by_all(text_embeddings,biz_embeddings,user_embeddings,star_embeddings)
f2v_all_error_rate = (f2v_all_pred_ratings!=testdf['stars']).mean()
f2v_all_MSE = np.mean( (f2v_all_pred_ratings - testdf['stars'])**2)
print '\n',pd.crosstab(testdf['stars'],f2v_all_pred_ratings,normalize='index')
print f2v_all_MSE,f2v_all_error_rate


#################
#w2v_cbow
################
w2v_test_embeddings={}
for w in ['stars','user_id','business_id','funny']:
    embed= w2v.loc[w].to_dict('index')
    embed = dict([ (k,np.array([embed[k][c] for c in embed_cols])) for k in embed.keys()   ])
    embed = np.concatenate([embed[u][:,np.newaxis].T for u in testdf[w]],axis=0)
    w2v_test_embeddings[w] = embed



##################
#now try W2V SG
##################
testdf.head()

#get vectors from W2V

user_ids = testdf['user_id'].map(lambda x: vocab_map['user_id'][x])
w2v_user_embeddings = w2v_vecs.loc['user_id'].to_dict('index')
w2v_user_embeddings = {k:np.array([w2v_user_embeddings[k][str(c)] for c in range(50) ])  for k in w2v_user_embeddings.keys()}
w2v_user_bank = set(w2v_user_embeddings.keys())
w2v_user_embeddings = [w2v_user_embeddings[u][:,np.newaxis].T if u in w2v_user_bank else np.zeros((1,embedding_dim)) for u in user_ids.values]
w2v_user_embeddings = np.concatenate(w2v_user_embeddings,axis=0)
print np.mean(w2v_user_embeddings.sum(axis=1)==0.)

biz_ids = testdf['business_id'].map(lambda x: vocab_map['business_id'][x])
w2v_biz_embeddings = w2v_vecs.loc['business_id'].to_dict('index')
w2v_biz_embeddings = {k:np.array([w2v_biz_embeddings[k][str(c)] for c in range(50) ])  for k in w2v_biz_embeddings.keys()}
w2v_biz_bank = set(w2v_biz_embeddings.keys())
w2v_biz_embeddings = [w2v_biz_embeddings[u][:,np.newaxis].T if u in w2v_biz_bank else np.zeros((1,embedding_dim)) for u in biz_ids.values]
w2v_biz_embeddings = np.concatenate(w2v_biz_embeddings,axis=0)
print np.mean(w2v_biz_embeddings.sum(axis=1)==0.)


funny_vals = testdf['funny'].map(lambda x: vocab_map['funny'][x])
w2v_funny_embeddings = w2v_vecs.loc['funny'].to_dict('index')
w2v_funny_embeddings = {k:np.array([w2v_funny_embeddings[k][str(c)] for c in range(50) ])  for k in w2v_funny_embeddings.keys()}
w2v_funny_bank = set(w2v_funny_embeddings.keys())
w2v_funny_embeddings = [w2v_funny_embeddings[u][:,np.newaxis].T if u in w2v_funny_bank else np.zeros((1,embedding_dim)) for u in funny_vals.values]
w2v_funny_embeddings = np.concatenate(w2v_funny_embeddings,axis=0)


w2v_int_vecs = w2v_vecs.loc['word'].copy()
w2v_int_vecs = w2v_int_vecs[[  w in vocab_map['textseq'] for w in w2v_int_vecs.index]]
w2v_int_vecs.index = [ vocab_map['textseq'][w] for w in w2v_int_vecs.index]
w2v_int_vecs = w2v_int_vecs.to_dict('index')
w2v_int_vecs = {k:np.array([w2v_int_vecs[k][str(c)] for c in range(50) ])  for k in w2v_int_vecs.keys()}
w2v_int_bank = set(w2v_int_vecs.keys())


w2v_text_embeddings=[]
w2v_text_count = []
for i,doc in enumerate(testdf['textseq'].values):
    if  i % 1000==0:
        sys.stdout.write("\r {s}/{l}".format(s=i,l=len(testdf)))
        sys.stdout.flush()

    count = np.sum([w in w2v_int_bank for w in doc])
    if count > 0:
        doc_vec = np.mean(np.array([w2v_int_vecs[w] for w in doc if w in w2v_int_bank ]),axis=0)
    else:
        doc_vec = np.zeros((50,))
    w2v_text_count.append(count)
    w2v_text_embeddings.append(doc_vec[np.newaxis,:])

w2v_text_embeddings = np.concatenate(w2v_text_embeddings,axis=0)

w2v_star_embeddings = w2v_vecs.loc['stars'].copy()
reverse_dict = {j:i for i,j in vocab_map['stars'].iteritems()}
w2v_star_embeddings.index = [reverse_dict[i] for i in w2v_star_embeddings.index]

w2v_text_pred_ratings = pred_by_text(w2v_text_embeddings,w2v_star_embeddings)
w2v_text_MSE = np.mean( (w2v_text_pred_ratings - testdf['stars'])**2)
print "*"*40
print pd.crosstab(testdf['stars'],w2v_text_pred_ratings,normalize='index')
w2v_text_error_rate = (w2v_text_pred_ratings!=testdf['stars']).mean()
print w2v_text_MSE,w2v_text_error_rate


w2v_all_pred_ratings = pred_by_all(w2v_text_embeddings,w2v_biz_embeddings,w2v_user_embeddings,w2v_star_embeddings)
w2v_all_MSE = np.mean( (w2v_all_pred_ratings - testdf['stars'])**2)
w2v_all_error_rate = (w2v_all_pred_ratings!=testdf['stars']).mean()
print '\n',pd.crosstab(testdf['stars'],w2v_all_pred_ratings,normalize='index')
print w2v_all_MSE,w2v_all_error_rate

#create a context vector
context_vecs=[]
num_words = 4 # num of words to scrape into given window size of 4
for i,doc in enumerate(testdf['textseq'].values):
    if  i % 1000==0:
        sys.stdout.write("\r {s}/{l}".format(s=i,l=len(testdf)))
        sys.stdout.flush()
    count = np.sum([w in w2v_int_bank for w in doc])
    filter_doc = [w for w in doc if w in w2v_int_bank ]
    if len(filter_doc) >=num_words:
        doc_vec = np.sum(np.array([w2v_int_vecs[w] for w in filter_doc[:num_words] if w in w2v_int_bank ]),axis=0)
    else:
        doc_vec = np.zeros((50,))
    context_vecs.append(doc_vec[np.newaxis,:])

context_vecs =  np.concatenate(context_vecs,axis=0)
context_vecs += w2v_funny_embeddings + w2v_biz_embeddings + w2v_user_embeddings
context_vecs/=7.
w2v_context_pred_ratings = pred_by_text(context_vecs,w2v_star_embeddings)
w2v_context_MSE = np.mean( (w2v_context_pred_ratings - testdf['stars'])**2)
w2v_context_error_rate = (w2v_context_pred_ratings!=testdf['stars']).mean()
print '\n',pd.crosstab(testdf['stars'],w2v_context_pred_ratings,normalize='index')
print w2v_context_MSE,w2v_context_error_rate



#uniform MSE?
unif_RMSE = np.mean( (testdf['stars'] - 3)**2 + 25./12.)

unif_RMSE
