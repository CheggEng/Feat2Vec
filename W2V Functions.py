
# coding: utf-8

# # Store W2V Functions in one place

# # Part 1: Fit W2V Model



import pandas as pd
import numpy as np
import pickle
import sys
import re
import os
import itertools
import subprocess
import gensim
from subprocess import Popen, PIPE, STDOUT
from gensim.models.wrappers import fasttext
from gensim.models import KeyedVectors
from hyperdash import monitor_cell
import implicitsampler
datadir=''



window=71 #max number of columns/words per row
embed_dim=30 #number of dimensions
output_dir= '' #dir to output model data
fasttext_dir  = 'fastText/fasttext' #where fasttext is in the AWS machine
fasttext_model = output_dir + 'model' #where we save the model output
#name of document lists we save
word2vec_train= '{}train_w2v_docs.txt'.format(datadir)
word2vec_dev= '{}dev_w2v_docs.txt'.format(datadir)
word2vec_test= '{}test_w2v_docs.txt'.format(datadir)



#load data
train=None
dev=None
test=None
exclude_tokens = set([v + '_0' for v in ['real_tokens_']])


doclist = [word2vec_train,word2vec_dev,word2vec_test]
if generate_wv_data:
    for df,i in enumerate([train,test,dev]):
        #export the train DF to a datalist
        batch_size = 1000000
        sentence_fcn = lambda x: \' \'.join([w for w in x if w not in exclude_tokens])
        with open(doclist[i],\'w\') as f:
            for pos in range(0,len(train),batch_size):
                print pos
                df_chunk = train.iloc[pos:pos + batch_size].copy()
                for c in train.columns:
                    tag_fcn = lambda x: c  + "_" + str(x)
                    df_chunk[c] = df_chunk[c].map(tag_fcn)
                if pos==0:
                    print df_chunk.head()
                df_chunk = df_chunk.values.tolist()
                df_chunk = '\n'.join([sentence_fcn(r) for r in df_chunk])
                f.write(df_chunk)




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

def run_bash_cmd(cmd):
    '''
    run a terminal command and feed the output to the terminal in real time
    '''
    cmd_ls = cmd.split()
    p = subprocess.Popen(cmd_ls, stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
    for line in iter(p.stdout.readline, ""):
        sys.stdout.write('\r'+line.replace('\n',''))
        sys.stdout.flush()



#read in development set doc list for evaluation in gensim
dev_sentences=[]
line_batch_size = 100000
with open(word2vec_dev,'r') as f:
    while True:
        sentence_iter = f.readlines(line_batch_size)
        if not sentence_iter:
                break
        sentence_iter = [s.split() for s in sentence_iter]]
    dev_sentences+=sentence_iter
    sys.stdout.write("\\r Read {s}M Sentences".format(s=len(dev_sentences)/1000000.))
    sys.stdout.flush()
    print "\\n"')



#define fasttext call
cmd = '''
{fastdir} cbow -input {datadir}train_w2v_docs.txt
-output {model_new}
-maxn 0
-minCount 0
-dim 30
-ws {window}
'''.format(fastdir=fasttext_dir,window=window,
           datadir=datadir,
           model_new=fasttext_model) #the command; note fasttext does alpha=.5 by default

cmd = cmd +  '-epoch {}'



# now train the model by doing line search over optimal # of epochs
num_epochs = 10
num_dev_obs = 100000
np.random.seed(9)
dev_subsample  = [dev_sentences[i] for i in np.random.randint(num_dev_obs,size=num_dev_obs)]
dev_losses=[]
dev_losses.append(0)
log = output_dir  + 'epoch_losses.txt'
with open(log,'w') as f:
    f.write('W2V Performance By Epochs\nEpoch 0 Similarity: 0 \n\')
    for e in range(1,num_epochs+1):
        print "Epoch Count: {}".format(e)
        run_bash_cmd(cmd.format(e))
        print "\\nNow Loading Computed Vectors..."
        ft_model = KeyedVectors.load_word2vec_format(fasttext_model+'.vec')
        print "Now Computing Loss..."
        loss = dev_loss(dev_subsample,ft_model)
        dev_losses.append(loss)
        with open(log,'a') as f:
        f.write('Epoch {e} Similarity: {l} \n'.format(e=e,l=loss))
        if loss > dev_losses[len(dev_losses)-2]:
            print "Exiting at Epoch {} due to decrease in Validation Similarity!!!".format(e)
            break
        print "\\n Avg Similarity: {}".format(loss)
    print "-------------------------"

    print dev_losses

opt_epoch = dev_losses.index(np.max(dev_losses))
#opt_epoch = 10
print "starting optimal {opt_epoch} ...".format(opt_epoch=opt_epoch)
run_bash_cmd(cmd.format(opt_epoch))')


# # Part 2: Compare cosine similarity of embeddings



from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from subprocess import Popen, PIPE, STDOUT
from gensim.models.wrappers import fasttext
from gensim.models import KeyedVectors
from hyperdash import monitor_cell
from sklearn.metrics.pairwise import cosine_similarity
w2v_model = KeyedVectors.load_word2vec_format('model.vec'.format(experiment_name_w2v))




def MRR(ranks):
    '''
    return the mean reciprocal ranks of a list of ranks
    '''
    return np.mean(1/(np.array(ranks,dtype='float')+1))
def avg_vector(doc):
    '''
    calculates the average of all words in a document fed to the function
    often used for calculating an average "context" vector
    '''
    #print doc
    vectors = np.array([w2v_model[w] for w in doc if w in w2v_model.vocab])
    #print vectors.shape
    return np.mean(vectors,axis=0)

def rank_isbns(doc_list):
    '''
    rank the actual isbn associated with each document across the set of isbn_tokens considered
    '''
    isbn_list = list(isbn_tokens)
    isbn_vecs = np.concatenate([np.expand_dims(w2v_model[i],axis=0) for i in isbn_list],axis=0)
    print isbn_vecs.shape
    rankings=[]
    for i,doc in enumerate(doc_list):
        isbn = doc[0]
        vec = np.expand_dims(avg_vector(doc[1:(len(doc)-1)]),axis=0)
        scores = cosine_similarity(vec,isbn_vecs)[0]
        temp = np.argsort(-scores)
        ranks = np.empty(len(scores), int)
        ranks[temp] = np.arange(len(scores))
        rank=ranks[isbn_list.index(isbn)]
        sys.stdout.write("\r Ranking: {s}/{l}".format(s=i,l=len(doc_list)))
        sys.stdout.flush()
        rankings.append(rank)
    return rankings

def rank_isbns_byfeature_w2v(doc_list,feature):
    '''
    Rank ISBNs associated with a given feature.
    feature should be the string prefix for all tokens of a given feature type
    '''
    isbn_list = list(isbn_tokens)
    isbn_vecs = np.concatenate([np.expand_dims(w2v_model[i],axis=0) for i in isbn_list],axis=0)
    print isbn_vecs.shape
    rankings=[]
    for i,doc in enumerate(doc_list):
        isbn = doc[0]
        featureid = (w for w in doc if w.startswith(feature)).next()
        if featureid not in w2v_model.vocab:
            continue
        y = np.zeros(len(isbn_list))
        y[isbn_list.index(isbn)] = 1
        featurevec = np.expand_dims(w2v_model[featureid],axis=0)
        scores = cosine_similarity(featurevec,isbn_vecs)[0]
        temp = np.argsort(-scores)
        ranks = np.empty(len(scores), int)
        ranks[temp] = np.arange(len(scores))
        rank=ranks[isbn_list.index(isbn)]
        sys.stdout.write("\r Ranking: {s}/{l}".format(s=i,l=len(doc_list)))
        sys.stdout.flush()
        rankings.append(rank)
    return rankings


def rank_isbns_byfeature_fm(df,feature,feature_weight_name):
    '''
    rank ISBNS associated with a given observation depending on dot product similarity with another feature
    feature is the var name
    feature_weight_name is the name of the embedding layer the embeddings come from.
    '''
    isbn_vecs = fm.get_layer('embedding_isbn').get_weights()[0]
    feature_vecs = fm.get_layer('embedding_{}'.format(feature_weight_name)).get_weights()[0]
    train_isbns = list(isbn_set)
    train_isbns.sort()
    isbn_train_vecs = isbn_vecs[train_isbns,:]
    rankings=[]
    for i,r in enumerate(df.iterrows()):
        row = r[1]
        if feature +'_' + str(row[feature]) not in w2v_model.vocab:
            continue
        feature_vec = np.expand_dims(feature_vecs[row[feature],:],axis=0)
        y = np.zeros(isbn_vecs.shape[0])
        y[row['equivalent_isbn13']] = 1
        #scores = cosine_similarity(user_vec,isbn_train_vecs)[0]
        scores= np.dot(feature_vec,isbn_train_vecs.T).flatten()
        temp = np.argsort(-scores)
        ranks = np.empty(len(scores), int)
        ranks[temp] = np.arange(len(scores))
        rank=ranks[y[train_isbns]==1][0]
        sys.stdout.write("\r Ranking: {s}/{l}".format(s=i,l=len(df)))
        sys.stdout.flush()
        rankings.append(rank)
    return rankings




w2v_user_rankings = rank_isbns_byfeature_w2v(test_docs_4rank,'user_id')
fm_user_rankings = rank_isbns_byfeature_fm(test_4rank,'user_id','user')
plt.figure(figsize=(10,10))
plt.hist(w2v_user_rankings,cumulative=False,label='W2V',normed=1,bins=100,histtype='step')
plt.hist(fm_user_rankings,cumulative=False,label='FM',normed=1,bins=100,histtype='step')
plt.legend(loc=1)
plt.savefig('../model/results/{}/hist_users_comp.pdf'.format(experiment_name))
plt.show()")


# # Part 3: kNN of embeddings


#Sanity check; find nearby ISBNs according to the MFDFM model
def knn_isbns_fm(k,isbn_code):
    isbn_vecs = fm.get_layer('embedding_isbn').get_weights()[0]
    isbn_vec = np.expand_dims(isbn_vecs[isbn_code,:],axis=0)
    scores = cosine_similarity(isbn_vec,isbn_vecs)[0]
    cur_isbn = scores.tolist().index(np.max(scores))
    print cur_isbn
    temp = np.argsort(-scores)
    ranks = np.empty(len(scores), int)
    ranks[temp] = np.arange(len(scores))
    ranks=ranks.tolist()
    print np.max(scores)
    #if title_dict[isbn_code] == '':
    #    print "you should really send an ISBN with a title..."
    #    return None
    print "Most {k} similar books to {title}".format(k=k,title=title_dict[isbn_code])
    cur_rank = 0
    num_ranks_shown = 0
    print len(ranks)
    while num_ranks_shown <= k:
        rank_ind = ranks.index(cur_rank)
        print cur_rank, isbn_code_dict[rank_ind]
        if title_dict[rank_ind]!='':
            print cur_rank,title_dict[rank_ind]
            num_ranks_shown+=1
        cur_rank+=1


def knn_isbns_w2v(k,isbn_code):
    isbn_list = list(isbn_tokens)
    isbn_vecs = np.concatenate([np.expand_dims(w2v_model[i],axis=0) for i in isbn_list],axis=0)
    isbn_vec = np.expand_dims(isbn_vecs[isbn_list.index('equivalent_isbn13_'+str(isbn_code)),:],axis=0)
    scores = cosine_similarity(isbn_vec,isbn_vecs)[0]
    sorted_scores = np.sort(scores)
    print scores
    print "---"
    print sorted_scores
    #print scores.shape
    cur_isbn = scores.tolist().index(np.max(scores))
    print cur_isbn
    temp = np.argsort(-scores)
    ranks = np.empty(len(scores), int)
    ranks[temp] = np.arange(len(scores))
    ranks=ranks.tolist()
    print "----"
    #print ranks
    print ranks[cur_isbn]
    #print ranks
    #if title_dict[isbn_code] == '':
    #    print "you should really send an ISBN with a title..."
    #    return None
    print "Most {k} similar books to {title}".format(k=k,title=title_dict[isbn_code])
    cur_rank = 0
    num_ranks_shown = 0
    while num_ranks_shown <= k:
        ranked_isbn_code = int(isbn_list[ranks.index(cur_rank)][len('equivalent_isbn13_'):])
        print cur_rank, isbn_code_dict[ranked_isbn_code]
        if title_dict[ranked_isbn_code]!='':
            print cur_rank,title_dict[ranked_isbn_code]
            num_ranks_shown+=1
        cur_rank+=1

#create mapper of ISBNs to their titles
isbn_codes = pd.concat([train['equivalent_isbn13'].cat.codes,
                    dev['equivalent_isbn13'].cat.codes,
                    test['equivalent_isbn13'].cat.codes],axis=0,ignore_index=True)
isbn_titles = pd.concat([train['book_title'],
                    dev['book_title'],
                    test['book_title']] ,axis=0,ignore_index=True)
title_dict = dict(pd.unique(pd.concat([isbn_codes,isbn_titles],axis=1).values).tolist())
isbn_code_dict = dict([(i,j) for j,i in feature_label_dict['equivalent_isbn13'].iteritems()])
knn_isbns_w2v(10,feature_label_dict['equivalent_isbn13'][9780201750546])
knn_isbns_fm(10,feature_label_dict['equivalent_isbn13'][9780201750546])
