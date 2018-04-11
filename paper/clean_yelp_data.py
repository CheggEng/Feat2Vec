#clean features + split test and train data

import pandas as pd
import numpy as np
import os
import json
import cPickle
import gc
import re
from keras.preprocessing.text import Tokenizer
import yaml
constants = yaml.load(open("paper/constants.yaml", 'rU'))
datadir =constants['datadir']

max_vocab = 100000 #max vocab size

with open(os.path.join(datadir,'yelp_data.p'),'r') as f:
    reviews= cPickle.load(f)
#

### Transform discrete vars to OHE
print "Preprocessing the embeddings..."
#vocab_map = {}
for c in ['business_id','user_id','stars','funny']:
    reviews[c] = pd.Categorical(reviews[c])



### Filter text
print "removing whitespace "
reviews['text'] = reviews['text'].map(lambda x: re.sub( '\s+', ' ', x)) #change all whitespace to spaces
print "Fitting review text data"
review_filter = Tokenizer(num_words = max_vocab)
review_filter.fit_on_texts(reviews.text.values.tolist())
print "creating sequences of text"
reviews['textseq'] = review_filter.texts_to_sequences(reviews.text.values.tolist())
reviews['textseq'] = reviews['textseq'].map(lambda x: tuple(x))
reviews.drop('text',axis=1,inplace=True)
#
#split train and test data
print "Splitting train-test data"
testsize = int(len(reviews)*.1)
np.random.seed(9)
reviews = reviews.sample(frac=1.)
reviews['numOccurences'] = 1
reviews['numBiz'] = reviews.groupby('business_id')['numOccurences'].transform('cumsum')
reviews['numUser'] = reviews.groupby('user_id')['numOccurences'].transform('cumsum')
#pd.crosstab(review_pd['numBiz'],review_pd['numUser'])
test_index_pool = reviews[(reviews['numBiz']>1) & (reviews['numUser']>1)].index

print "Generating Indices..."
test_index = np.random.choice(test_index_pool,size=testsize,replace=False)
test_set = set(test_index)
len(reviews)
train_index = [x for x in reviews.index if x not in test_set]
len(train_index) + len(test_index)
testdf = reviews.loc[test_index,:]
traindf= reviews.loc[train_index,:]

#save both the filter and the PD as pickled python objects, and the vocab map.
print "Dumping files to HDD..."

with open(os.path.join(datadir,'yelp_clean_data.p'),'w') as f:
    cPickle.dump(reviews,f)
with open(os.path.join(datadir,'yelp_train_data.p'),'w') as f:
    cPickle.dump(traindf,f)
with open(os.path.join(datadir,'yelp_test_data.p'),'w') as f:
    cPickle.dump(testdf,f)


with open(os.path.join(datadir,'yelp_textfilter.p'),'w') as f:
    cPickle.dump(review_filter,f)
