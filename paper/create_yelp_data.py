#create_yelp_data.py
#create the yelp train/test datasets.
import pandas as pd
import numpy as np
import os
import json
import cPickle
import re
from keras.preprocessing.text import Tokenizer
datadir = '/home/luis/Data/Yelp/dataset/'

with open(os.path.join(datadir,"review.json"),"r") as f:
    review_json=f.read()
parsed_jsons=[]
num_reviews=1000000
max_vocab = 100000 #max vocab size
for i in review_json.split("\n")[0:num_reviews]:
#for i in review_json.split("\n"):
    try:
        parsed_jsons.append(json.loads(i))
    except:
        pass
review_pd=pd.DataFrame(parsed_jsons)
print review_pd.head()

#remove ambience indicators
review_pd.drop(['cool','useful','funny','date'],axis=1,inplace=True)

### Transform discrete vars to OHE
print "Preprocessing the embeddings..."
#vocab_map = {}
for c in ['business_id','user_id','stars']:
    review_pd[c] = pd.Categorical(review_pd[c])



### Filter text
print "Fitting review text data"
review_pd['text'] = review_pd['text'].map(lambda x: re.sub( '\s+', ' ', x)) #change all whitespace to spaces
review_filter = Tokenizer(num_words = max_vocab)
review_filter.fit_on_texts(review_pd.text.values.tolist())

#split train and test data
print "Splitting train-test data"
testsize = int(len(review_pd)*.1)
np.random.seed(9)
review_pd = review_pd.sample(frac=1.)
review_pd['numOccurences'] = 1
review_pd['numBiz'] = review_pd.groupby('business_id')['numOccurences'].transform('cumsum')
review_pd['numUser'] = review_pd.groupby('user_id')['numOccurences'].transform('cumsum')
pd.crosstab(review_pd['numBiz'],review_pd['numUser'])
test_index_pool = review_pd[(review_pd['numBiz']>1) & (review_pd['numUser']>1)].index


test_index = np.random.choice(test_index_pool,size=testsize,replace=False)
train_index = [x for x in review_pd.index if x not in test_index]

testdf = review_pd.loc[test_index,:]
traindf= review_pd.loc[train_index,:]

#save both the filter and the PD as pickled python objects, and the vocab map.
print "Dumping files to HDD..."
with open(os.path.join(datadir,'yelp_train_data.p'),'w') as f:
    cPickle.dump(traindf,f)
with open(os.path.join(datadir,'yelp_test_data.p'),'w') as f:
    cPickle.dump(testdf,f)
with open(os.path.join(datadir,'yelp_textfilter.p'),'w') as f:
    cPickle.dump(review_filter,f)
