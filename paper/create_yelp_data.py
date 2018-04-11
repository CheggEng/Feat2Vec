#create_yelp_data.py
#create the yelp train/test datasets.
import pandas as pd
import numpy as np
import os
import json
import cPickle
import gc
from keras.preprocessing.text import Tokenizer
import yaml
constants = yaml.load(open("paper/constants.yaml", 'rU'))
datadir =constants['datadir']

with open(os.path.join(datadir,"review.json"),"r") as f:
    review_jsons=f.read()

num_reviews=1000000
max_vocab = 100000 #max vocab size
#for i in review_json.split("\n")[0:num_reviews]:
print "Splitting json file..."
review_jsons=review_jsons.split("\n")
print len(review_jsons)
print "passing JSON files to pandas"
reviews = []
for i in review_jsons:
#for i in review_jsons[0:num_reviews]:
    try:
        reviews.append(json.loads(i))
    except:
        pass
print "Concatenating..."
reviews=pd.DataFrame(reviews)
print reviews.head()
del review_jsons
#remove ambience indicators (except funny which we keep)
reviews.drop(['cool','useful','date',],axis=1,inplace=True)
reviews.drop(['review_id'],axis=1,inplace=True)
gc.collect()
reviews.head()
print "Dumping..."
with open(os.path.join(datadir,'yelp_data.p'),'w') as f:
    cPickle.dump(reviews,f)
#
