import numpy as np
import pandas as pd
from unittest import TestCase
from feat2vec import ImplicitSampler

np.random.seed(1)
#generate some data
samplesize = 100000
batch_size=2000
neg_samples=5
embed_dim = 10

class TestDeepFM(TestCase):
    def test_sampler(self):
        attrs = np.random.multivariate_normal(mean = np.array([0.,0.]), cov = np.array([[1.,.1],[.1,1]]),size=samplesize)
        testdata = pd.DataFrame({ 'item': np.random.randint(0,10,size=samplesize),
                                  'id': np.random.randint(0,50,size=samplesize),
                                  'attr1':attrs[:,0],
                                  'attr2':attrs[:,1],
                                  'attr3': np.random.uniform(size=samplesize),
                                  'offset_':np.ones(samplesize)
            })

        testdata['latenty'] = ( (testdata.attr3)**4/600. + np.exp(testdata.attr1*testdata.attr2) \
                               + (testdata.item<=3)*np.pi - ((testdata.item==2) | (testdata.item==4))*testdata.attr3 \
                               + ((testdata.item==8) | (testdata.item==7) )*testdata.attr2 \
                               + (testdata.item>=6)*testdata.attr1 + (testdata.id%3)*testdata.attr3 \
                               + np.random.normal(size=samplesize) - 5 )
        testdata['latenty'][np.abs(testdata['latenty']) >= 10] = 0
        testdata['y'] = np.floor(testdata.latenty)
        print testdata.y.value_counts()/samplesize

        model_features = [['item'],['id'],['attr1','attr2','attr3'],['y'],['offset_']]
        features = ['item','id','attrs','y','offset']
        sampling_features = model_features
        #itemset_list = [items3,items12]

        #build the item list myself



        sampler = ImplicitSampler(testdata,model_features=model_features,sampling_features=sampling_features,batch_size=batch_size,
                               sampling_alpha=.25,sampling_bias=0,oversampler='twostep',negative_samples=neg_samples)


        print "^*^*^*^*^*"
        ind=1
        for x,y,w in sampler.keras_generator():
            ycounts= pd.Series(x[features.index('y')].flatten()).value_counts()/(batch_size*(1+neg_samples))
            print ycounts
            if ind >=1:
                break
            ind+=1
            print "----"

