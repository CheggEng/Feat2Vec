#define a "complete" sampler for each major
#that will generate all negative labels during data generation
import pandas as pd
import sys
import numpy as np
class CompleteSampler():
    def __init__(self, df,sampled_feature,items,model_features,identifier,batch_size=None):
        '''
        create a CompleteSampler object
        params: 
        df: the df to get batches from
        batch_size: how many positive labels per batch
        model_features: which models go into the keras model
        items: the set of all possible classes
        sampled_feature: which feature we have to make negative samples of.
        identifier: column name in df identifying entity we want to do the rankings along.
        '''
        self.df=df
        self.batch_size=batch_size
        self.model_features=model_features
        self.items = items
        self.sampled_feature=sampled_feature
        self.identifier=identifier
    def chunker(self,df):
        '''
        turns a df into batches of appropriate size
        '''
        return (df.iloc[pos:pos + self.batch_size] for pos in xrange(0, len(df), self.batch_size))
    def keras_generator(self):
        '''
        transforms batches into something keras can process
        '''
        while True:
            xs, ys, ws = [], [], []
            for x, y, w in self.sample_fast():
                yield x.to_keras_flex_format(feature_cols = self.model_features), y, w
    def sample_fast(self):
        '''
        shuffles a dataframe, then samples batches
        '''
        shuffled_df = self.df.sample(frac=1)
        for batch in self.chunker(shuffled_df):
            aug_batch, ys, ws = self.extend_batch_totalsample(batch)
            yield aug_batch, ys, ws
    def extend_batch_totalsample(self,batch):
        '''
        populates a batch to include all negative examples per user_id 
        '''
        pos_pairs = set([tuple(p) for p in batch[[self.identifier, self.sampled_feature]].values.tolist()])
        #print pos_pairs
        aug_batch = pd.concat([batch]*len(self.items)).sort_values([self.identifier])
        aug_batch[self.sampled_feature] =  self.items * len(batch)
        y = np.zeros(shape=(len(aug_batch),))
        is_pos = np.array([r in pos_pairs for r in \
                           [tuple(p) for p in aug_batch[[self.identifier,self.sampled_feature]].values.tolist()] ])
        y[is_pos] = 1.
        w = np.array([1.]*len(aug_batch))
        return aug_batch,y,w
    def evaluate_rank(self,model):
        '''
        ranks predictions of model across all classes.
        '''
        num_samples =  sum(1 for e in self.chunker(self.df))
        #print num_samples
        sample_index= 0
        ranks=[]
        for x,y,_ in self.sample_fast():
            sys.stdout.write("\r Evaluated {s}/{ns} Batches".format(s=sample_index,ns=num_samples))
            sys.stdout.flush()
            y_predictions = model.predict_on_batch(x.to_keras_flex_format(feature_cols = self.model_features))
            x['__y__'] = y
            x['__y_prob__'] = y_predictions
            x['__ranks__'] = x.groupby(self.identifier)["__y_prob__"].rank(method='dense',ascending=False)
            ranks.append(x.loc[x['__y__']==1,'__ranks__'])
            sample_index+=1
            if sample_index >= num_samples:
                break
        rankings = pd.concat(ranks)
        return rankings