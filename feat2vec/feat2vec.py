from . import deepfm
from . import implicitsampler
reload(deepfm)
from implicitsampler import ImplicitSampler,gen_step1_probs
from deepfm import DeepFM

from keras.optimizers import TFOptimizer
import tensorflow as tf
import numpy as np
import pandas as pd
import keras

class Feat2Vec:
    def __init__(self,df,model_feature_names,feature_dimensions,model_features,sampling_features,
        embedding_dim,feature_alpha=None,sampling_alpha=None,
        obj='nce', negative_samples=1,  sampling_bias=None,batch_size=None,
        realvalued=None,deepin_feature=None,deepin_inputs=None, deepin_layers = None,
        custom_formats=None,
        dropout = 0.,
        mask_zero=False,step1_probs=None,
        sampling_items=None, sampling_probs=None,prob_dict=None,**kwargs):
        '''
        Initialize a feat2vec object
        '''
        #core parameters
        self.df = df
        self.feature_dimensions = [i for i in feature_dimensions]
        self.model_features = [i for i in model_features]
        self.sampling_features = [i for i in sampling_features]
        self.model_feature_names = [i for i in model_feature_names]
        self.deepin_feature = [i for i in deepin_feature]
        self.deepin_inputs = deepin_inputs
        self.deepin_layers = deepin_layers
        self.step1_probs = step1_probs

        self.mask_zero=mask_zero
        #sampling parameters
        self.negative_samples=negative_samples
        self.feature_alpha = feature_alpha
        self.sampling_alpha = sampling_alpha
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        self.batch_size = batch_size
        self.obj=obj
        #infer realvalued arg by dimensions
        #use fact that only realvalued args can be multicolumn
        if realvalued is None:
            self.realvalued = [False]*len(self.model_feature_names)
            for m,d,i in zip(self.model_features,self.feature_dimensions,range(len(self.model_features))):
                #print m,d,i
                if  d==len(m):
                    #print "inferring {} is real (not categorical)".format(self.model_feature_names[i])
                    self.realvalued[i]=True
            print "inferred the following values for realvalued arg:",zip(self.model_features,self.realvalued)
        else:
            self.realvalued = [i for i in realvalued]
        if custom_formats is None:
            self.custom_formats = [None]*len(model_features)
        else:
            self.custom_formats=[i for i in custom_formats]
        #add intercept term
        self.df['__offset__'] = 1
        self.model_feature_names.append('offset')
        self.model_features.append(['__offset__'])
        if self.deepin_feature is not None:
            self.deepin_feature.append(False)
        self.feature_dimensions.append(1)
        self.realvalued.append(True)
        self.custom_formats.append(None)
        #create additional deepFM args to build the model
        embeddings_only=[True]*len(self.model_features)
        embeddings_only[self.model_feature_names.index('offset')] = False
        bias_only=[False]*len(self.model_features)
        bias_only[self.model_feature_names.index('offset')] = True
        deepfm_obj = DeepFM(model_features = self.model_features,
                  feature_dimensions=self.feature_dimensions,
                  embedding_dimensions=self.embedding_dim ,
                  feature_names=self.model_feature_names, realval=self.realvalued, obj=self.obj, mask_zero = self.mask_zero,
                  deepin_feature=self.deepin_feature,deepin_inputs=deepin_inputs, deepin_layers = deepin_layers)
        self.deepfm_obj = deepfm_obj
        self.model = deepfm_obj.build_model(l2_bias=0.0, l2_factors=0.0, l2_deep=0.0,
                  deep_out=False,
                  bias_only=bias_only,embeddings_only=embeddings_only,
                  dropout_input=dropout,
                  dropout_layer=0.,
                 **kwargs)
        if step1_probs is None:
            #self.step1_probs = gen_step1_probs(self.model,self.model_feature_names[:-1],self.feature_alpha)
            self.step1_probs = self.gen_step1_probs()

        self.sampler = ImplicitSampler( df, negative_samples=self.negative_samples,
                  sampling_features=self.sampling_features,model_features=self.model_features,
                  sampling_items=sampling_items, sampling_probs=sampling_probs,
                  batch_size=batch_size,
                  sampling_strategy=None, sampling_alpha=self.sampling_alpha,sampling_bias=sampling_bias,
                  oversampler='twostep',init_probs = self.step1_probs,
                  keep_noise_probs=(self.obj=='nce'),
                  custom_formats=self.custom_formats,
                  cache_epoch=False, sample_once=False,prob_dict=prob_dict)


    def gen_step1_probs(self):
        '''
        given a keras model and feature names, extract parameter counts and return the step 1 probabilities
        for the twostep sampler
        params:
            fm: a keras model generated by DeepFM.build_model()
            feature_names: the NAMES of the features in the keras model. this is stored in the self.feature_names in a DeepFM object. if sampling_features != model_features (i.e. if some model_features are sampled together, such as, say, state and city) then you must specify a list of length = #sampling_features, where each element is itself a list of the model feature names to include for each sampling feature (in order).
            alpha: flattening parameter
        '''
        init_probs =  np.zeros(shape=len(self.sampling_features))
        for i,sample_feats in enumerate(self.sampling_features):
            for m, model_feats in enumerate(self.model_features):
                if set(model_feats).issubset(sample_feats):
                    if self.deepfm_obj.deepin_feature[m]:
                        print "deep",self.deepfm_obj.deepin_layers[m].name
                        embed_name = self.deepfm_obj.deepin_layers[m].name
                        if embed_name.find('/')!=-1:
                            embed_name = embed_name[0:embed_name.find('/')]
                        layers = [l for l in self.model.layers if l.name==embed_name]
                        print layers
                        while len(layers) > 0:
                            l = layers.pop()
                            init_probs[i] +=l.count_params()
                            for node in l.inbound_nodes:
                                print node
                                for li in node.inbound_layers:
                                    print li.name,li.count_params()
                                    layers.append(li)
                    else:
                        l = self.model.get_layer('embedding_{}'.format(self.model_feature_names[m]) )
                        init_probs[i] += l.count_params()
        print init_probs
        print "--Original Probs--"
        print init_probs/np.sum(init_probs)
        init_probs = np.power(init_probs,self.feature_alpha)
        init_probs = init_probs/np.sum(init_probs)
        print "--New Probs with alpha={}--".format(self.feature_alpha)
        print init_probs
        return init_probs


    def fit_model(self,epochs=10,num_workers=1,max_queue_size = 1,
                  optimizer= TFOptimizer(tf.train.AdamOptimizer()),callbacks = None,
                  validation_split=None,validation_generator=None):
        '''
        fits the model. basically just a wrapper for keras fit_generator
        '''
        #split validation and training data up:
        if validation_split is not None and validation_generator is not None:
            raise ValueError("you shouldn't pass args for both validation_split and validation_generator. either split the existing data, or pass a separate generator")
        elif validation_split is not None:
            isValidation = np.random.random(size=len(self.df))
            isValidation = isValidation <= np.percentile(isValidation,validation_split*100.)
            traindata = self.df.loc[isValidation==False,:]
            validationdata = self.df.loc[isValidation==True,:]
            print len(traindata),len(validationdata)
            trainsampler = ImplicitSampler( traindata, negative_samples=self.negative_samples,
                      sampling_features=self.sampling_features,model_features=self.model_features,
                      sampling_items=self.sampler.items, sampling_probs=self.sampler.probabilities,
                      batch_size=self.batch_size,
                      sampling_alpha=self.sampling_alpha,sampling_bias=self.sampler.sampling_bias,
                      oversampler='twostep',init_probs = self.sampler.init_probs,
                      custom_formats = self.custom_formats,
                      keep_noise_probs=(self.obj=='nce'),prob_dict=self.sampler.prob_dict)
            validationsampler = ImplicitSampler( validationdata, negative_samples=self.negative_samples,
                      sampling_features=self.sampling_features,model_features=self.model_features,
                      sampling_items=self.sampler.items, sampling_probs=self.sampler.probabilities,
                      batch_size=self.batch_size,
                      sampling_alpha=self.sampling_alpha,sampling_bias=self.sampler.sampling_bias,
                      oversampler='twostep',init_probs = self.sampler.init_probs,
                      custom_formats=self.custom_formats,
                      keep_noise_probs=(self.obj=='nce'),prob_dict=self.sampler.prob_dict)
            num_train_steps = sum([1 for e in trainsampler.chunker(traindata,self.batch_size)])
            num_validation_steps = sum([1 for e in validationsampler.chunker(validationdata,self.batch_size)])
            validation_data = validationsampler.keras_generator()
            generator = trainsampler.keras_generator()
        elif validation_generator is not None:
            raise ValueError('Unfortunately, this is not implemented yet. =[')
        elif validation_generator is None:
            #if we want no validation data
            validation_data = None
            num_validation_steps=0
            generator = self.sampler.keras_generator()
            num_train_steps = sum([1 for e in self.sampler.chunker(self.df,self.batch_size)])
        #-----------------
        #fit model
        self.model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())
        for x,y,w in generator:
            break
        history = self.model.fit_generator(generator=generator,
            epochs=epochs,steps_per_epoch=num_train_steps,
            validation_data=validation_data,
            validation_steps=num_validation_steps,
            callbacks=callbacks,
            max_queue_size=max_queue_size,workers=num_workers,
            use_multiprocessing=True)
        print "Done!"
        return history

    def get_embeddings(self,vocab_map = None):
        '''
        retrieve embeddings from keras model for each feature value, and export into a pretty df.
        args:

        vocab_map: an optional dictionaries of dictionaries for each model feature name.
        the keys should be elements of model_feature_names and should point themselves to dictionaries
        mapping categories to integers.
        '''
        embed_names = ['dim_{}'.format(i) for i in range(1,self.embedding_dim+1)]
        embeddings = []
        deep_count = 0
        print self.model_feature_names
        for idx,l in enumerate(self.model_feature_names):
            print l
            idx = self.model_feature_names.index(l)
            if l=='offset':
                continue
            if self.deepin_feature[idx]:
                print self.df[self.model_features[idx]]
                if self.custom_formats[idx] is not None:
                    #assumes custom format spits out something with at least 2 cols- otherwise, it would make more sense to just use another routine.
                    levels = np.array(pd.DataFrame(self.custom_formats[idx](self.df)).drop_duplicates())
                    print levels.shape
                elif len(self.model_features[idx])==1:
                    levels = np.array(pd.unique(self.df[self.model_features[idx][0]]))
                    levels = levels[:,np.newaxis]
                    print levels.shape
                else:
                    levels= np.array(self.df[self.model_features[idx]].drop_duplicates())
                if len(self.deepin_inputs) != len(self.model_features):
                    deepinNetwork = keras.models.Model(inputs=[self.deepin_inputs[deep_count]],outputs=[self.deepin_layers[deep_count]])
                else:
                    deepinNetwork = keras.models.Model(inputs=[self.deepin_inputs[idx]],outputs=[self.deepin_layers[idx]])
                deepinNetwork.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())
                weights=deepinNetwork.predict(x=[levels])
                print weights
                weights= pd.DataFrame(weights,columns = embed_names)
                weights['feature'] = l
                if 'max' in vocab_map[l].keys():
                    levels[:,self.model_features[idx].index(l)] = levels[:,self.model_features[idx].index(l)] * (vocab_map[l]['max']-vocab_map[l]['min']) + vocab_map[l]['min']

                #if self.custom_formats[idx] is not None:
                #    rawlevels =  np.array(self.df[self.model_features[idx]].drop_duplicates())
                #    print rawlevels.shape
                #else:
                #    rawlevels = levels

                if len(self.model_features[idx])==1 and  self.custom_formats[idx] is None:
                    weights['values'] = levels
                else:
                    levels =  tuple(map(tuple, levels))
                    weights['values'] = levels
                deep_count +=1
            else:
                weights = self.model.get_layer('embedding_{}'.format(l)).get_weights()[0]
                weights = pd.DataFrame(weights,columns = embed_names)
                weights['feature'] = l
                if self.realvalued[idx]:
                    weights['values'] = self.model_features[idx]
                else:
                    if self.mask_zero:
                        weights = weights.iloc[1:,:]
                    if vocab_map is None:
                        if self.mask_zero:
                            weights['values'] = range(1,self.feature_dimensions[idx])
                        else:
                            weights['values'] = range(self.feature_dimensions[idx])
                    else:
                        reverse_dict = dict([(j,i) for i,j in vocab_map[l].iteritems()])
                        if self.mask_zero:
                            weights['values'] = [reverse_dict[i] for i in range(1,self.feature_dimensions[idx])]
                        else:
                            weights['values'] = [reverse_dict[i] for i in range(self.feature_dimensions[idx])]
            embeddings.append(weights)
        embeddings = pd.concat(embeddings)
        embeddings = embeddings[['feature','values'] + embed_names]
        return embeddings
