from . import deepfm
from . import implicitsampler
from implicitsampler import ImplicitSampler,gen_step1_probs
from deepfm import DeepFM
from keras.optimizers import TFOptimizer
import tensorflow as tf
import numpy as np
import keras
class Feat2Vec:
    def __init__(self,df,model_feature_names,feature_dimensions,model_features,sampling_features,
        embedding_dim,feature_alpha=None,sampling_alpha=None,
        obj='nce', negative_samples=1,  sampling_bias=None,batch_size=None,
        realvalued=None,deepin_feature=None,deepin_inputs=None, deepin_layers = None,
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
        #add intercept term
        self.df['__offset__'] = 1
        self.model_feature_names.append('offset')
        self.model_features.append(['__offset__'])
        self.feature_dimensions.append(1)
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
                if len(m)>1 or d==len(m):
                    #print "inferring {} is real (not categorical)".format(self.model_feature_names[i])
                    self.realvalued[i]=True
            print "inferred the following values for realvalued arg:",zip(self.model_features,self.realvalued)
        else:
            self.realvalued = realvalued

        #create additional deepFM args to build the model
        embeddings_only=[True]*len(self.model_features)
        embeddings_only[self.model_feature_names.index('offset')] = False
        bias_only=[False]*len(self.model_features)
        bias_only[self.model_feature_names.index('offset')] = True
        deepfm_obj = DeepFM(feature_dimensions=self.feature_dimensions,
                  embedding_dimensions=self.embedding_dim ,
                  feature_names=self.model_feature_names, realval=self.realvalued, obj=self.obj,
                  deepin_feature=deepin_feature,deepin_inputs=deepin_inputs, deepin_layers = deepin_layers)
        self.model = deepfm_obj.build_model(l2_bias=0.0, l2_factors=0.0, l2_deep=0.0,
                  deep_out=False,
                  bias_only=bias_only,embeddings_only=embeddings_only,
                  dropout_input=0.,
                  dropout_layer=0.,
                 **kwargs)
        step1_probs = gen_step1_probs(self.model,self.model_feature_names[:-1],self.feature_alpha)

        self.sampler = ImplicitSampler( df, negative_samples=self.negative_samples,
                  sampling_features=self.sampling_features,model_features=self.model_features,
                  sampling_items=sampling_items, sampling_probs=sampling_probs,
                  batch_size=batch_size,
                  sampling_strategy=None, sampling_alpha=self.sampling_alpha,sampling_bias=sampling_bias,
                  oversampler='twostep',init_probs = step1_probs,
                  keep_noise_probs=(self.obj=='nce'),
                  text_dict=None,text_cutoff=None,
                  cache_epoch=False, sample_once=False,prob_dict=prob_dict)

    def fit_model(self,epochs=10,num_workers=1,max_queue_size = 1,
                  optimizer= TFOptimizer(tf.train.AdamOptimizer()),callbacks = [],
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
                      keep_noise_probs=(self.obj=='nce'),prob_dict=self.sampler.prob_dict)
            validationsampler = ImplicitSampler( validationdata, negative_samples=self.negative_samples,
                      sampling_features=self.sampling_features,model_features=self.model_features,
                      sampling_items=self.sampler.items, sampling_probs=self.sampler.probabilities,
                      batch_size=self.batch_size,
                      sampling_alpha=self.sampling_alpha,sampling_bias=self.sampler.sampling_bias,
                      oversampler='twostep',init_probs = self.sampler.init_probs,
                      keep_noise_probs=(self.obj=='nce'),prob_dict=self.sampler.prob_dict)
            num_train_steps = sum([1 for e in trainsampler.chunker(traindata,self.batch_size)])
            num_validation_steps = sum([1 for e in validationsampler.chunker(validationdata,self.batch_size)])
        elif validation_generator is not None:
            raise ValueError('Unfortunately, this is not implemented yet. =[')
        #-----------------
        #fit model
        self.model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=tf.train.AdamOptimizer())
        self.model.fit_generator(generator=trainsampler.keras_generator(),
            epochs=epochs,steps_per_epoch=num_train_steps,
            validation_data=validationsampler.keras_generator(),
            validation_steps=num_validation_steps,
            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss')],
            max_queue_size=max_queue_size,workers=num_workers,
            use_multiprocessing=True)
        print "Done!"

    def get_embeddings(self):
        '''
        extract embeddings from keras model into a pretty pandas df
        '''
        print "not ready!"
