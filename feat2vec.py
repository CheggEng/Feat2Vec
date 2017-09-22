from . import deepfm
from . import implicitsampler
from implicitsampler import ImplicitSampler,gen_step1_probs
from deepfm import DeepFM


class Feat2Vec:
    def __init__(df,model_feature_names,feature_dims,model_features,sampling_features,
        embedding_dim,feature_alpha=None,sampling_alpha=None,
        obj='nce', negative_samples=1,  sampling_bias=None,
        deepin_feature=None,deepin_inputs=None, deepin_layers = None,
        sampling_items=None, sampling_probs=None,prob_dict=None):
        '''
        Initialize a feat2vec object
        '''
        #core parameters
        self.df = df
        self.feature_dims = feature_dims
        self.model_features = model_features
        self.sampling_features = sampling_features
        self.model_feature_names = model_feature_names
        #add intercept term
        self.df['__offset__'] = 1
        self.model_feature_names.append('__offset__')
        self.model_features.append(['__offset__'])
        self.feature_dims.append(1)

        #sampling parameters
        self.negative_samples=negative_samples
        self.feature_alpha = feature_alpha
        self.sampling_alpha = sampling_alpha
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples

        #infer realvalued arg by dimensions
        #use fact that only realvalued args can be multicolumn
        self.realvalued = [False]*len(model_feature_names)
        for m,d,i in zip(model_features,feature_dims,range(len(model_features))):
         if len(m)>1 or d==len(m):
             self.realvalued[i]==True

        #create additional deepFM args to build the model
        embeddings_only=[True]*len(model_features)
        embeddings_only[model_feature_names.index('__offset__')] = False
        bias_only=[False]*len(model_features)
        bias_only[model_feature_names.index('__offset__')] = True

        deepfm_obj = DeepFM(feature_dimensions=self.feature_dimensions,
                  embedding_dimensions=self.embedding_dim ,
                  feature_names=self.model_feature_names, realval=self.realvalued, obj=self.obj,
                  deepin_feature=deepin_feature,deepin_inputs=deepin_inputs, deepin_layers = deepin_layers)
        self.model = deepfm_obj.build_model(l2_bias=0.0, l2_factors=0.0, l2_deep=0.0,
                  deep_out=False,
                  bias_only=bias_only,embeddings_only=embeddings_only,
                  dropout_input=0.,
                  dropout_layer=0.,
                 **kwargs))
        step1_probs = gen_step1_probs(self.model,self.model_feature_names,self.feature_alpha)

        self.sampler = ImplicitSampler( df, negative_samples=self.negative_samples,
                  sampling_features=self.sampling_features,model_features=self.model_features,
                  sampling_items=sampling_items, sampling_probs=sampling_probs,
                  batch_size=self.batch_size,
                  sampling_strategy=None, sampling_alpha=self.sampling_alpha,sampling_bias=self.sampling_bias,
                  oversampler='twostep',init_probs = step1_probs,
                  keep_noise_probs=(self.obj=='nce'),
                  text_dict=None,text_cutoff=None,
                  cache_epoch=False, sample_once=False,prob_dict=prob_dict):)

    def fit_model(self,num_epochs=10,num_workers=1,max_queue_size = 1,
                  optimizer= TFoptimizer(tf.train.optimizer.Adam))


    def get_embeddings(self):
