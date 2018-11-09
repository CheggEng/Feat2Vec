#This file contains the code to build a Deep Factorization Machine model
#
import tensorflow as tf
import numpy as np
from keras.layers import Embedding, Reshape, Activation, Lambda, Input, Dropout, Dense,SpatialDropout1D
from keras.regularizers import l2
from keras.initializers import RandomNormal
from keras.models import Model
from keras.utils.generic_utils import Progbar
from keras.layers.merge import Add,Multiply,Concatenate,Dot
from keras.constraints import non_neg
from scaler import Scaler
import keras.backend as K
import itertools

def flatten(list_of_lists):
    flattened = []
    for sublist in list_of_lists:
        flattened.append("_".join(sublist))
    return flattened

def NCEobj(probs):
        '''
        transforms the objective so that we can evaluate the model via noise contrastive estimation
        noise_probs = k*q(w)
        model_probs = p_theta (w|c)
        NOISE_PROBS_MULTIPLIER is a constant used to scale noise_probabilities so they are "close" to 1.
        so the model doesn't get stuck with astronomically low or high noise probs
        '''
        NOISE_PROB_MULTIPLIER = 1.
        import tensorflow as tf
        #model_probs,noise_probs,noise_multiplier = probs
        model_probs,noise_probs = probs
        #print model_probs,noise_probs
        return tf.div( tf.exp(model_probs),tf.add(tf.exp(model_probs),noise_probs*NOISE_PROB_MULTIPLIER) )
def nce_output_shape(input_shape):
    return (1)




class DeepFM():
    def __init__(self, model_features, feature_dimensions, feature_names=None,
                 realval=None, obj='ns', mask_zero=False,
                 deepin_feature=None, deepin_inputs=[], deepin_layers =[]):
        """
        Initializes a Deep Factorization Machine Model
        :param model_features: a list of lists of columns for each feature / embedding in the model
        :param feature_dimensions: A list where each entry represents the number of possible values
            a discrete feature has if it is a categorical. Otherwise, if the feature is real-valued, it
            indicates the dimensionality of the feature (how many cols)
         (just a number)
         If zero, only biases will be used, and there will be no embeddings.
         If you only have one feature, this MUST be zero (because there are no interactions)
        :param feature_names: the names of the features to be used in the training sample
        :param realval: whether or not features are real-valued:
        (either True or False which indicates all are Real-valued or categories/indices, respectively,
        or a list as long as feature_dimensions of Booleans indicating status of each feature.
        only relevant for features that are not deep-in
        :param obj: the objective function used for evaluating the model (ns=negative sampling, nce=noise contrastive estimation)
        :param deepin_feature: a bool list of length features that specifies whether the
         feature requires a deep feature extraction. if a
        :param deepin_layers: a list of keras layers, corresponding to each "deep" feature. this will be directly input into the FM
            as if it were an factor. It is assumed that deepin_layers output a real-valued matrix of dimension = feature_dimensions
            specified in the feature_dimensions list.
        :param deepin_inputs: a list of keras layers, corresponding to each raw input feature for the feature extraction.
            this will be directly input into the keras model as an Input tensor.
        :param mask_zero: a toggle to mask ALL zero values for categoricals as zero vectors
        """


        if realval is None:
            self.realval = [False]*len(feature_dimensions) #default to all categoricals
        else:
            self.realval = realval
        assert (type(self.realval)==list) and len(self.realval) == len(feature_dimensions),            "realval must either be a boolean list with length = #features, or None"



        self.model_features = model_features
        self.feature_dimensions = feature_dimensions
        if feature_names is None:
            self.feature_names =  flatten(model_features)
        else:
            self.feature_names = feature_names

        assert len(self.feature_names) == len(self.feature_dimensions), "lengths do not match"

        assert obj=='ns' or obj=='nce',"obj. function must be negative sampling (ns) or noise contrastive estimation (nce)"
        self.obj = obj
        self.mask_zero=mask_zero
        #####
        #Deep-in feature indicators
        if deepin_feature == None:
            self.deepin_feature = [False]*len(feature_dimensions) #default to all categoricals
        else:
            print len(deepin_feature),len(feature_dimensions)
            assert len(deepin_feature) == len(feature_dimensions), "must provide boolean list w/ length=#features"
            self.deepin_feature = deepin_feature

        #construct list of deep-in inputs
        self.deepin_inputs = [deepin_inputs.pop(0) if self.deepin_feature[i] else None for i in xrange(len(feature_dimensions))]
        assert(len(deepin_inputs) == 0), "provide deep input list of length = #deep features or # features"

        #construct list of extracted deep-in features
        self.deepin_layers = [deepin_layers.pop(0) if self.deepin_feature[i] else None for i in xrange(len(feature_dimensions))  ]
        assert (len(deepin_layers) == 0), "provide deep feature layer list of length = #deep features or # features"

    def check_build_params(self,l2_bias, l2_factors, l2_deep,bias_only,embeddings_only,deep_weight_groups,
                    deep_out_bias, deep_out_activation,
                    dropout_input,
                    dropout_layer):
        '''
        confirm that all passed params are of correct format and attach to model object
        '''
        assert type(l2_bias)==float and type(l2_factors)==float and type(l2_deep)==float, \
        "L2 regularization terms must all be floats"
        assert l2_bias >= 0. and l2_bias >= 0. and l2_bias >= 0., \
        "L2 regularization terms must be non-negative"
        self.l2_bias = l2_bias
        self.l2_factors = l2_factors
        self.l2_deep = l2_deep
        if bias_only is None:
            self.bias_only = [False]*len(self.feature_dimensions)
        else:
            assert type(bias_only)==list and len(bias_only) == len(self.feature_dimensions), \
            "bias_only must be a boolean list with length = #features, or None"
            self.bias_only = bias_only

        if embeddings_only is None:
            self.embeddings_only = [True]*len(self.feature_dimensions)
        else:
            assert type(embeddings_only)==list and len(embeddings_only) == len(self.feature_dimensions),\
            "embeddings_only    must either be a boolean list with length = #features, or None"
            self.embeddings_only = embeddings_only

        if deep_weight_groups is None:
            self.deep_weight_groups= [ itertools.combinations(self.feature_names, 2) ]
        else:
            for g in deep_weight_groups:
                if not isinstance(g, list):
                    raise RuntimeError("deep_weight_groups must be a list of lists. Where each element is a tuple for an interaction")
            self.deep_weight_groups=deep_weight_groups


        assert type(deep_out_bias) == bool,'deepout_bias must be a boolean'
        self.deep_out_bias = deep_out_bias
        self.deep_out_activation = deep_out_activation
        assert dropout_layer>=0. and dropout_layer <=1. and dropout_input >=0. and dropout_input <=1, \
        "Dropout args should be in [0,1]"
        self.dropout_input = dropout_input
        self.dropout_layer = dropout_layer

    def build_discrete_feature_layers(self,feature_index):
        '''
        create keras bias and embedding layers (where relevant depending on bias_only, embeddings_only)
        for a discrete categorical feature, where each integer represents a new category
        args:
            feature_index: the position of the feature in question in our list of features
        '''
        feature_dim = self.feature_dimensions[feature_index]
        feature_cols = len(self.model_features[feature_index])
        feature = Input(batch_shape=(None, feature_cols), name=self.feature_names[feature_index])
        if (self.embedding_dimensions > 0) and (not self.bias_only[feature_index]):
            ftemp = Embedding(input_dim=feature_dim,
                          output_dim=self.embedding_dimensions,
                          embeddings_regularizer=l2(self.l2_factors),
                          input_length=feature_cols,
                          embeddings_initializer='normal',
                          mask_zero = self.mask_zero,
                          name="embedding_{}".format(self.feature_names[feature_index]))(feature)
            if self.dropout_input > 0:
                ftemp_filtered = SpatialDropout1D(self.dropout_input,
                    name='dropout_embedding_{}'.format(self.feature_names[feature_index]))(ftemp)
            else:
                ftemp_filtered = ftemp
            if feature_cols > 1:
                ftemp_filtered = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), name="avg_embedding_{}".format(self.feature_names[feature_index]))(ftemp_filtered)
                #if self.mask_zero ==True:
            ftemp_filtered = Lambda(lambda x: x, output_shape=lambda s:s,name='unmasker_{}'.format(self.feature_names[feature_index]))(ftemp_filtered)
            factor = Reshape((self.embedding_dimensions,),
                name="embedding_{}_reshaped".format(self.feature_names[feature_index]))(ftemp_filtered)
        else:
            factor=None
        #bias term for categ. feature
        if not self.embeddings_only[feature_index]:
            btemp = Embedding(input_dim=feature_dim,
                              output_dim=1,
                              input_length=feature_cols,
                              mask_zero = self.mask_zero,
                              embeddings_regularizer=l2(self.l2_bias),
                              embeddings_initializer='normal',
                              name="bias_{}".format(self.feature_names[feature_index]))(feature)
            if self.dropout_input > 0:
                btemp_filtered = SpatialDropout1D(self.dropout_input,
                    name='dropout_biased_{}'.format(self.feature_names[feature_index]))(btemp)
            else:
                btemp_filtered = btemp
            if feature_cols > 1:
                btemp_filtered = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), name="avg_bias_{}".format(self.feature_names[feature_index]))(btemp_filtered)
            if self.mask_zero ==True:
                btemp_filtered = Lambda(lambda x: x, output_shape=lambda s:s,name='unmasker_bias_{}'.format(self.feature_names[feature_index]))(btemp_filtered)
            bias = Reshape((1,),
                           name="bias_{}_reshaped".format(self.feature_names[feature_index]))(btemp_filtered)
        else:
            bias=None
        return feature,factor,bias
    def build_realval_feature_layers(self,feature_index):
        '''
        create keras bias and embedding layers (where relevant depending on bias_only, embeddings_only)
        realvalued variable. so each column in this feature is passed is interpreted as a number and passed through a linear fully connected layer
        '''
        feature_dim = self.feature_dimensions[feature_index]
        feature = Input(batch_shape=(None, feature_dim), name=self.feature_names[feature_index])
        if self.dropout_input > 0:
            feature_filtered = Dropout(self.dropout_input,name='dropout_{}'.format(self.feature_names[feature_index]))(feature)
        else:
            feature_filtered = feature

        if (self.embedding_dimensions > 0) and (not self.bias_only[feature_index]):
            factor = Dense(units=self.embedding_dimensions,
                  use_bias = False,
                  kernel_regularizer=l2(self.l2_factors),
                  kernel_initializer='normal',
                  name="embedding_{}".format(self.feature_names[feature_index]))(feature_filtered)

        else:
            factor=None
        if not self.embeddings_only[feature_index]:
            bias = Dense(units=1,
                use_bias=False,
                kernel_regularizer=l2(self.l2_bias),
                kernel_initializer='normal',
                name="bias_{}".format(self.feature_names[feature_index]))(feature_filtered)
        else:
            bias=None
        return feature, factor, bias


    def build_variables(self, i,  inputs, biases, factors):

        if inputs[i] is None: # The input hasn't been created, so we must do so:
            #create bias/embeddings for each feature

            if self.deepin_feature[i]:
                feature, factor, bias = None, None, None

                feature = self.deepin_inputs[i]
                factor = self.deepin_layers[i]
                if not self.embeddings_only[i]:
                    bias = Dense(units=1,
                            use_bias=False,
                            kernel_initializer='normal',
                            name="bias_{}".format(self.feature_names[i]))(factor)


            elif self.realval[i]:
                feature,factor,bias = self.build_realval_feature_layers(i)
            else:
                feature, factor, bias = self.build_discrete_feature_layers(i)

            # Save layers:
            inputs[i] = feature
            factors[i] = factor
            biases[i] = bias

        else: # We've created the input, so no need to do anything:
            factor = factors[i]

        return factor



    def two_way_interactions(self, collapsed_type, deep_out, deep_kernel_constraint):
        # Calculate interactions with a dot product

        inputs =  [None] * len(self.feature_names)
        biases =  [None] * len(self.feature_names)
        factors = [None] * len(self.feature_names)
        interactions = []

        for i, groups in enumerate(self.deep_weight_groups):

            for grp, (feature_i, feature_j) in enumerate(groups):
                #factor_i = factors[self.feature_names.index(feature_i)]
                index_i = self.feature_names.index(feature_i)
                factor_i = self.build_variables(index_i, inputs, biases, factors)
                if isinstance(feature_j, str) or isinstance(feature_j, unicode):
                    #factor_j = factors[ self.feature_names.index(feature_j) ]
                    index_j = self.feature_names.index(feature_j)
                    factor_j = self.build_variables(index_j, inputs, biases, factors)
                    name_j = feature_j
                elif isinstance(feature_j, list):
                    name_j = "grp_{}".format("{:03d}".format(i))

                    if collapsed_type is not None:
                        #constant = np.zeros( (self.embedding_dimensions,) )
                        #k_constant = K.constant(constant, shape=(self.embedding_dimensions,))

                        collapsed_input = Input( shape=(1, ), name="input_{}".format(name_j))
                        embedding = Embedding(input_dim=collapsed_type,
                                              name = "embedding_{}".format(name_j),
                                              output_dim=self.embedding_dimensions)(collapsed_input)

                        inputs.append (collapsed_input)
                        factor_j = embedding
                    else:
                        factors_j = []
                        for feature_j_n in feature_j:
                            #factor =  factors[ self.feature_names.index(feature_j_n) ]
                            index_j_n = self.feature_names.index(feature_j_n)
                            factor = self.build_variables(index_j_n, inputs, biases, factors)

                            if deep_out:
                                if self.dropout_layer > 0:
                                    factor = Dropout(self.dropout_layer, name="dropout_terms_{}_{}".format(feature_j_n, i))(factor)
                                factor = Scaler(name="scaler_{}_{}".format(feature_j_n, i), constraint=deep_kernel_constraint)(factor)
                            factors_j.append(factor)

                        if len(factors_j) == 1:
                            factor_j = factor_j[0]
                        else:
                            factor_j = Add(name=name_j)(factors_j) # collapse them

                if factor_i is None:
                    print "Warning... {} does not have an embedding".format(feature_i)
                    continue
                if factor_j is None:
                    print "Warning... {} does not have an embedding".format(feature_j)
                    continue

                dot_product = Dot(axes=-1, name="interaction_{}X{}".format(feature_i, name_j))
                interactions.append(dot_product([factor_i,factor_j]))

        # Add whatever you have now:
        if len(interactions)==0:
            two_way = None
        elif len(interactions)==1:
            two_way =  interactions[0:]
        else:
            two_way = Add(name="factors_term")(interactions)

        return inputs, biases, two_way



    def build_model(self,
                    embedding_dimensions,
                    collapsed_type=None,
                    l2_bias=0.0, l2_factors=0.0, l2_deep=0.0, deep_out=True,
                    bias_only=None, embeddings_only=None, deep_weight_groups=None,
                    deep_out_bias=True, deep_out_activation = 'linear', deep_kernel_constraint=None,
                    dropout_input=0.,
                    dropout_layer=0.,

                    **kwargs):
        """
        Builds the FM model in Keras Network
        :param embedding_dimensions: The number of dimensions of the embeddings
        :param collapsed_type: None if the model shouldn't collapsed, or # of embedding types if it should
        :param l2_bias: L2 regularization for bias terms
        :param l2_factors: L2 regularization for interaction terms
        :param l2_deep: L2 regularization for the deep layer
        :param deep_out: Whether to have a fully connected "deep" layer to weight the factors
        :param deep_out_bias: whether to use bias terms in the "deep" intermediate layer. Only applies if the deep=True
        :paran deep_out_activation: the activation function for the "deep" intermediate layer. Only applies if the deep=True.
            should be a keyword keras recognizes
        :param deep_kernel_constraint whether to have a constraint on the deep layer
        :param weight_groups: an ordered list of integers representing weight groups for each feature; these correspond to the
            t() function in the DeepFM paper(section II.B.1). Default is each input feature gets there own unique weight for the deep layer.
            NOTE: my current implementation , for dropout in the deep layer, when using weight groups, will equally weight each unique weight group
            cross-sectionm rather than individual interactions.
        :param bias_only: a list of booleans of length # features that determines whether to only use the bias and not allow the designated
            feature to be interacted
        :param dropout_input: Dropout rate for the features,
        :param dropout_layer: Dropout rate for the factors entering the deep layer. Only applies if the deep=True
        :param
        :param kwargs: any additional arguments passed directly to the Model Keras class initializer
        Returns a Keras model
        """
        self.embedding_dimensions = embedding_dimensions

        self.check_build_params(l2_bias, l2_factors, l2_deep,bias_only,embeddings_only,deep_weight_groups,
                    deep_out_bias, deep_out_activation,
                    dropout_input,
                    dropout_layer)

        features, biases,factors =  self.two_way_interactions(collapsed_type, deep_out, deep_kernel_constraint)

        features = [f for f in features if f is not None]
        biases   = [f for f in biases if f is not None]

        #1-way interactions:
        if len(biases) == 0:
            bias_term = None
        elif len(biases) == 1:
            bias_term = biases[0]
        else:
            bias_term = Add(name="biases_terms")(biases)

        #2-way interactions
        # interact embedding layers corresponding to indiv. features together via dot-product


        # Combine 1-way and 2-way interactions:
        if (bias_term is not None) and (factors is not None):
            output_layer = Add(name="output_layer")([bias_term, factors])
        elif factors is not None:
            output_layer = factors
        else:
            output_layer = bias_term  # Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), name="output_layer")(biases)

        assert output_layer is not None

        if self.obj=='ns':
            output = Reshape((1,),name='final_output')(Activation('sigmoid', name="sigmoid")(output_layer))
        elif self.obj=='nce':
            noise_probs = Input(batch_shape=(None, 1), name='noise_probs')
            features.append(noise_probs)
            #noise_mult_tensor = K.constant([self.noise_multiplier],name='noise_mult_tens')
            #noise_multiplier = Input(batch_shape=(None,1),name='noise_multiplier',tensor=noise_mult_tensor)
            output = Lambda(NCEobj,name='nce_obj')([output_layer,noise_probs])
        global graph
        graph = tf.get_default_graph()

        return Model(inputs=features, outputs=output, name="Factorization Machine",**kwargs)
        # Model.__init__(self, features, sigmoid)


class Feat2VecModel(DeepFM):
    def __init__(self, features, obj='ns', mask_zero=False):
        sequence_lengths     = []
        vocabulary_sizes     = []
        feature_names        = []
        continuous_variables = []
        deepin_features      = []
        deepin_inputs        = []
        deepin_layers        = []
        for i, feature in enumerate(features):
            feature_name    = feature.get("name", "feature_{}".format(i))
            sequence_length = ["feature_{}_{}".format(i, j) for j in range(0, feature.get("length", 1))]
            feature_type = feature["type"]

            vocabulary_size, continuous, deepin_feature, deepin_input, deepin_output = None, None, None, None, None
            if feature_type == "discrete":
                if "vocab" not in feature:
                    raise RuntimeError("Feature '{}' is discrete, but does not have vocab_size property".format(feature_name))
                vocabulary_size = feature["vocab"]
                continuous = False
                deepin_feature = False
            elif feature_type ==  "real":
                if "vocab" in feature:
                    raise  RuntimeError("Vocabulary size not expected in real feature {}".format(feature_name))
                vocabulary_size = 1
                continuous = True
                deepin_feature = False
            elif isinstance(feature, dict):
                if "vocab" in feature:
                    raise  RuntimeError("Vocabulary size not expected in complex feature {}".format(feature_name))
                if ("input" not in feature["type"]) or ("output" not in feature["type"]):
                    raise RuntimeError ("Feature '{}' is complex, but does not have input and output embeddings. {}".format(feature_name, feature["type"]))
                vocabulary_size = 1
                continuous = True
                deepin_feature = True
                deepin_input  = feature["type"]["input"]
                deepin_output = feature["type"]["output"]
            else:
                raise RuntimeError("Feature type should be 'discrete'|'real'|or a dictionary with layers")

            sequence_lengths.append(sequence_length)
            vocabulary_sizes.append(vocabulary_size)
            feature_names.append(feature_name)
            continuous_variables.append(continuous)
            deepin_features.append(deepin_feature)

            if deepin_feature:
                deepin_inputs.append(deepin_input)
                deepin_layers.append(deepin_output)

        DeepFM.__init__(self,
                        model_features=sequence_lengths,
                        feature_dimensions=vocabulary_sizes,
                        feature_names=feature_names,
                        realval=continuous_variables,
                        obj=obj,
                        mask_zero=mask_zero,
                        deepin_feature=deepin_features,
                        deepin_inputs=deepin_inputs,
                        deepin_layers =deepin_layers)


