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
                 deepin_feature=None, deepin_inputs=None, deepin_layers = None, deep_kernel_constraint=None):
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
        :param deep_kernel_constraint: the contraint for the deep out parameters
        """
        self.deep_kernel_constraint = deep_kernel_constraint

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
        if deepin_inputs==None:
            self.deepin_inputs = [None]*len(feature_dimensions)
        else:
            assert len(deepin_inputs) == sum(deepin_feature) or len(deepin_inputs) == len(feature_dimensions), "provide deep input list of length = #deep features or # features"
            if len(deepin_inputs) == len(feature_dimensions):
                self.deepin_inputs = deepin_inputs
            #if provided list = #deep features, space it out to get list of length equal to total #features
            elif len(deepin_inputs) == sum(deepin_feature):
                self.deepin_inputs = [None]*len(feature_dimensions)
                layer_count = 0
                for i in range(len(feature_dimensions)):
                    if self.deepin_feature[i]:
                        self.deepin_inputs[i] = deepin_inputs[layer_count]
                        layer_count+=1

        #construct list of extracted deep-in features
        if deepin_layers==None:
            self.deepin_layers = [None]*len(feature_dimensions)
        else:
            assert len(deepin_layers) == sum(deepin_feature) or len(deepin_layers) == len(feature_dimensions), "provide deep feature layer list of length = #deep features or # features"
            if len(deepin_layers) == len(feature_dimensions):
                self.deepin_layers = deepin_layers
            #if provided list = #deep features, space it out to get list of length equal to total #features
            elif len(deepin_layers) == sum(deepin_feature):
                self.deepin_layers = [None]*len(feature_dimensions)
                layer_count = 0
                for i in range(len(feature_dimensions)):
                    if self.deepin_feature[i]:
                        self.deepin_layers[i] = deepin_layers[layer_count]
                        layer_count+=1


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

        if deep_weight_groups==None:
            self.deep_weight_groups=None
        else:
            assert type(deep_weight_groups)==list and len(deep_weight_groups) == len(self.feature_dimensions), \
            "weight_groups must either be a list of identifiers with length = #features"
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
    def build_deep_fm_layer(self,interactions,factors):
        if self.deep_weight_groups==None:
            factors_term = Concatenate(name="factors_term")(interactions)

            return  Dense(units=1,
                          name="factor_weights", use_bias=self.deep_out_bias,
                          activation=self.deep_out_activation,
                          kernel_initializer='normal',
                          bias_initializer='normal',
                          kernel_regularizer=l2(self.l2_deep),
                          bias_regularizer=l2(self.l2_deep),
                          kernel_constraint=self.deep_kernel_constraint
                          )(factors_term)
        else:
            unique_weight_groups = []
            for w in self.deep_weight_groups:
                if w not in unique_weight_groups:
                    unique_weight_groups.append(w)
            grouped_interactions = []
            #now pre-aggregate (sum up) unique weight-index pairs
            for t1 in xrange(0, len(unique_weight_groups)):
                for t2 in xrange(t1, len(unique_weight_groups)):
                    if t1 == t2:
                        continue # do not interact with self

                    g1=unique_weight_groups[t1]
                    g2=unique_weight_groups[t2]
                    sub_interactions = []
                    int_index = 0
                    for i in xrange(0, len(factors)):
                        for j in xrange(i+1, len(factors)):
                            if (self.deep_weight_groups[i]==g1 and self.deep_weight_groups[j]==g2) \
                                or (self.deep_weight_groups[i]==g2 and self.deep_weight_groups[j]==g1):
                                sub_interactions.append(interactions[int_index])
                            int_index +=1
                    if len(sub_interactions) == 1:
                        grouped_interactions.append(sub_interactions[0])
                    elif len(sub_interactions) > 1:
                        group = Concatenate(name='group_{}x{}'.format(g1, g2))(sub_interactions)
                        if self.dropout_layer > 0:
                            group = Dropout(self.dropout_layer, name='dropout_{}x{}'.format(g1, g2))(group)
                        grouped_interactions.append(Dense(units=1,
                                                          name='grouped_interaction_{}x{}'.format(g1,g2),
                                                          activation = self.deep_out_activation,
                                                          kernel_initializer = 'normal',
                                                          bias_initializer = 'normal',
                                                          kernel_regularizer = l2(self.l2_deep),
                                                          bias_regularizer = l2(self.l2_deep),
                                                          kernel_constraint = self.deep_kernel_constraint
                                                          )(group))


            if len(grouped_interactions) == 1:
                return grouped_interactions[0]
            else:
                return Add(name="factors_term")(grouped_interactions)

    def build_model(self,
                    embedding_dimensions,
                    l2_bias=0.0, l2_factors=0.0, l2_deep=0.0, deep_out=True,
                    bias_only=None,embeddings_only=None,deep_weight_groups=None,
                    deep_out_bias=True, deep_out_activation = 'linear',
                    dropout_input=0.,
                    dropout_layer=0.,
                    **kwargs):
        """
        Builds the FM model in Keras Network
        :param embedding_dimensions: The number of dimensions of the embeddings
        :param l2_bias: L2 regularization for bias terms
        :param l2_factors: L2 regularization for interaction terms
        :param l2_deep: L2 regularization for the deep layer
        :param deep_out: Whether to have a fully connected "deep" layer to weight the factors
        :param deep_out_bias: whether to use bias terms in the "deep" intermediate layer. Only applies if the deep=True
        :paran deep_out_activation: the activation function for the "deep" intermediate layer. Only applies if the deep=True.
            should be a keyword keras recognizes
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
        features = []
        biases = []
        factors = []
        factor_features = []

        for i, dim in enumerate(self.feature_dimensions):
            #create bias/embeddings for each feature
            if self.deepin_feature[i]==True:
                #built embeddings / bias for deepin layers
                feature = self.deepin_inputs[i]
                if self.dropout_input > 0:
                    factor = Dropout(rate=self.dropout_input,name='dropout_{}'.format(self.feature_names[i]))(self.deepin_layers[i])
                else:
                    factor = self.deepin_layers[i]
                features.append(feature)
                factors.append(factor)
                factor_features.append(self.feature_names[i])
                if not self.embeddings_only[i]:
                    bias = Dense(units=1,
                            use_bias=False,
                            kernel_regularizer=l2(l2_bias),
                            kernel_initializer='normal',
                            name="bias_{}".format(self.feature_names[i]))(factor)
                    biases.append(bias)
            else:
                #initialize Input layer, then apply dropout
                if self.realval[i]==False:
                    feature,factor,bias = self.build_discrete_feature_layers(i)

                elif self.realval[i]==True:
                    feature,factor,bias = self.build_realval_feature_layers(i)
                features.append(feature)
                factor_features.append(self.feature_names[i])
                if factor is not None:
                    factors.append(factor)
                if bias is not None:
                    biases.append(bias)


        #consolidate biases to one term for the input -> embedding portion of model
        if len(biases) == 0:
            bias_term = None
        elif len(biases) == 1:
            bias_term = biases[0]
        else:
            bias_term = Add(name="biases_terms")(biases)
        #interact embedding layers corresponding to indiv. features together
        #via dot-product
        if len(factors) > 0:
            interactions = []
            for i in xrange(0, len(factors)):
                for j in xrange(i+1, len(factors)):
                    interaction = Dot(axes=-1,name="interaction_{}X{}".format(factor_features[i],
                                                                                  factor_features[j]))([factors[i],factors[j]])
                    interactions.append(interaction)
            if deep_out:
                #adds additional layer on top of embeddings that weights the interactions by feature group,
                #and adds a relu transformation.
                print "deep"
                factors_revised = self.build_deep_fm_layer(interactions,factors)
            else:
                #just outputs the linear interactions straight-up as in normal FM
                print "shallow" #,factors,len(factors)
                if len(interactions)==1:
                    factors_revised = interaction
                else:
                    factors_revised = Add(name="factors_term")(interactions)
            if bias_term is None:
                output_layer = factors_revised
            else:
                output_layer = Add(name="output_layer")([bias_term, factors_revised])
        else:
            output_layer = bias_term  # Lambda(lambda x: K.sum(x, axis=-1, keepdims=True), name="output_layer")(biases)
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
        print "~~~", features
        return Model(inputs=features, outputs=output, name="Factorization Machine",**kwargs)
        # Model.__init__(self, features, sigmoid)
