from keras.utils import Sequence
import numpy as np
import math
import scipy

class FMData(Sequence):

    def __init__(self, inputs, output, batch_size, implicit_samples=0,splits=None, feature_extraction=None, sample_probabilities={},  mask=None, shuffle=True, nce=None):
        #validate inputs:
        check_length = -1
        for feature in inputs:
            if check_length == -1:
                check_length = feature.shape[0]
                continue
            if check_length != feature.shape[0]:
                raise RuntimeError("Input features do not have same length: {}, {}".format(check_length, feature.shape[0]))
        if check_length != output.shape[0]:
            raise RuntimeError("Input ({}) and output ({}) have different lengths".format(check_length, output.shape[0]))


        self.splits = None

        # Find the explicit indixes
        if splits is not None:  # We are provided with groups of indices, need to set self.splits and self.explicit_ix
            self.splits = []
            self.explicit_ix = []
            
            mask_set = frozenset(mask if mask is not None else [])
            for split in splits:
                if mask is None:
                    masked_split = split # All data is good
                else:
                    masked_split = [s for s in split if s in mask_set]  # Fold information
                self.explicit_ix += masked_split
                self.splits.append( masked_split )

            self.explicit_ix = np.array(self.explicit_ix)

        else: #No groups, need to set self.explicit_ix
            if mask is None:
                self.explicit_ix = np.arange(len(output)) # All data is good
            else:
                self.explicit_ix = mask #  Use data from the fold
            


        # Instance variables:

        self.batch_size = batch_size
        self.inputs = inputs
        self.output = output

        self.sample_probabilities = sample_probabilities
        self.implicit_samples = implicit_samples

        self.implicit_ix = None
        self.length = len(self.explicit_ix)
        self.shuffle = shuffle
        self.feature_extraction = feature_extraction
        self.nce = nce
        
        self.shuffle_indexes()



    def __len__(self):
        return math.ceil(float(self.length) / self.batch_size)

    
    def shuffle_indexes(self): 
        
        if self.splits is None:
            np.random.shuffle(self.explicit_ix)
            self.implicit_ix = np.random.randint(0, self.length, self.length * self.implicit_samples ) 
        else:
            #np.random.shuffle(self.splits) 
            shuffled_splits = []
            for s in self.splits:
                np.random.shuffle(s)
                shuffled_splits.append(s)
            
            self.explicit_ix = [item for sublist in shuffled_splits for item in sublist]   # flatten
            #self.implicit_ix = np.tile(self.explicit_ix, self.implicit_samples ) 
            self.implicit_ix = np.repeat(self.explicit_ix, self.implicit_samples ) 
        

    def on_epoch_end(self):
        if self.shuffle:
            self.shuffle_indexes()

            
    def __getitem__(self, idx):
        if (idx >= len(self)) or (idx < 0):
            raise IndexError("No such mini-batch. Index error")

        
        start_ex = idx * self.batch_size
        end_ex   = (idx + 1) * self.batch_size
        start_im = idx * self.batch_size * self.implicit_samples
        end_im   = (idx+1) * self.batch_size * self.implicit_samples
        
        
        # Get x from the mini-batch:
        mini_batch_x = []
        
        old_size = -1
        for i,l in enumerate(self.inputs):

            # Get data from the mini-batch:
            if (end_ex-start_ex) == self.batch_size:
                explicit = l[self.explicit_ix[start_ex:end_ex]]
            else:
                explicit = l[self.explicit_ix[start_ex:]]     # sparse arrays can't overflow in index

            if isinstance(explicit, scipy.sparse.csr.csr_matrix):
                explicit = explicit.toarray() # densify if it's sparse    


            noise = self.sample_probabilities.get(i, None)
            if noise is None:
                implicit = l[self.implicit_ix[start_im:end_im]] 
                if isinstance(implicit, scipy.sparse.csr.csr_matrix):
                    implicit = implicit.toarray() # densify if it's sparse 
                
            else:
                implicit = np.random.choice(noise.keys(),
                                            explicit.shape[0]*self.implicit_samples,
                                            p=noise.values())
        
            feature = np.concatenate( (explicit, implicit) )
        
        

            assert (old_size == -1) or (len(feature) == old_size), "Sizes don't match {} / {}".format(len(feature), old_size)
            old_size = feature.shape[0]

            mini_batch_x.append(feature)
        
        # Noise contrastive estimation?
        if self.nce is not None:
            q = np.array([self.sample_probabilities[self.nce][feat] for feat in mini_batch_x[self.nce] ])
            mini_batch_x.append(q)
        
        # Extract features from the minibatch
        if self.feature_extraction is not None:
            mini_batch_x = self.feature_extraction.extract_features(mini_batch_x)
        
        # Get y from the mini-batch:
        mini_batch_y = np.concatenate( (self.output[self.explicit_ix[start_ex:end_ex]],  
                                        np.zeros( explicit.shape[0] * self.implicit_samples )) )
   
        assert len(feature) == len(mini_batch_y), "Sizes don't match {} / {}".format(len(feature), mini_batch_y)            

        return mini_batch_x, mini_batch_y
