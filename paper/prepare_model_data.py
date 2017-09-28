
import pandas as pd
import numpy as np
import cPickle
import os
datadir = '/home/luis/Downloads/'
datadir = '/media/luis/hdd3/Data/IMDB/'

with open(os.path.join(datadir,'imdb_title_data.p'),'r') as f:
    df = cPickle.load(f)
print df.shape
