#create_data.py
#merge together IMDB title-level data
import pandas as pd
import numpy as np
import gzip
import cPickle
import os
import matplotlib.pyplot as plt
#datadir = '/home/luis/Data/IMDB/'
datadir = '/media/luis/hdd3/Data/IMDB/'
movies_only = True
#load data
with gzip.open(datadir + 'title.basics.tsv.gz') as f:
    df = pd.read_csv(f,sep='\t')

#throw out some titles with tabs in it.
df['isAdult']=df['isAdult'].astype('str')
df = df.loc[df.isAdult.isin(['0','1'])==True,:]
df['isAdult'] = df['isAdult'].astype('int8')
print df.head()
if movies_only:
    print df.shape
    df = df.loc[df['titleType']=='movie',:]
    print df.shape
#fill Nans with null identifier
for c in ['titleType','isAdult','startYear','endYear','runtimeMinutes']:
    print c
    vcounts = df[c].value_counts(dropna=False)
    print len(vcounts)
    print vcounts


#create genre matrix
#print "creating genre matrix"
#genstrMat = df['genres'].str.split(',',expand=True)
#genstrMat = np.array(genstrMat.fillna(r'\N'))
#genre_types = np.unique(genstrMat).tolist()
#for g in genre_types:
#    df[r'genre_{}'.format(g)] = np.any(genstrMat==g,axis=1)
#    df[r'genre_{}'.format(g)] = df[r'genre_{}'.format(g)].astype('bool')



#import crew and principals data
print "importing crew data"
with gzip.open(datadir + 'title.crew.tsv.gz') as f:
    crew = pd.read_csv(f,sep='\t')

print crew.head()
df = df.join(crew.set_index('tconst'),on='tconst')

print "importing principal cast data"
with gzip.open(datadir + 'title.principals.tsv.gz') as f:
    princ = pd.read_csv(f,sep='\t')

print princ.head()
df = df.join(princ.set_index('tconst'),on='tconst')

print "importing ratings data"
with gzip.open(datadir + 'title.ratings.tsv.gz') as f:
    ratings = pd.read_csv(f,sep='\t')
print ratings.head()
df = df.join(ratings.set_index('tconst'),on='tconst')
df['mi_rating'] = df['averageRating'].isnull().astype('int8')
df.loc[df['mi_rating']==1,'averageRating'] = 0
df.loc[df['mi_rating']==1,'numVotes'] = 0

#convert strings to lists
print "list-ifying sequence vars"
for c in ['directors','writers','principalCast','genres']:
    print c
    df[c] = df[c].fillna(r'\N')
    df[c] = df[c].str.split(',',expand=False)
    print df[c].map(lambda x: len(x)).value_counts()


print df.shape
print df.head()
print "saving to file"
if movies_only:
    with open(os.path.join(datadir,'imdb_movie_data.p'),'w') as f:
        cPickle.dump(df,f)
else:
    with open(os.path.join(datadir,'imdb_title_data.p'),'w') as f:
        cPickle.dump(df,f)

print "Done!"
