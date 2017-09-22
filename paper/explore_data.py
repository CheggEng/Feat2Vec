#explore_data.py
#explore the IMDB data, clean it up, etc.
import pandas as pd
import numpy as np
import gzip
import matplotlib.pyplot as plt
datadir = '/home/luis/Downloads/'
#load data
with gzip.open(datadir + 'title.basics.tsv.gz') as f:
    df = pd.read_csv(f,sep='\t')

#throw out some titles with tabs in it.
df['isAdult']=df['isAdult'].astype('str')
df = df.loc[df.isAdult.isin(['0','1'])==True,:]

#create genre matrix
genstrMat = df['genres'].str.split(',',expand=True)
genstrMat = np.array(genstrMat.fillna(r'\N'))
genre_types = np.unique(genstrMat).tolist()
for g in genre_types:
    df[r'genre_{}'.format(g)] = np.any(genstrMat==g,axis=1)
    df[r'genre_{}'.format(g)] = df[r'genre_{}'.format(g)].astype('int8')




#import crew and principals data
with gzip.open(datadir + 'title.crew.tsv.gz') as f:
    crew = pd.read_csv(f,sep='\t')
print crew['directors'].str.split(',',expand=True).head()
print crew.head()

df = df.join(crew.set_index('tconst'),on='tconst')

with gzip.open(datadir + 'title.principals.tsv.gz') as f:
    princ = pd.read_csv(f,sep='\t')
print princ.head()
df = df.join(princ.set_index('tconst'),on='tconst')

#princ = df['principalCast'].str.split(',',expand=True)
print princ.head()


with gzip.open(datadir + 'title.ratings.tsv.gz') as f:
    ratings = pd.read_csv(f,sep='\t')
print ratings.head()
df = df.join(ratings.set_index('tconst'),on='tconst')

for c in ['titleType','isAdult','startYear']:
    print df[c].value_counts()

df.isAdult.value_counts()


#plot start year of films
plt.hist(pd.to_numeric(df.startYear,errors='coerce').dropna(),bins=100)
plt.show()
