#create_data.py
#merge together IMDB title-level data
import pandas as pd
import numpy as np
import gzip
import re
import nltk
import cPickle
import os
import matplotlib.pyplot as plt
nltk.download('stopwords')
sys.path.append('feat2vec/')
datadir=''
movies_only = True
#load data
with gzip.open(datadir + 'title.basics.tsv.gz') as f:
    df = pd.read_csv(f,sep='\t')

#throw out some titles with tabs in it ,IDd by distorted isAdult values.
df['isAdult']=df['isAdult'].astype('str')
print "Throwing out {} titles due to malformed titles".format(np.sum(df.isAdult.isin(['0','1'])==False))
df = df.loc[df.isAdult.isin(['0','1'])==True,:]
df['isAdult'] = df['isAdult'].astype('int8')
print df.head()
if movies_only:
    print df.shape
    df = df.loc[df['titleType']=='movie',:]
    print df.shape
#fill Nans with null identifier
#unneccessary since data is already "clean"
for c in ['titleType','isAdult','startYear','endYear','runtimeMinutes']:
    print c
    vcounts = df[c].value_counts(dropna=False)
    print len(vcounts)
    print vcounts

if movies_only:
    df.drop('titleType',axis=1,inplace=True)

#Clean movie titles to sequence of words
print "Cleaning title text... "
texts = df['primaryTitle'].map(lambda x: (" ".join(re.findall('[A-Za-z]+',x))).encode('utf8')) #clean icky characters out
texts = texts.map(lambda x: x.lower().split(" ")) #lowercase
#remove stopwords
stopWords = set(nltk.corpus.stopwords.words('english'))
texts = texts.map(lambda x: [w for w in x if w not in stopWords])
#stem words
stemmer = nltk.stem.snowball.SnowballStemmer("english")
texts = texts.map(lambda x: [stemmer.stem(w) for w in x])
print df['primaryTitle'].tail(15)
df['titleSeq'] = texts
df.loc[df['titleSeq']==r'\N','titleSeq'] = df.loc[df['titleSeq']==r'\N','titleSeq'].map(lambda x: [r'\N'])
print df['titleSeq'].head(15)


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

#convert sequence strings to lists
print "list-ifying sequence vars"
for c in ['directors','writers','principalCast','genres']:
    print c
    df[c] = df[c].fillna(r'\N')
    df[c] = df[c].str.split(',',expand=False)
    print df[c].map(lambda x: len(x)).value_counts()



#save a separate test dataset consisting of directors with multiple obs
#that we will use to predict OOS matches
print "Splitting test and train data for later evaluation..."
testsize = int(len(df)*.1)
np.random.seed(9)
df = df.sample(frac=1.)
df['firstDirector'] = df['directors'].map(lambda x: x[0])
df['dirOccurences'] = 1
df.loc[df['firstDirector']==r'\N','dirOccurences'] = 0
df['dirOccurences'] = df.groupby('firstDirector')['dirOccurences'].transform('cumsum')
print df.dirOccurences.value_counts().head()
#df = df.sort_values('firstDirector')
#print df
test_index = np.random.choice(df[df['dirOccurences']>1].index,size=testsize,replace=False)
train_index = [x for x in df.index if x not in test_index]
df.drop(['dirOccurences','firstDirector'],axis=1,inplace=True)

testdf = df.loc[test_index,:]
traindf= df.loc[train_index,:]

print testdf.head()
print traindf.head()
print len(df),len(testdf)+len(traindf)

print df.shape
print df.head()
print "saving to file"
if movies_only:
    with open(os.path.join(datadir,'imdb_movie_data.p'),'w') as f:
        cPickle.dump(df,f)
    with open(os.path.join(datadir,'imdb_train_movie_data.p'),'w') as f:
        cPickle.dump(traindf,f)
    with open(os.path.join(datadir,'imdb_test_movie_data.p'),'w') as f:
        cPickle.dump(testdf,f)
else:
    with open(os.path.join(datadir,'imdb_title_data.p'),'w') as f:
        cPickle.dump(df,f)

print "Done!"
