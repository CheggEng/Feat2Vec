#explore_data.py
#explore the IMDB data, clean it up, etc.
import pandas as pd
import numpy as np
import cPickle
import gzip
import os
import matplotlib.pyplot as plt
datadir = '/home/luis/Downloads/'
datadir = '/media/luis/hdd3/Data/IMDB/'
outputdir = '/home/luis/feat2vec/paper/output/'
#load titles data

with open(os.path.join(datadir,'imdb_movie_data.p'),'r') as f:
    df= cPickle.load(f)

print df['directors'].map(len).value_counts()/np.sum(df['directors'].map(len).value_counts())

with gzip.open(datadir + 'title.basics.tsv.gz') as f:
    titles = pd.read_csv(f,sep='\t')

#throw out some titles with tabs in it.
titles['isAdult']=titles['isAdult'].astype('str')
titles = titles.loc[titles.isAdult.isin(['0','1'])==True,:]
print titles.head()

#import crew and principals data
with gzip.open(datadir + 'title.crew.tsv.gz') as f:
    crew = pd.read_csv(f,sep='\t')

print crew.head()

with gzip.open(datadir + 'title.principals.tsv.gz') as f:
    princ = pd.read_csv(f,sep='\t')

print princ.head()

with gzip.open(datadir + 'name.basics.tsv.gz') as f:
    names = pd.read_csv(f,sep='\t')

print names.head()


with gzip.open(datadir + 'title.ratings.tsv.gz') as f:
    ratings = pd.read_csv(f,sep='\t')
print ratings.head()



#plot start year of films
startyears = pd.to_numeric(titles.startYear,errors='coerce').dropna()
print startyears.describe()
plt.hist(startyears,bins=100)
plt.xlim([np.min(startyears),2017])
plt.savefig(os.path.join(outputdir,'startYear_hist.pdf'))
plt.show()


#plot endyears (TV only)
endyears = pd.to_numeric(titles.endYear,errors='coerce').dropna()
print endyears.describe()
plt.hist(endyears,bins=100)
plt.xlim([np.min(endyears),2017])
plt.savefig(os.path.join(outputdir,'endYear_hist.pdf'))
plt.show()

#plot runtime of films
minutes =  pd.to_numeric(titles.runtimeMinutes,errors='coerce').dropna()
print minutes.describe()
plt.hist(minutes,bins=np.logspace(0, np.log10(np.max(minutes)),100))
plt.xscale('log')
plt.savefig(os.path.join(outputdir,'runTime_hist.pdf'))
plt.show()

#plot ratings of films
plt.hist(pd.to_numeric(ratings.averageRating,errors='coerce').dropna(),bins=100)
plt.savefig(os.path.join(outputdir,'ratings_hist.pdf'))
plt.show()
#plot votes of films
votes = pd.to_numeric(ratings.numVotes,errors='coerce').dropna()
print votes.describe()
plt.hist(votes,bins=np.logspace(0, np.log10(np.max(votes)),100))
plt.xscale('log')
plt.savefig(os.path.join(outputdir,'numVotes_hist.pdf'))
plt.show()



###store some code here =/

#some summ stats of the data
for c in ['writers','directors','principalCast']:
    print c
    lencounts = df[c].map(lambda x:len(x))
    print lencounts.describe()
    plt.hist(lencounts,bins=100, normed=1, histtype='step',
                           cumulative=True)
    plt.xlim([0,10])
    plt.show()
    print ( lencounts <= 5 ).astype(int).mean()
