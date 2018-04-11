#plot_ranks.py
#make pretty plots for the paper
import numpy  as np
import pandas as pd
import sys,os
import cPickle
import matplotlib.pyplot as plt
import feat2vec
datadir = '/home/luis/Data/IMDB/'
datadir = '/media/luis/hdd3/Data/IMDB/'
#datadir = ''
outputdir= 'iclr/'

# with open(os.path.join(datadir,'w2v_test_ranks_cbow.p'),'r') as f:
#     w2v_ranks_cbow=cPickle.load(f)
# with open(os.path.join(datadir,'w2v_test_ranks_sg.p'),'r') as f:
#     w2v_ranks_sg=cPickle.load(f)
with open(os.path.join(datadir,'w2v_test_ranks.p'),'r') as f:
     w2v_ranks=cPickle.load(f)
with open(os.path.join(datadir,'alpha_75_75/f2v_test_ranks.p'),'r') as f:
    f2v_ranks_25=cPickle.load(f)

maxrank =  173620
print maxrank
np.max(f2v_ranks_25)
#bins = [r/1000. for r in range(maxrank)]
#plt.hist([r/1000. for r in w2v_ranks],alpha=1.,label='W2V',bins=bins,cumulative=True,normed=1,histtype='step')
#plt.hist([r/1000 for r in f2v_ranks_25],alpha=1.,label='F2V',bins=bins,cumulative=True,normed=1,histtype='step')
#plt.xlim([0, maxrank/1000.-1/1000.])
#plt.xlabel('Rankings (1000s)')
bins = range(maxrank)
plt.hist(f2v_ranks_25,alpha=1.,label='Feat2Vec',bins=bins,cumulative=True,normed=1,histtype='step')
plt.hist(w2v_ranks,alpha=1.,label='CBOW',bins=bins,cumulative=True,normed=1,histtype='step')
#plt.hist(w2v_ranks_sg,alpha=1.,label='Skipgram',bins=bins,cumulative=True,normed=1,histtype='step')
plt.xlim([0, maxrank])
plt.xlabel('Director Ranking')
plt.ylabel('CDF')
plt.xticks(np.arange(0,maxrank,30000))
plt.legend(loc=2)
plt.savefig(os.path.join(outputdir,'rankcdf.pdf'))
plt.show()


plt.hist(f2v_ranks_25,alpha=1.,label='Feat2Vec',bins=range(maxrank),cumulative=True,normed=1,histtype='step')
plt.hist(w2v_ranks,alpha=1.,label='CBOW',bins=range(maxrank),cumulative=True,normed=1,histtype='step')
#plt.hist(w2v_ranks_sg,alpha=1.,label='Skipgram',bins=range(maxrank),cumulative=True,normed=1,histtype='step')
plt.xlim([0, 25])
plt.ylim([0.,.1])
#plt.xlabel('Rankings')
#plt.ylabel('CDF')
#plt.legend(loc=2)
plt.savefig(os.path.join(outputdir,'rankcdf_top25.pdf'),transparent=True)
plt.show()

#hings to plot: diff CDF , Rankings by sparsity
