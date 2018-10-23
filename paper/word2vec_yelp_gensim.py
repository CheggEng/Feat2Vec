
#########################
####
#w2v in gensim


class YelpW2V(object):
    def __init__(self,doclist):
        self.doclist = doclist
    def __iter__(self):
        for d in self.doclist:
            yield d.rstrip('\n').split(' ')


iterator = YelpW2V(full_docs)
#make docs
int_docs = []
for i,d in enumerate(full_docs):
    if i % 10000==0:
        sys.stdout.write('\r %g' % i)
        sys.stdout.flush()
    int_seq = [word_dict[w] for w in d.rstrip('\n').split() if word_counts[w] >=5]
    int_docs.append(int_seq)


#glovevecs = np.zeros(shape=(max_vocab,dim))
numwords =0
with open(glovefile,'r') as f:
    for line in f:
        splitLine = line.split()
        word = splitLine[0].lower()
        if word in vocab_map['textseq']:
            idx = vocab_map['textseq'][word]
            if idx < max_vocab:
                numwords+=1
                initvecdict['textseq_%g' % idx] = np.array([float(val) for val in splitLine[1:]])
for i in range(max_vocab):
    key = 'textseq_%g' % i
    if key not in initvecdict:
        initvecdict[key] = np.zeros(dim)



from gensim.scripts.glove2word2vec import glove2word2vec

glovefile =os.path.join(glovedir,'glove.6B.{d}d.fmt.txt'.format(d=dim))

glove2word2vec('/media/luis/hdd3/Data/WordEmbeddings/glove.6B.100d.txt',glovefile)

#opt_epoch = np.argmin(dev_losses)
for skipgram in [False,True]:
    opt_epoch = 3
    print skipgram
    print "Now Completing full training on full corpus ({} epochs)...".format(opt_epoch)
    w2v = Word2Vec(size=dim, window=window,
     min_count=0, max_vocab_size=None, sample=0.,
     seed=9, workers=8,
     alpha=0.025,min_alpha=0.0001,
     sg=skipgram, hs=0, negative=negative_samples, cbow_mean=0,
     iter=opt_epoch, sorted_vocab=1, batch_words=1000, compute_loss=False)
    print "building vocab"
    w2v.build_vocab(iterator)
    w2v.intersect_word2vec_format(glovefile, binary=False, lockf=1.0)
    for i in range(opt_epoch):
        print "epoch %g" % i
        w2v.train(iterator,epochs=1,total_examples=len(full_docs))
    print "exporting embeddings..."
    if skipgram:
        w2v.save(os.path.join(yelpdir,'w2v/w2v_gensim_model_sg'))
        w2v.wv.save_word2vec_format(os.path.join(yelpdir,'w2v/w2v_vectors_sg.txt'))
    else:
        w2v.save(os.path.join(yelpdir,'w2v/w2v_gensim_model_cbow'))
        w2v.wv.save_word2vec_format(os.path.join(yelpdir,'w2v/w2v_vectors_cbow.txt'))



#############
#doc2vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#add tags to
class YelpD2V(object):
    def __init__(self,doclist):
        self.doclist = doclist
    def __iter__(self):
        for d in self.doclist:
            tokens = d.rstrip('\n').split(' ')
            tags = tokens[:4]
            words = tokens[4:]
            yield TaggedDocument(words,tags)
dociter = YelpD2V(full_docs)
d2v = Doc2Vec(documents=dociter,  dm=True, dm_mean = True, vector_size=dim,
         docvecs=None, docvecs_mapfile=None, comment=None,workers=8,
         negative=5,seed=9,iter=opt_epoch)

d2v.wv.save_word2vec_format(os.path.join(yelpdir,'w2v/d2v_word_vectors.txt'))
d2v.docvecs.save_word2vec_format(os.path.join(yelpdir,'w2v/d2v_word_vectors.txt'),prefix='')

d2v.save(os.path.join(yelpdir,'w2v/d2v_gensim_model'))
