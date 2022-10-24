def get_word2vec(t_, o):
    import re
    import datetime as DT
    import sys
    from sklearn import decomposition
    import matplotlib.pyplot as MP
    MP.rcParams["figure.figsize"] = (5, 4)
    import gensim as W2V

    if '-u.w2v=' in o:
        r_ = re.findall(r'-u.w2v=(.+?) ', o)
        fw = int(r_[0][0])
        nw = int(r_[0][1])
        fw = fw * 10
        nw = nw * 10
    else:
        nw = 20
    print(f'apply word2vec, extract dictionary of most freq. words from rank {fw} to {nw}\ntheory: https://en.wikipedia.org/wiki/Word2vec')
    t_ = t_.str.split(pat=" ")
    wmodel = W2V.models.Word2Vec(t_, min_count=2)
    words = list(wmodel.wv.vocab)
    wmodel.wv.save_word2vec_format('w2v.txt', binary=False)
    wmodel.save('w2v.bin')  # save word2vec dictionary
    X = wmodel[wmodel.wv.index2entity[fw:nw]]
    pca = decomposition.PCA(n_components=2)
    result = pca.fit_transform(X)
    MP.scatter(result[:, 0], result[:, 1])
    words_ = list(wmodel.wv.index2entity[fw:nw])
    # print(words_)  # fit a 2d PCA model to the w2v vectors
    [MP.annotate(word, xy=(result[i, 0], result[i, 1])) for i, word in enumerate(words_)]
    MP.title('w2v 2d space')
    MP.savefig(fname='w2v-space')
    MP.show()
    MP.clf()  # visualize w2v-space and save it
    print('extracted word2vec dictionary from text. save w2v.txt, w2v.bin and w2v-space.png')
    timestamp = DT.datetime.now()
    print(f"-u.w2v stops other tasks\ntime:{timestamp}")
    sys.exit()