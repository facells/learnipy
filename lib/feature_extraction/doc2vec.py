def get_doc2vec(t_, o):
    import re
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from nltk.tokenize import word_tokenize
    import pandas as PD

    orig_t_ = t_
    print('apply doc2vec (will be removed in the next version) \ntheory: https://en.wikipedia.org/wiki/Word2vec#Extensions')
    r_ = re.findall(r'-x.d2v=(.+?) ', o)
    size = int(r_[0])
    fx = 1
    t_ = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(t_)]
    wmodel = Doc2Vec(vector_size=size, seed=123, workers=1)
    wmodel.build_vocab(t_)
    wmodel.train(t_, total_examples=len(t_), epochs=1)
    t_ = PD.DataFrame(wmodel.docvecs.vectors_docs)  # turn to doc2vec vectors
    print('sync dense doc2vec matrix:\n', t_) if '-d.data' in o else print(
        f'extracting {size} doc2vec features from text')

    return orig_t_, t_, fx