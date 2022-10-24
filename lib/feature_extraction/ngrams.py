def ng(t_, o, lsadim): #TODO oppure svdim ma sono uguali
    import re
    from sklearn import feature_extraction
    import pandas as PD

    orig_t_ = t_
    r_ = re.findall(r'-x.ng=(.+?) ', o)
    mi = int(r_[0][0])
    ma = int(r_[0][1])
    ty = (r_[0][2])
    mo = (r_[0][3])
    if len(r_[0]) == 5:
        mxf = int(r_[0][4]) * 100
    else:
        mxf = 1000
    if ty == 'c' and mo == 'f':
        w = feature_extraction.text.CountVectorizer(ngram_range=(mi, ma), analyzer='char_wb', max_features=mxf)
        wv_ = w.fit_transform(t_)
        fx = 1
        fn_ = []
        [fn_.append(i) for i in w.get_feature_names()]
        t_ = PD.DataFrame(wv_.toarray(), columns=fn_)
        print('async sparse char ngram matrix:\n', t_) if '-d.data' in o else print(
            f'extract {mi}-{ma} char ngram from text')
        print(fn_)
    if ty == 'w' and mo == 'f':
        w = feature_extraction.text.CountVectorizer(ngram_range=(mi, ma), analyzer='word', max_features=mxf)
        wv_ = w.fit_transform(t_)
        fx = 1
        fn_ = []
        [fn_.append(i) for i in w.get_feature_names()]
        t_ = PD.DataFrame(wv_.toarray(), columns=fn_)
        print('async sparse word ngram matrix:\n', t_) if '-d.data' in o else print(
            f'extract word {mi}-{ma}gram from text')
        print(fn_)
    if ty == 'c' and mo == 't':
        w = feature_extraction.text.TfidfVectorizer(ngram_range=(mi, ma), analyzer='char_wb', max_features=mxf)
        wv_ = w.fit_transform(t_)
        fx = 1
        fn_ = []
        [fn_.append(i) for i in w.get_feature_names()]
        t_ = PD.DataFrame(wv_.toarray(), columns=fn_)
        print('async sparse char ngram matrix:\n', t_) if '-d.data' in o else print(
            f'extract tf-idf {mi}-{ma} char ngram from text')
        print(fn_)
    if ty == 'w' and mo == 't':
        w = feature_extraction.text.TfidfVectorizer(ngram_range=(mi, ma), analyzer='word', max_features=mxf)
        wv_ = w.fit_transform(t_)
        fx = 1
        fn_ = []
        [fn_.append(i) for i in w.get_feature_names()]
        t_ = PD.DataFrame(wv_.toarray(), columns=fn_)
        print('async sparse word ngram matrix:\n', t_) if '-d.data' in o else print(
            f'extract tf-idf word {mi}-{ma}grams from text')
        print(fn_)


    # lsadim=int(len(fn_)/2)
    #if not '-d.r=0' in o:
    #    svd = SK.decomposition.TruncatedSVD(lsadim, random_state=1)
    #    t_ = PD.DataFrame(svd.fit_transform(t_))

    if not '-d.r=0' in o:
        from ..feature_reduction import singular_value_decomposition
        t_ = singular_value_decomposition.apply_svd(lsadim, t_, o)
        print('sync dense LSA ngram matrix:\n', t_) if '-d.data' in o else print(
            'apply LSA to ngrams by default, obtain dense sync matrix')
        print('theory: https://en.wikipedia.org/wiki/Latent_semantic_analysis')


    if '-d.save ' in o:
        print(f"WARNING: the test set must contain at least {lsadim} instances for compatibility with the model")

    return orig_t_, t_, fx