def apply_svd(svdim, x_, o):
    from sklearn import decomposition
    import pandas as PD

    svd = decomposition.TruncatedSVD(svdim, random_state=1)
    x_ = PD.DataFrame(svd.fit_transform(x_))
    print('sync dense SVD matrix from one-hot labels:\n', x_) if '-d.data' in o else print(
        'apply Singular Value Decomposition of data by default, obtain dense sync matrix')
    print('theory: https://en.wikipedia.org/wiki/Singular_value_decomposition')

    return x_