def optics(x_, o):
    from sklearn import cluster
    import pandas as PD
    import numpy as NP

    clust = cluster.OPTICS(min_cluster_size=None).fit(x_)
    l_ = PD.DataFrame(clust.labels_)
    l_.columns = ['optics']
    x_ = PD.concat([x_, l_], axis=1)
    g_ = x_.groupby('optics').mean()
    print('applied optics clustering. added 1 feature')
    print('theory: https://en.wikipedia.org/wiki/OPTICS_algorithm')
    af = open('analysis.txt', 'a')
    af.write(g_.to_string() + "\n\n")
    af.close()
    ynp_ = l_.to_numpy()
    classes, counts = NP.unique(ynp_, return_counts=True)
    nk = len(classes)
    print(f"num clusters= {nk}")
    if '-d.viz' in o:
        from ..data_management.visualization import pca_projected
        pca_projected(x_, clust, nk, 'optics')


def mean_shift(x_, o):
    from sklearn import cluster
    import pandas as PD
    import numpy as NP

    clust = cluster.MeanShift().fit(x_)
    l_ = PD.DataFrame(clust.labels_)
    l_.columns = ['mshift']
    x_ = PD.concat([x_, l_], axis=1)
    g_ = x_.groupby('mshift').mean()
    print('applied mshift clustering. added 1 feature')
    print('theory: https://en.wikipedia.org/wiki/Mean_shift')
    af = open('analysis.txt', 'a')
    af.write(g_.to_string() + "\n\n")
    af.close()
    ynp_ = l_.to_numpy()
    classes, counts = NP.unique(ynp_, return_counts=True)
    nk = len(classes)
    print(f"num clusters= {nk}")
    if '-d.viz' in o:
        from ..data_management.visualization import pca_projected
        pca_projected(x_, clust, nk, 'mean shift')
