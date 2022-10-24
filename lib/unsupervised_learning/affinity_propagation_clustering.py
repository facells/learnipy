def affinity_propagation(x_, o):
    from sklearn import cluster
    import pandas as PD
    import numpy as NP

    clust = cluster.AffinityPropagation(damping=0.5).fit(x_)
    l_ = PD.DataFrame(clust.labels_)
    l_.columns = ['affinity']
    x_ = PD.concat([x_, l_], axis=1)
    g_ = x_.groupby('affinity').mean()
    print('applied affinity propagation clustering. added 1 feature')
    print('theory: https://en.wikipedia.org/wiki/Affinity_propagation')
    af = open('analysis.txt', 'a')
    af.write(g_.to_string() + "\n\n")
    af.close()
    ynp_ = l_.to_numpy()
    classes, counts = NP.unique(ynp_, return_counts=True)
    nk = len(classes)
    print(f"num clusters= {nk}")
    if '-d.viz' in o:
        from ..data_management.visualization import pca_projected
        pca_projected(x_, clust, nk, 'affinity propagation clustering')

    return x_