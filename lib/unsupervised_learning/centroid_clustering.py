def kmeans(x_, o):
    import re
    from sklearn import cluster
    import pandas as PD

    r_ = re.findall(r'-u.km=(.+?) ', o)
    nk = int(r_[0])
    clust = cluster.KMeans(n_clusters=nk, random_state=0).fit(x_)
    l_ = PD.DataFrame(clust.labels_)
    l_.columns = ['kmeans']
    x_ = PD.concat([x_, l_], axis=1)
    g_ = x_.groupby('kmeans').mean()
    print('applied kmeans clustering. added 1 feature')
    print('theory: https://en.wikipedia.org/wiki/K-means_clustering')
    af = open('analysis.txt', 'a')
    af.write(g_.to_string() + "\n\n")
    af.close()
    print(f"num clusters= {nk}")
    if '-d.viz' in o:
        from ..data_management.visualization import pca_projected
        pca_projected(x_, clust, nk, 'kmeans clustering')