def self_organizing_map(x_, o, feat):
    import os
    import math
    import pandas as PD
    import numpy as NP
    os.system('pip install minisom')
    from minisom import MiniSom  # library

    xn_ = x_.to_numpy()  # change dataset x_ format to numpy to  use minisom
    nodes = int(4 * math.sqrt(len(xn_)))  # compute nodes of the matrix based on instances
    clust = MiniSom(nodes, nodes, feat, sigma=0.3, learning_rate=0.5, random_seed=2)  # initialize SOM
    clust.train(xn_, 100)  # train SOM with 100 iterations
    l_ = []
    col_ = []  # new column with som results
    for i in xn_:
        w = clust.winner(i)
        l = (w[0] + w[1]) / (nodes * 2)
        l_.append(l)  # compute som results, put in l_
        col = int(w[0] + w[1])
        col_.append(col)  # compute int for colors, put in col_
    l_ = PD.DataFrame(l_)
    l_.columns = ['som']
    x_ = PD.concat([x_, l_], axis=1)
    # print(l_)
    g_ = x_.groupby('som').mean()
    print(f'apply self organizing maps clustering with {nodes} x {nodes} matrix. added 1 feature')
    print('theory: https://en.wikipedia.org/wiki/Self-organizing_map')
    af = open('analysis.txt', 'a')
    af.write(g_.to_string() + "\n\n")
    af.close()
    ynp_ = l_.to_numpy()
    classes, counts = NP.unique(ynp_, return_counts=True)
    nk = len(classes)
    print(f"num clusters= {nk}")
    if '-d.viz' in o:
        from ..data_management.visualization import pca_projected
        pca_projected(x_, clust, nk, 'self organizing map')

    return x_
