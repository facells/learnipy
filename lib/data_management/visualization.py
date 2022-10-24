def pca_projected(x_, clust, nk, algo):
    import sklearn as SK
    import pandas as PD
    import matplotlib.pyplot as MP
    MP.rcParams["figure.figsize"] = (5, 4)

    pca = SK.decomposition.PCA(2)
    projected = pca.fit_transform(x_)
    MP.scatter(projected[:, 0], projected[:, 1], c=PD.DataFrame(clust.labels_), edgecolor='none', alpha=0.8,
               cmap=MP.cm.get_cmap('brg', nk))
    MP.xlabel('component 1')
    MP.ylabel('component 2')
    MP.colorbar()
    MP.title(f'2D PCA data space {algo}')
    MP.savefig(fname='pca-cluster-space.png')
    MP.show()
    MP.clf()  # pca space

def feature_distribution(x_):
    for col in x_:
        mu = x_[col].mean()
        mi = x_[col].min()
        ma = x_[col].max()
        me = x_[col].median()
        sd = x_[col].std()
        print(f"min={mi:.1f} max={ma:.1f} avg={mu:.3f} sd={sd:.3f} med={me:.3f}  distribution of feature {col}")
