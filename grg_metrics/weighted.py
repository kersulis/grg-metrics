import scipy as sp
import numpy as np
import grg_metrics
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

def impedance_distance(grg_data):
    Y, buses = grg_metrics.getY(grg_data)
    Z = sp.sparse.linalg.inv(Y).todense() # completely dense!

    def z_thev_dist(Zbus):
        Zdist = np.zeros_like(Zbus)
        l = Zbus.shape[0]
        for i in range(l):
            for j in range(l):
                Zdist[i,j] = Zbus[i,i] + Zbus[j,j] - Zbus[i,j] - Zbus[j,i]
        return Zdist

    Zdist = z_thev_dist(Z)
    return np.abs(Zdist)

def mds(Zdist, seed_idx=1):
    seed = np.random.RandomState(seed=seed_idx)
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                       dissimilarity="precomputed", n_jobs=-1)
    return mds.fit(Zdist).embedding_

def algebraic_connectivity(graphs):
    Gids = [G.graph['id'] for G in graphs]
    metrics = [np.max(np.abs(np.real(nx.adjacency_spectrum(G)))) for G in graphs]
    return pd.Series(metrics, index=Gids, name='algebraic_connectivity')
