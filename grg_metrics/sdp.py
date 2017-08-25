import networkx as nx
import numpy as np
import scipy as sp
from sksparse.cholmod import cholesky

def chordal_extension(G):
    """Return a chordan extension of G.
    Input and output are both NetworkX graphs.
    """
    adj = nx.adjacency_matrix(G)
    n = adj.shape[0]
    diag = sum(adj, 0).todense() + 1
    W = (adj + sp.sparse.spdiags(diag, 0, n, n)).tocsc()

    # obtain chordal extension of network
    # via Cholesky decomposition
    F = cholesky(W)
    Rchol = F.L().todense()
    np.fill_diagonal(Rchol, 0)

    # build adjacency matrix of chordal extension
    f, t = np.nonzero(Rchol)
    Aadj = sp.sparse.csc_matrix((np.ones_like(f), (f,t)), shape=(n, n)).todense()
    Aadj = np.maximum(Aadj, Aadj.T)
    Gchord = nx.from_numpy_matrix(Aadj)
    return Gchord
