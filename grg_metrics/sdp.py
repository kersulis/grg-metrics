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

def clique_graph_spanning_tree(cliques):
    """Generates a clique graph from the provided chordal
    graph, then returns a clique graph minimal spanning tree.
    """
    n = max([max(c) for c in cliques]) + 1

    # each column of MC represents a clique
    # with 1s in appropriate bus indices
    MC = np.zeros((n,len(cliques)))
    for i, c in enumerate(cliques):
        MC[list(c),i] = 1

    # cliqueOverlap[i,j] (for i != j) is minus the number of nodes
    # shared by cliques i and j
    cliqueOverlap = -MC.T.dot(MC)
    np.fill_diagonal(cliqueOverlap, 0)

    # cliqueOverlap is a clique adjacency matrix. Gclique is the
    # corresponding NetworkX graph
    Gclique = nx.from_numpy_matrix(cliqueOverlap)

    # embed clique node info in clique graph object
    for i, c in enumerate(cliques):
        Gclique.node[i] = c

    # Minimum spanning tree, computed via Prim's algorithm.
    # The mst is the clique graph spanning tree (so it includes
    # all cliques) with lowest possible edge weights.
    # By construction, this corresponds to greatest node overlap.
    return nx.prim_mst(Gclique)

def merge_cost(ci, ck):
    """Return cost of combining cliques i and j into
    a new clique, ck. Cost grows with the size of each
    clique, but drops with clique overlap (fewer linking
    constraints).

    Inputs are cliques, represented by sets of node
    indices.
    """
    di = len(ci)
    dk = len(ck)
    sik = len(ci & ck)
    dik = len(ci | ck)

    nvars = lambda nc: nc*(2*nc + 1)
    return nvars(dik) - nvars(di) - nvars(dk) - nvars(sik)

def sdp_cost(Gmst):
    """Approximate computational cost as sum of
    variables and linking constraints.
    """
    nvars = lambda nc: nc*(2*nc + 1)
    # variables for each clique
    vars_cost = sum([nvars(len(c[1]))
                     for c in Gmst.nodes_iter(data=True)])
    # linking constraints
    link_cost = sum([nvars(-e[2]['weight'])
                     for e in Gmst.edges_iter(data=True)])
    return vars_cost + link_cost
