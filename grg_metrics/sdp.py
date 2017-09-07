import networkx as nx
import numpy as np
import scipy as sp
from sksparse.cholmod import cholesky

def chordal_extension(G):
    """Return a chordal extension of `G`.
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

def old_clique_graph_spanning_tree(cliques):
    """     T = clique_graph_spanning_tree(cliques)
    Generates a clique graph from the provided chordal
    graph, then returns a clique graph minimal spanning tree.
    Each node of a clique graph is one of the maximal cliques,
    and each edge's weight is the number of nodes shared by
    its endpoint cliques.

    Input:
    - `cliques`: a list of sets where each set consists of node
    indices for a maximal clique.

    Output:
    - `T`: a minimal spanning tree of the clique graph
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

def clique_graph_spanning_tree(cliques):
    """     T = clique_graph_spanning_tree(cliques)
    Generates a clique graph from the provided chordal
    graph, then returns a clique graph minimal spanning tree.
    Each node of a clique graph is one of the maximal cliques,
    and each edge's weight is minus the number of nodes shared by
    its endpoint cliques. The minimal spanning tree is the tree
    with lowest total edge weight, which corresponds to
    maximum clique overlap by construction.

    Input:
    - `cliques`: a list of sets where each set consists of node
    indices for a maximal clique.

    Output:
    - `T`: a minimal spanning tree of the clique graph
    """
    G = nx.Graph()
    G.add_nodes_from(cliques)

    nc = len(cliques)
    for i in range(nc):
        ci = cliques[i]
        for j in range(nc):
            cj = cliques[j]
            if i != j:
                o = len(ci & cj)
                if o > 0:
                    G.add_edge(ci, cj, weight=-o)
    return nx.prim_mst(G)

def clique_merge_cost(ci, ck):
    """     Dik = clique_merge_cost(ci, ck)
    Return cost of combining cliques ci and ck. Cost
    grows with the size of each clique, but drops with
    clique overlap (fewer linking constraints).

    Input:
    - `ci`, `ck`: cliques represented as sets of node
    indices.

    Output:
    - `Dik`: change in number of variables (primal + dual)
    correpsonding to a merge of ci and ck.
    """
    di = len(ci)
    dk = len(ck)
    sik = len(ci & ck)
    dik = len(ci | ck)

    nvars = lambda nc: nc*(2*nc + 1)
    return nvars(dik) - nvars(di) - nvars(dk) - nvars(sik)

def sdp_cost_heuristic(T):
    """     cost = sdp_cost_heuristic(T)
    Approximate SDP computational cost as sum of
    variables and linking constraints.

    Input
    - `T`: minimal spanning tree of the clique graph,
    as a NetworkX graph object.

    Output
    - `cost`: the total number of scalar variables
    (real and imag voltage scalars) and dual variables
    (linking constraints for overlapping cliques).
    """
    nvars = lambda nc: nc*(2*nc + 1)
    # variables for each clique
    vars_cost = sum([nvars(len(c[0]))
                     for c in T.nodes_iter(data=True)])
    # linking constraints
    link_cost = sum([nvars(-e[2]['weight'])
                     for e in T.edges_iter(data=True)])
    return vars_cost + link_cost

def min_cost_cliques(T):
    """    ci, ck, min_cost = min_cost_cliques(T)
    Given a clique graph minimal spanning tree, return
    the pair of cliques corresponding to the min-cost merge.

    Input:
    - `T`: minimal spanning tree of the clique graph, as a
    NetworkX graph object.

    Output:
    - `ci`, `ck`: the pair of cliques with greatest positive
    impact on SDP performance (represented by greatest
    reduction in number of primal and dual variables).
    """
    min_cost = np.inf
    ci = {}
    ck = {}
    for idx, e in enumerate(T.edges_iter()):
        i, k = e
        cost = clique_merge_cost(i, k)
        if cost < min_cost:
            ci, ck = i, k
            min_cost = cost
    return ci, ck, min_cost

def clique_merge(cliques):
    """     M = clique_merge(cliques)
    Returns linkage (used for generating dendrograms)
    corresponding to Dan's greedy clique merge algorithm.
    Also returns merge graph and other data as fields of `M`.

    Input:
    - `cliques`: list of sets, where each set contains indices
    of buses in a maximal clique.

    Output:
    - `M`: dict with fields:
        - `linkage`: linkage matrix. See documentation for
        scipy.cluster.hierarchy.linkage.
        - `Gmerge`: NetworkX merge graph for generating
        Sankey diagrams.
    """
    M = {}

    # all_cliques is used for absolute clique references
    all_cliques = cliques.copy()

    # merged_cliques shrinks as cliques are merged
    # (use this to update clique tree)
    merged_cliques = cliques.copy()

    ncliques = len(cliques)
    merge_sizes = np.ones(ncliques)

    Z = np.zeros((ncliques-1,4))
    sdp_sizes = []
    linking_constraints = []
    largest_group = []

    # directed graph for tracking merges
    Gm = nx.DiGraph()
    for i in sorted(list(frozenset().union(*cliques))):
        Gm.add_node(int(i), name='Bus ' + str(i+1),
            nodes=1,
            type='bus', xPos=0)
    nbus = int(i + 1)
    next_node_idx = nbus
    for i, c in enumerate(cliques):
        Gm.add_node(next_node_idx,
            name='Clique ' + str(next_node_idx - nbus + 1),
            nodes=len(c),
            type='clique',
            xPos=1)
        for b in c:
            Gm.add_edge(b, next_node_idx, value=1)
        next_node_idx += 1

    xPos=2
    midx = 0 # index of current merge
    dmax = 0 # largest clique size (increases with merging)
    nbus = max([max(c) for c in cliques]) + 1

    # clique merging
    while len(merged_cliques) >= 1:
        # update clique graph spanning tree
        T = clique_graph_spanning_tree(merged_cliques)

        # record data
        sdp_sizes.append(sdp_cost_heuristic(T))
        linking_constraints.append(abs(T.size(weight='weight')))
        dmax = max([len(c) for c in merged_cliques])
        largest_group.append(dmax)

        if len(merged_cliques) == 1:
            # last merge already performed, so only
            # recording was necessary.
            break

        # identify best merge
        ci, ck, min_cost = min_cost_cliques(T)
        i, k = all_cliques.index(ci), all_cliques.index(ck)

        # update linkage
        merge_size = sum(merge_sizes[[i,k]])
        merge_sizes = np.hstack((merge_sizes, merge_size))
        Z[midx,:] = [i, k, min_cost, merge_size]

        # update clique lists to reflect merge
        cmerged = ci | ck
        all_cliques.append(cmerged)
        merged_cliques.remove(ci)
        merged_cliques.remove(ck)
        merged_cliques.append(cmerged)

        # update clique merge graph to reflect merge
        merge_idx = next_node_idx - ncliques - nbus + 1
        Gm.add_node(next_node_idx,
            name='Merge %s' % str(merge_idx),
            nodes=len(cmerged),
            cost=min_cost,
            type='merge',
            xPos=xPos)
        xPos += 1
        Gm.add_edge(i + nbus, next_node_idx, value=len(ci))
        Gm.add_edge(k + nbus, next_node_idx, value=len(ck))
        next_node_idx += 1
        midx += 1

    M['linkage'] = Z
    M['Gmerge'] = Gm
    M['sdp_sizes'] = sdp_sizes
    M['linking_constraints'] = linking_constraints
    M['largest_group'] = largest_group
    return M
