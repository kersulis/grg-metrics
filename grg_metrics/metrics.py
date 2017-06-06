import networkx as nx

def node_degree_distribution(graphs, metrics):
    """Add degree distribution to metrics.
    (Modifies input `metrics`.)
    """
    for G in graphs:
        Gid = G.graph['id']
        if Gid not in metrics:
            metrics[Gid] = {}
        metrics[Gid]['node_degrees'] = list(nx.degree(G).values())
    return metrics

def degree_assortativity(graphs, metrics):
    """Add degree assortativity coefficient to metrics.
    (Modifies input `metrics`.)
    """
    for G in graphs:
        Gid = G.graph['id']
        if Gid not in metrics:
            metrics[Gid] = {}
        metrics[Gid]['degree_assortativity'] = nx.degree_assortativity_coefficient(G)
    return metrics

def load_centrality(graphs, metrics):
    """Add load centrality to metrics.
    (Modifies input `metrics`.)
    Takes a little while to run.
    """
    for G in graphs:
        Gid = G.graph['id']
        if Gid not in metrics:
            metrics[Gid] = {}
        metrics[Gid]['load_centrality'] = nx.load_centrality(G)
    return metrics

def degree_centrality(graphs, metrics):
    """Add degree centrality to metrics.
    (Modifies input `metrics`.)
    Takes a little while to run.
    """
    for G in graphs:
        Gid = G.graph['id']
        if Gid not in metrics:
            metrics[Gid] = {}
        metrics[Gid]['degree_centrality'] = nx.degree_centrality(G)
    return metrics
