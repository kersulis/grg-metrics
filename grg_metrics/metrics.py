import networkx as nx
import pandas as pd
import numpy as np

def node_degree_distribution(graphs):
    Gids = [G.graph['id'] for G in graphs]
    metrics = [np.flipud(np.sort(np.array(list(nx.degree(G).values()))))
               for G in graphs]
    return pd.Series(metrics, index=Gids, name='node_degree_distribution')

def degree_assortativity(graphs):
    Gids = [G.graph['id'] for G in graphs]
    metrics = [nx.degree_assortativity_coefficient(G) for G in graphs]
    return pd.Series(metrics, index=Gids, name='degree_assortativity')

def rich_club(graphs):
    Gids = [G.graph['id'] for G in graphs]
    metrics = [nx.rich_club_coefficient(G, normalized=False) for G in graphs]
    return pd.Series(metrics, index=Gids, name='rich_club')

def load_centrality(graphs):
    Gids = [G.graph['id'] for G in graphs]
    metrics = [nx.load_centrality(G) for G in graphs]
    return pd.Series(metrics, index=Gids, name='load_centrality')
