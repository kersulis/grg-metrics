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

def clustering(graphs):
    Gids = [G.graph['id'] for G in graphs]
    metrics = [np.flipud(np.sort(np.array(list(nx.clustering(G).values()))))
               for G in graphs]
    return pd.Series(metrics, index=Gids, name='node_degree_distribution')

def nesta_v11_representative():
    """Returns a representative sample of 33 NESTA GRG v1.1 networks,
    sorted by size.
    """
    return [
    'nesta_case13659_pegase',
    'nesta_case9241_pegase',
    'nesta_case6515_rte',
    'nesta_case3120sp_mp',
    'nesta_case2869_pegase',
    'nesta_case2868_rte',
    'nesta_case2736sp_mp',
    'nesta_case2383wp_mp',
    'nesta_case2224_edin',
    'nesta_case1951_rte',
    'nesta_case1397sp_eir',
    'nesta_case1354_pegase',
    'case_403_rte',
    'nesta_case300_ieee',
    'nesta_case240_wecc',
    'nesta_case189_edin',
    'nesta_case162_ieee_dtc',
    'nesta_case118_ieee',
    'nesta_case89_pegase',
    'nesta_case73_ieee_rts',
    'nesta_case57_ieee',
    'nesta_case39_epri',
    'nesta_case30_fsr',
    'nesta_case30_as',
    'nesta_case30_ieee',
    'nesta_case29_edin',
    'nesta_case24_ieee_rts',
    'nesta_case14_ieee',
    'nesta_case9_wscc',
    'nesta_case6_ww',
    'nesta_case6_c',
    'nesta_case5_pjm',
    'nesta_case4_gs',
    'nesta_case3_lmbd'
    ]
