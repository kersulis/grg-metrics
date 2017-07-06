import networkx as nx
import pandas as pd
import numpy as np
import grg_metrics

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

def compute_metrics(x):
    """
        metrics = compute_metrics(dir_path)
        metrics = compute_metrics(list_of_file_paths)
        metrics = compute_metrics(list_of_networkx_graphs)
    Return a DataFrame with metric data.
    """
    if isinstance(x, str):
        # assume input is directory
        files = grg_metrics.find_files(x)
        graphs = []
        for file_name in files:
            graphs.append(grg_metrics.grg2nx(grg_metrics.parse_grg_case_file(file_name)))
    elif isinstance(x, list):
        if isinstance(x[0], str) and (x[0][-5:] == '.json'):
            # assume list of file paths
            graphs = []
            for file_name in x:
                data = grg_metrics.io.parse_grg_case_file(file_name)
                graphs.append(grg_metrics.grg2nx(data))
        elif isinstance(x[0], nx.Graph):
            # assume list of graph objects
            graphs = x
        else:
            print('A list input must consist of file paths or networkx graphs.')
            return []
    else:
        print('Input should be a directory path, list of file paths, or list of networkx graphs.')
        return []

    Gids = [G.graph['id'] for G in graphs]
    metrics = pd.DataFrame(pd.Series(graphs, index=Gids, name='graph'))
    metrics['nodes'] = [len(G.nodes()) for G in metrics.graph]
    metrics['edges'] = [len(G.edges()) for G in metrics.graph]
    bins = [0, 20, 1000, 5000, np.inf]
    labels = ['tiny', 'small', 'medium', 'large']
    size_groups = pd.cut(metrics.nodes, bins, labels=labels)
    metrics['size'] = size_groups
    metrics['node_degree_distribution'] = node_degree_distribution(graphs)
    metrics['max_degree'] = metrics['node_degree_distribution'].apply(max)
    metrics['mean_degree'] = metrics['node_degree_distribution'].apply(np.mean)
    metrics['median_degree'] = metrics['node_degree_distribution'].apply(np.median)
    metrics['degree_assortativity'] = degree_assortativity(graphs)
    metrics['rich_club'] = rich_club(graphs)
    metrics['clustering'] = clustering(graphs)
    return metrics

def check_max_degree(metrics):
    """Warning: max. degree greater than 10.
    Error: max. degree greater than 3.7*log10(x) + 3.4.

    Input `metrics` must have columns 'max_degree' and 'nodes'.
    """
    error = metrics.max_degree > 3.7*np.log10(metrics.nodes) + 3.4
    warning = (metrics.max_degree > 10) & ~error

    msg = pd.Series(['error' if error[i] else ('warning' if warning[i] else '')
        for i in range(len(error))], metrics.index)
    return msg

def check_mean_degree(metrics):
    """Warning: mean degree above 3.0.
    Error: mean degree above 4.0.
    """
    error = metrics.mean_degree > 4
    warning = (metrics.mean_degree > 3) & ~error
    msg = pd.Series(['error' if error[i] else ('warning' if warning[i] else '')
        for i in range(len(error))], metrics.index)
    return msg

def check_median_degree(metrics):
    """Warning: median degree = 3 and nodes > 200.
    Error: median degree > 3.
    """
    error = metrics.median_degree > 3
    warning = (metrics.median_degree == 3) & (metrics.nodes > 200) & ~error
    msg = pd.Series(['error' if error[i] else ('warning' if warning[i] else '')
        for i in range(len(error))], metrics.index)
    return msg

def check_degree_assortativity(metrics):
    """Warning: outside [-0.3, 0.15].
    Error: outside [-0.5, 0.3].
    """
    error = (metrics.degree_assortativity < -0.5) | (metrics.degree_assortativity > 0.3)
    warning = (metrics.degree_assortativity < -0.3) | (metrics.degree_assortativity > 0.15) & ~error
    msg = pd.Series(['error' if error[i] else ('warning' if warning[i] else '')
        for i in range(len(error))], metrics.index)
    return msg

def check_rich_club(metrics):
    """Warning: let K_0.8 be the set of degrees with
    rich club coefficients >= 0.8. Warn when there are
    at least 10 nodes with those degrees.
    """
    rc_nodes = []
    for i, rc in enumerate(metrics.rich_club):
        K08 = [k for k, v in rc.items() if v >= 0.8]
        if K08:
            rc_nodes.append(sum(metrics.node_degree_distribution[i] >= np.min(K08)))
        else:
            rc_nodes.append(0)
    warning = np.array(rc_nodes) >= 10
    msg = pd.Series(['warning' if warning[i] else ''
        for i in range(len(warning))], metrics.index)
    return msg

def analyze_metrics(metrics):
    msg = pd.DataFrame(index=metrics.index)
    msg['max_degree'] = check_max_degree(metrics)
    msg['mean_degree'] = check_mean_degree(metrics)
    msg['median_degree'] = check_median_degree(metrics)
    msg['degree_assortativity'] = check_degree_assortativity(metrics)
    msg['rich_club'] = check_rich_club(metrics)
    return msg

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
