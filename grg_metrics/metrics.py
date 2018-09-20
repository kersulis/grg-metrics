import networkx as nx
import pandas as pd
import numpy as np
import grg_metrics

def node_degree_distribution(graphs, Gids):
    metrics = [np.flipud(np.sort(np.array(list(dict(nx.degree(G)).values()))))
               for G in graphs]
    return pd.Series(metrics, index=Gids, name='node_degree_distribution')

def degree_assortativity(graphs, Gids):
    metrics = [nx.degree_assortativity_coefficient(G) for G in graphs]
    return pd.Series(metrics, index=Gids, name='degree_assortativity')

def rich_club(graphs, Gids):
    metrics = [nx.rich_club_coefficient(G, normalized=False) for G in graphs]
    return pd.Series(metrics, index=Gids, name='rich_club')

def load_centrality(graphs, Gids):
    metrics = [nx.load_centrality(G) for G in graphs]
    return pd.Series(metrics, index=Gids, name='load_centrality')

def clustering(graphs, Gids):
    metrics = [np.flipud(np.sort(np.array(list(nx.clustering(G).values()))))
               for G in graphs]
    return pd.Series(metrics, index=Gids, name='node_degree_distribution')

def average_clustering(graphs, Gids):
    metrics = [nx.average_clustering(G) for G in graphs]
    return pd.Series(metrics, index=Gids, name='average_clustering')

def average_shortest_path_length(graphs, Gids):
    """This is an expensive metric to compute.
    """
    metrics = [nx.average_shortest_path_length(G) for G in graphs]
    return pd.Series(metrics, index=Gids, name='average_shortest_path_length')

def maximal_cliques(graphs, Gids):
    metrics = [list(nx.clique.find_cliques(G)) for G in graphs]
    return pd.Series(metrics, index=Gids, name='maximal_cliques')

def fiedler_value(graphs, Gids):
    """Second-smallest Eigenvalue of the graph Laplacian.
    """
    metrics = [nx.algebraic_connectivity(G) for G in graphs]
    return pd.Series(metrics, index=Gids, name='fiedler_value')

def compute_metrics(x, compute_average_shortest_path_length=False, compute_fiedler_value=False, compute_adj_spectral_radius=False, compute_maximal_cliques=False):
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
    metrics['node_degree_distribution'] = node_degree_distribution(graphs, Gids)
    metrics['max_degree'] = metrics['node_degree_distribution'].apply(max)
    metrics['mean_degree'] = metrics['node_degree_distribution'].apply(np.mean)
    metrics['median_degree'] = metrics['node_degree_distribution'].apply(np.median)
    metrics['degree_assortativity'] = degree_assortativity(graphs, Gids)
    metrics['rich_club'] = rich_club(graphs, Gids)
    metrics['clustering'] = clustering(graphs, Gids)
    metrics['average_clustering'] = average_clustering(graphs, Gids)
    if compute_maximal_cliques:
        metrics['maximal_cliques'] = maximal_cliques(graphs, Gids)
    if compute_adj_spectral_radius:
        metrics['adj_spectral_radius'] = adj_spectral_radius(graphs, Gids)
    if compute_fiedler_value:
        metrics['fiedler_value'] = fiedler_value(graphs, Gids)
    if compute_average_shortest_path_length:
        metrics['average_shortest_path_length'] = average_shortest_path_length(graphs, Gids)
    return metrics

def check_max_degree(metrics, describe=True):
    """Warning: max. degree greater than 10.
    Error: max. degree greater than 4.22*log10(x) + 3.87.

    Input `metrics` must have columns 'max_degree' and 'nodes'.
    """
    error = metrics.max_degree > 4.22*np.log10(metrics.nodes) + 3.87
    warning = (metrics.max_degree > 10) & ~error

    msg = pd.Series(index=metrics.index)
    for i in msg.index:
        if error.loc[i]:
            if describe:
                msg.loc[i] = "Error: \'%s\' has maximum degree %d, which is unusually large for a network with %d nodes." % (i, metrics.max_degree[i], metrics.nodes[i])
            else:
                msg.loc[i] = "Error"
        elif warning.loc[i]:
            if describe:
                msg.loc[i] = "Warning: \'%s\' has maximum degree %d. Nodes of such high degree are rare in power systems." % (i, metrics.max_degree[i])
            else:
                msg.loc[i] = "Warning"
    return msg

def check_mean_degree(metrics, describe=True):
    """Warning: mean degree above 3.0.
    Error: mean degree above 4.0.
    """
    error = metrics.mean_degree > 4
    warning = (metrics.mean_degree > 3) & ~error
    msg = pd.Series(index=metrics.index)
    for i in msg.index:
        if error.loc[i]:
            if describe:
                msg.loc[i] = "Error: \'%s\' has mean degree %.2f; above 4 is unrealistic." % (i, metrics.mean_degree[i])
            else:
                msg.loc[i] = "Error"
        elif warning.loc[i]:
            if describe:
                msg.loc[i] = "Warning: \'%s\' has mean degree %.2f; above 3 is rare." % (i, metrics.mean_degree[i])
            else:
                msg.loc[i] = "Warning"
    return msg

def check_median_degree(metrics, describe=True):
    """Warning: median degree = 3 and nodes > 200.
    Error: median degree > 3.
    """
    error = metrics.median_degree > 3
    warning = (metrics.median_degree == 3) & (metrics.nodes > 200) & ~error
    msg = pd.Series(index=metrics.index)
    for i in msg.index:
        if error.loc[i]:
            if describe:
                msg.loc[i] = "Error: \'%s\' has median degree %d; above 3 is unrealistic." % (i, metrics.median_degree[i])
            else:
                msg.loc[i] = "Error"
        elif warning.loc[i]:
            if describe:
                msg.loc[i] = "Warning: \'%s\' has median degree %d, which is rare for networks larger than 200 buses." % (i, metrics.median_degree[i])
            else:
                msg.loc[i] = "Warning"
    return msg

def check_degree_assortativity(metrics, describe=True):
    """Warning: outside [-0.37, 0.18].
    Error: outside [-0.64, 0.45].
    """
    error = (metrics.degree_assortativity < -0.64) | (metrics.degree_assortativity > 0.45)
    warning = (metrics.degree_assortativity < -0.37) | (metrics.degree_assortativity > 0.18) & ~error
    msg = pd.Series(index=metrics.index)
    for i in msg.index:
        if error.loc[i]:
            if describe:
                msg.loc[i] = "Error: \'%s\' has degree assortativity coefficient %.2f, more than 2 standard deviations from the mean of -0.1 for 41 test networks." % (i, metrics.degree_assortativity[i])
            else:
                msg.loc[i] = "Error"
        elif warning.loc[i]:
            if describe:
                msg.loc[i] = "Warning: \'%s\' has degree assortativity coefficient %.2f, which is over one standard deviation from the mean of -0.1 for 41 test networks." % (i, metrics.degree_assortativity[i])
            else:
                msg.loc[i] = "Warning"
    return msg

def check_rich_club(metrics, describe=True):
    """Warning: let K_0.8 be the set of degrees with
    rich club coefficients >= 0.8. Warn when there are
    at least 10 nodes with those degrees.
    """
    K08_mins = []
    rc_nodes = pd.Series(index=metrics.index)
    for i, rc in enumerate(metrics.rich_club):
        K08 = [k for k, v in rc.items() if v >= 0.8]
        if K08:
            K08_mins.append(np.min(K08))
            rc_nodes.iloc[i] = sum(metrics.node_degree_distribution[i] > np.min(K08))
        else:
            K08_mins.append(0)
            rc_nodes.iloc[i] = 0
    warning = rc_nodes >= 10
    msg = pd.Series(index=metrics.index)
    for idx, i in enumerate(msg.index):
        if warning.loc[i]:
            if describe:
                msg.loc[i] = "Warning: \'%s\' has a rich club consisting of %d nodes with degree above %d." % (i, rc_nodes[idx], K08_mins[idx])
            else:
                msg.loc[i] = "Warning"
    return msg

def analyze_metrics(metrics, describe=True):
    msg = pd.DataFrame(index=metrics.index)
    msg['max_degree'] = check_max_degree(metrics, describe=describe)
    msg['mean_degree'] = check_mean_degree(metrics, describe=describe)
    msg['median_degree'] = check_median_degree(metrics, describe=describe)
    msg['degree_assortativity'] = check_degree_assortativity(metrics, describe=describe)
    msg['rich_club'] = check_rich_club(metrics, describe=describe)
    msg = msg.fillna('')
    return msg

def nesta_v11_representative(case403=True):
    """Returns a representative sample of 33 NESTA GRG v1.1 networks,
    sorted by size.
    """
    names = [
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
    if ~case403:
        names.pop(12)
    return names

def pglib_representative():
    """Returns a representative sample of 33 PGLIB networks,
    sorted by size.
    """
    names = [
    'pglib_opf_case13659_pegase',
    'pglib_opf_case10000_tamu',
    'pglib_opf_case9241_pegase',
    'pglib_opf_case6515_rte',
    'pglib_opf_case4661_sdet',
    'pglib_opf_case3120sp_k',
    'pglib_opf_case2869_pegase',
    'pglib_opf_case2868_rte',
    'pglib_opf_case2853_sdet',
    'pglib_opf_case2736sp_k',
    'pglib_opf_case2383wp_k',
    'pglib_opf_case2316_sdet',
    # 'pglib_opf_case2224_edin',
    'pglib_opf_case2000_tamu',
    'pglib_opf_case1951_rte',
    # 'pglib_opf_case1397sp_eir',
    'pglib_opf_case1354_pegase',
    # 'pglib_opf403_rte',
    'pglib_opf_case588_sdet',
    'pglib_opf_case500_tamu',
    'pglib_opf_case300_ieee',
    'pglib_opf_case240_pserc',
    # 'pglib_opf_case189_edin',
    'pglib_opf_case200_tamu',
    'pglib_opf_case179_goc',
    'pglib_opf_case162_ieee_dtc',
    'pglib_opf_case118_ieee',
    'pglib_opf_case89_pegase',
    'pglib_opf_case73_ieee_rts',
    'pglib_opf_case57_ieee',
    'pglib_opf_case39_epri',
    'pglib_opf_case30_fsr',
    # 'pglib_opf_case30_as',
    'pglib_opf_case30_ieee',
    # 'pglib_opf_case29_edin',
    'pglib_opf_case24_ieee_rts',
    'pglib_opf_case14_ieee',
    # 'pglib_opf_case9_wscc',
    # 'pglib_opf_case6_ww',
    # 'pglib_opf_case6_c',
    'pglib_opf_case5_pjm',
    # 'pglib_opf_case4_gs',
    'pglib_opf_case3_lmbd'
    ]
    return names
