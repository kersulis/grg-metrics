# This script generates clique merge graphs that encode the
# algorithm's behavior. Each node in the graph represents a
# power system bus, clique, or clique merge. These nodes are
# connected according to the progress of the algorithm. The
# JSON data written by this script is intended to be rendered
# as a Sankey diagram by D3.

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys
sys.path.append('../')
import grg_metrics
import json
from networkx.readwrite import json_graph

################ Configuration ###################
# NESTA GRG location
nesta_folder = '/home/jk/gdrive/GridChar/NESTA_GRGv1.1_PU/'

# where to put clique merge behavior JSON files
out_folder = 'sankey/data/'

# which networks to generate data from:
cnames = grg_metrics.nesta_v11_representative()[-21:28]
################ End configuration ###############

for cname in cnames:
    data = grg_metrics.parse_grg_case_file(nesta_folder + cname + '.json')
    G = grg_metrics.grg2nx(data)

    Gchord = grg_metrics.chordal_extension(G)
    cliques = list(nx.chordal_graph_cliques(Gchord))
    T = grg_metrics.clique_graph_spanning_tree(cliques)
    M = grg_metrics.clique_merge(cliques)
    Gm = M['Gmerge']

    d = json_graph.node_link_data(Gm)

    # write json
    fname = out_folder + cname + '.json'
    f = open(fname, 'w')
    try:
        json.dump(d, f)
    finally:
        f.close()
        print('Wrote clique merge behavior JSON data to ' + fname)
