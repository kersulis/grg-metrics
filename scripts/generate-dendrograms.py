# This script generates clique merge dendrograms for
# NESTA GRG-format test networks. See the configuration
# block.

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import sys
sys.path.append('../')
import grg_metrics
from scipy.cluster.hierarchy import dendrogram

################ Configuration ###################
# NESTA GRG location
nesta_folder = '/home/jk/gdrive/GridChar/NESTA_GRGv1.1_PU/'

# where to put generated dendrograms
out_folder = 'dendrograms/'

# output format
fmt = 'pdf'

# which cases to generate from
cnames = grg_metrics.nesta_v11_representative()[-21:28]
################### End Configuration #############

for cname in cnames:
    data = grg_metrics.parse_grg_case_file(nesta_folder + cname + '.json')
    G = grg_metrics.grg2nx(data)

    Gchord = grg_metrics.chordal_extension(G)
    cliques = list(nx.chordal_graph_cliques(Gchord))

    M = grg_metrics.clique_merge(cliques)
    Z = M['linkage']

    Zpos = Z.copy()
    offset = -min(Z[:,2])
    Zpos[:,2] = Z[:,2] + offset

    mpl.rcParams['font.family'] = 'monospace'
    fig, ax = plt.subplots(figsize=(11,8.5))
    grg_metrics.remove_spines(ax)
    plt.title('Clique Merge Dendrogram for ' + cname)
    plt.ylabel('Change in number of variables due to merge')

    labels = [np.sort(np.array(list(c))) + 1 for c in cliques]
    pad3 = lambda s: str(s).ljust(4)
    labels = [''.join(map(pad3, l)) for l in labels]

    dendrogram(Zpos, leaf_rotation=90, labels=labels,
              p=7, truncate_mode=None, leaf_font_size=10,
              color_threshold=offset, above_threshold_color='k');

    # add line at 0
    plt.plot(np.array(ax.get_xlim()),[offset, offset],'k--')

    yticks = np.linspace(0, 3*offset, 7)
    yticklabels = np.round(yticks - offset)
    plt.yticks(yticks, yticklabels)
    plt.ylim(0, 3*offset)

    plt.tight_layout()
    fname = out_folder + cname + fmt
    plt.savefig(fname)
    print('Generated dendrogram: ' + fname)
