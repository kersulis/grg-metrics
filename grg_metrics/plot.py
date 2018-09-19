import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def remove_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def draw_graph_lines(buses, positions, G):
    """Let `buses` (list of strings) be embedded in a plane
    with coordinates given by `positions`. Let `G` be a
    NetworkX graph containing edge information (keyed to same
    strings as `buses`). Return a LineCollection representing
    all connections between buses in G.
    """
    segments = []
    voltages = np.zeros(len(G.edges()))
    for i, e in enumerate(G.edges(data=True)):
        fidx, tidx = buses.index(e[0]), buses.index(e[1])
        segments.append([positions[fidx, :], positions[tidx, :]])
        fv = G.graph['voltage_levels'][e[2]['voltage_level_1_id']]['nominal_value']
        tv = G.graph['voltage_levels'][e[2]['voltage_level_2_id']]['nominal_value']
        if fv == tv:
            voltages[i] = fv
        else:
            voltages[i] = min(fv,tv)
    widths = np.maximum(1, 2*voltages/np.max(voltages))
    lc = matplotlib.collections.LineCollection(segments, zorder=0, cmap=plt.cm.Blues, linewidths=widths)
    return lc
