import numpy as np
import grg_metrics

def return_generator_buses(grg_data):
    """Returns a vector of all bus names with
    attached generators.
    """
    generator_buses = []
    for identifier, component in grg_metrics.walk_components(grg_data['network']['components']):
        if component['type'] == 'generator':
            generator_buses.append(component['link'])
    return generator_buses

def return_participation_factors(gen_buses, all_buses):
    """Return a vector of participation factors that sum to 1.
    Each element corresponding to a bus with a generator
    (belonging to gen_buses) gets a value of 1/ng,
    where ng = len(gen_buses). All other elements are 0.
    """
    k = np.zeros(len(all_buses))
    for g in gen_buses:
        k[all_buses.index(g)] = 1.0/len(gen_buses)
    return k

def shift_factor_distance(isf, ref):
    """Calculate pairwise power transfer distance.
    `ref` is the index of the reference bus.
    """
    l, n = isf.shape
    n += 1
    PT = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                PT[i, j] = 0
            else:
                P = np.zeros(n)
                P[i] = 1
                P[j] = -1
                P = np.delete(P, ref)
                PT[i, j] = np.sum(np.abs(isf.dot(P)))
    return PT

def inj_shift_factors(Y, lines, ref, k=[]):
    """Calculate injection shift factor matrix.
    Each row corresponds to a line in the network.
    Each column corresponds to a node.
    Credit to Jonathon Martin for derivation.

    Inputs:
    * `Y`: full admittance matrix (only imaginary part is used)
    * `lines`: vector of tuples; each tuple encodes a line as (i,j)
    * `ref`: index of angle reference bus
    * `k`: vector of generator participation factors
    """
    Y = np.imag(Y.todense())
    n, l = Y.shape[0], len(lines)

    Bflow = np.zeros((l,n))
    for idx in range(l):
        i, j = lines[idx]
        Bflow[idx, i] = Y[i, j]
        Bflow[idx, j] = -Y[i, j]

    if len(k) != 0:
        Y[:, ref] = k.reshape(n,1)
        B = Y
        Bflow[:, ref] = np.zeros(l)
    else:
        nonref = list(set(range(n)) - {ref})
        B = Y[nonref, :][:, nonref]
        Bflow = Bflow[:, nonref]

    return Bflow.dot(np.linalg.inv(B))
