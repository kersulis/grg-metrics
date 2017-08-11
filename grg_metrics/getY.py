import numpy as np
import scipy as sp
import grg_metrics

def getY(grg_data, dc=False):
    """Given a GRGv1.0 JSON document, return the admittance matrix.

    Translated from MATPOWER. Checked against MATPOWER output for
    several networks. Maximum elementwise difference was ~6%.
    Networkx verified isomorphism as well.

    If `dc` is True, only the imaginary portion of Y is returned.
    """

    f, t, Zs, b1, b2, tap, shift = get_Ybus_vectors(grg_data)
    buses = sorted(list(set(f) | set(t)))
    nl, nb = len(f), len(buses)
    f = str2ind(f, buses)
    t = str2ind(t, buses)
    Ysh = get_shunt(grg_data, buses)

    Ys = 1./Zs  # series admittance

    tap*np.exp(1j*shift*np.pi/180) # add tap shifters

    Ytt = Ys + b2
    Yff = (Ys + b1)/(tap*np.conj(tap))
    Yft = -Ys/np.conj(tap)
    Ytf = -Ys/tap

    Cf = sp.sparse.csc_matrix((np.ones(nl), (range(nl), f)), shape=(nl, nb))
    Ct = sp.sparse.csc_matrix((np.ones(nl), (range(nl), t)), shape=(nl, nb))

    i = np.tile(range(nl), 2)
    Yf = sp.sparse.csc_matrix((np.concatenate((Yff, Yft)), (i, np.concatenate((f, t)))), shape=(nl, nb))
    Yt = sp.sparse.csc_matrix((np.concatenate((Ytf, Ytt)), (i, np.concatenate((f, t)))), shape=(nl, nb))

    # build Ybus
    Ybus = np.dot(Cf.T, Yf) + np.dot(Ct.T, Yt) + sp.sparse.csc_matrix((Ysh, (range(nb), range(nb))), shape=(nb, nb)).tocsc()
    if dc:
        return sp.sparse.csc_matrix(np.imag(Ybus.todense())), buses
    else:
        return Ybus, buses

def get_Ybus_vectors(grg_data):
    """
        (f, t, Zs, b1, b2, tap, shift) = get_Ybus_vectors(grg_data)

    Return vectors needed to compute Ybus.
    """

    def extract_ftzb(grg_component):
        """For a given `ac_line` or `two_winding_transformer` object,
        return:

        - f (from)
        - t (to)
        - Zs (series impedance)
        - b1 (side 1 shunt)
        - b2 (side 2 shunt)
        - tap (tap ratio)
        - shift (phase shift)
        """
        def extract_var(var):
            return var['lb'], var['ub']

        def extract_z(imp):
            r, x = imp['resistance'], imp['reactance']
            if not isinstance(r, float):
                lb, ub = extract_var(r['var'])
                # assert lb == ub
                r = (lb + ub)/2
            if not isinstance(x, float):
                lb, ub = extract_var(x['var'])
                # assert lb == ub
                x = (lb + ub)/2
            return r + 1j*x

        def extract_b(shunt):
            g, b = shunt['conductance'], shunt['susceptance']
            if not isinstance(g, float):
                lb, ub = extract_var(g['var'])
                assert lb == ub
                g = lb
            if not isinstance(b, float):
                lb, ub = extract_var(b['var'])
                assert lb == ub
                b = lb
            return g + 1j*b

        f, t = grg_component['link_1'], grg_component['link_2']
        c = grg_component if (grg_component['type'] == 'ac_line') else grg_component['tap_changer']
        Zs = extract_z(c['impedance'])

        if 'shunt_1' in c.keys():
            b1 = extract_b(c['shunt_1'])
            b2 = extract_b(c['shunt_2'])
            tap = 1.0
            shift = 0.0
        else:
            b1 = extract_b(c['shunt'])/2
            b2 = extract_b(c['shunt'])/2
            tap = c['tap_ratio']
            if not isinstance(tap, float):
                lb, ub = extract_var(tap['var'])
                if lb != ub:
                    tap = 1.0
                else:
                    tap = lb
            shift = c['angle_shift']
            if not isinstance(shift, float):
                lb, ub = extract_var(shift['var'])
                # assert lb == ub
                shift = lb

        return f, t, Zs, b1, b2, tap, shift

    lines  = {k:v for k,v in grg_metrics.walk_components(grg_data['network']['components'])
              if (v['type'] == 'ac_line') | (v['type'] == 'two_winding_transformer')}
    d = [extract_ftzb(line) for line_id, line in lines.items()]

    return [np.array(x) for x in zip(*d)]

def str2ind(vs, buses):
    """In:
    - `vs`: a vector of bus names (strings)
    - `buses`: a list of all buses in the network
    Out: `vi`, corresponding to vs where strings are replaced by indices in `buses`.

    This function uses zero-based indexing, so a `vs` elemnt
    equal to `buses[0]` becomes 0.
    """
    vs = np.array(vs)
    vi = np.zeros(len(vs), dtype=int)
    for i, s in enumerate(buses):
        vi[np.where(vs==s)] = i
    return vi

def get_shunt(grg_data, buses):
    """Returns a vector of buses with shunts
    and corresponding shunt values (complex).
    """
    shunt_buses = []
    shunts = []
    for cid, c in grg_metrics.walk_components(grg_data['network']['components']):
        if c['type'] == 'shunt':
            shunt_buses.append(c['link'])
            shunts.append(c['shunt']['conductance'] + 1j*c['shunt']['susceptance'])

    if shunts == []:
        return np.zeros(len(buses))
    else:
        # convert bus names to indices:
        shunt_buses = str2ind(shunt_buses, buses)
        Ysh = np.zeros(len(buses), dtype=complex)
        Ysh[shunt_buses] = shunts
        return Ysh
