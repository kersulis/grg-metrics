# tests take ~20s on a desktop
# make sure to update nesta_dir below!
nesta_dir = '../../NESTA_GRGv1.0'

import os, sys, json, collections
import numpy as np
sys.path.append('C:\\Users\\J K\\Google Drive\\GridChar\\grg-metrics')
from grg_metrics import grg2nx
import networkx as nx

def find_files():
    files = []

    for file_name in os.listdir(nesta_dir):
        if file_name.endswith('.json'):
            files.append(os.path.join(nesta_dir, file_name))

    return files
file_names = find_files()

def subcomponents(comp):
    result = {}
    types = [v['type'] for k, v in comp.items()]
    for t in set(types):
        result[t] = types.count(t)
    return result

"""
for all of NESTA: show types and frequencies of elements in components lists,
substation_components lists, and voltage_level_components lists
"""
def test_components():
    nesta_components = collections.Counter()
    nesta_substation = collections.Counter()
    nesta_vlevel = collections.Counter()
    components = []
    substation = []
    for file_name in file_names:
        with open(file_name, 'r') as file:
            data = json.load(file)
            subcomps = subcomponents(data['network']['components'])
            nesta_components.update(subcomps)
            for k, v in data['network']['components'].items():
                if v['type'] == 'substation':
                    nesta_substation.update(subcomponents(v['substation_components']))
                    for kk, vv in v['substation_components'].items():
                        if vv['type'] == 'voltage_level':
                            nesta_vlevel.update(subcomponents(vv['voltage_level_components']))
    # network['components'] should have only 'ac_line' and 'substation'
    # substations have only transformers and voltage levels
    # voltage levels have everything else
    assert set(nesta_components.keys()) == set(['ac_line', 'substation'])
    assert set(nesta_substation.keys()) == set(['two_winding_transformer', 'voltage_level'])
    assert set(nesta_vlevel.keys()) == set(['bus', 'generator', 'load', 'shunt', 'synchronous_condenser'])

"""
Ensure assumptions related to bus definitions and references in GRG docs
are valid. Verify grg2nx() is working properly.
"""
def test_graphs():
    for file_name in file_names:
        with open(file_name, 'r') as file:
            data = json.load(file)
            bus_references = []
            bus_definitions = []
            for k, v in data['network']['components'].items():
                if v['type'] == 'ac_line':
                    bus_references.append(v['link_1'])
                    bus_references.append(v['link_2'])

                if v['type'] == 'substation':
                    for kk, vv in v['substation_components'].items():
                        if vv['type'] == 'two_winding_transformer':
                            bus_references.append(vv['link_1'])
                            bus_references.append(vv['link_2'])

                        if vv['type'] == 'voltage_level':
                            for kkk, vvv in vv['voltage_level_components'].items():
                                nbus = 0
                                if vvv['type'] == 'bus':
                                    bus_definitions.append(kkk)
                                    nbus += 1
                                if nbus == 2:
                                    print('two buses at one vlevel')
            # if following 2 assertions hold, then grg2nx captures all
            # node and edge definitions
            G = grg2nx(data)
            assert len(G.nodes()) == len(bus_definitions)
            assert len(set([e[0] for e in G.edges()]) | (set([e[1] for e in G.edges()]))) == len(set(bus_references))

            # every referenced bus should be defined:
            for b in bus_references:
                assert b in bus_definitions
            # every defined bus should be referenced:
            for b in bus_definitions:
                # if file_name.split('\\')[-1] != 'nesta_case3375wp_mp.json':
                assert b in bus_references

def test_data():
    assert file_names[0].split('\\')[-1] == 'nesta_case118_ieee.json'
    with open(file_names[0], 'r') as file:
        data = json.load(file)
        G = grg2nx(data)
    assert G.graph['id'] == 'nesta_case118_ieee'
    assert sorted(G.nodes())[-10:] == [
    'bus_109',
    'bus_110',
    'bus_111',
    'bus_112',
    'bus_113',
    'bus_114',
    'bus_115',
    'bus_116',
    'bus_117',
    'bus_118'
    ]
    assert G.node['bus_002']['id'] == 'bus_450'
    assert len(G.edges()) == 179
    assert G.edge['bus_005']['bus_003']['voltage_level_1_id'] == 'voltage_level_3'
