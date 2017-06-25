import os, json
import grg_grgdata
import networkx as nx
import numpy as np

def walk_components(grg_data):
    """Recursively walk grg components.
    Code taken from Carleton Coffrin.
    """
    for key, value in grg_data.items():
        yield key, value
        if isinstance(value, dict):
            for key in value:
                if 'components' in key:
                    for nested_key, nested_value in walk_components(value[key]):
                        yield nested_key, nested_value

def grg2nx(data):
    """Given a GRGv1.0 JSON document, return a networkx graph.

    Properties embedded in the graph:
    - network
        - id
        - subtype
        - per_unit
        - description (optional)
    - bus
        - type (can be 'bus', 'busbar', or 'logical_bus')
        - id
        - link (only for 'busbar' and 'logical_bus')
        - voltage
    - ac_line
        - id
        - voltage_level_1_id
        - voltage_level_2_id
        - shunt_1
        - shunt_2
        - impedance
        - current_limits_1
        - current_limits_2
    - two_winding_transformer
        - id
        - voltage_level_1_id
        - voltage_level_2_id

    Notes:
    - 'bus', 'busbar', and 'logical_bus' are all considered buses.
    - Edges are taken from both 'ac_line' and 'two_winding_transformer' objects.
    - Unlike iGRG, GRG has no 'status' field, so all buses and lines are included.
    """
    G = nx.Graph()

    # embed these properties into networkx graph
    network_props = ['id', 'subtype', 'per_unit', 'description']
    bus_props = ['type', 'id', 'link', 'voltage']
    line_props = [
        'id', 'voltage_level_1_id', 'voltage_level_2_id',
        'shunt_1', 'shunt_2', 'impedance',
        'current_limits_1', 'current_limits_2'
    ]
    transformer_props = ['id', 'voltage_level_1_id', 'voltage_level_2_id']

    # embed network properties
    for p in network_props:
        if p in data['network']:
            G.graph[p] = data['network'][p]

    for identifier, component in walk_components(data['network']['components']):
        if component['type'] in ['bus', 'busbar', 'logical_bus']:
            G.add_node(identifier, type=component['type'])
            for p in bus_props:
                if p in component:
                    G.node[identifier][p] = component[p]
        elif component['type'] == 'ac_line':
            f, t = component['link_1'], component['link_2']
            G.add_edge(f,t, type=component['type'])
            for p in line_props:
                if p in component:
                    G.edge[f][t][p] = component[p]
        elif component['type'] == 'two_winding_transformer':
            f, t = component['link_1'], component['link_2']
            G.add_edge(f,t, type=component['type'])
            for p in transformer_props:
                if p in component:
                    G.edge[f][t][p] = component[p]
    return G
