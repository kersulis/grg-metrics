import os, json
import networkx as nx
import numpy as np
import warnings
import grg_grgdata

warnings.simplefilter('once')

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

def grg2nx(data, remove_stepup_transformers=False):
    """Given a GRGv4.0 JSON document, return a networkx graph.

    Properties embedded in the graph:
    - network
        - id
        - subtype
        - per_unit
        - description (optional)
        - voltage level information
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
    grg_version = data['grg_version']
    G = nx.Graph()

    # embed these properties into networkx graph
    network_props = ['id', 'type', 'subtype', 'per_unit', 'description', 'base_mva']
    bus_props = ['type', 'id', 'link', 'voltage']
    line_props = [
        'id', 'voltage_level_1_id', 'voltage_level_2_id',
        'shunt_1', 'shunt_2', 'impedance',
        'current_limits_1', 'current_limits_2'
    ]
    transformer_props = ['id', 'voltage_level_1_id', 'voltage_level_2_id']

    # for removing step-up transformers
    transformer_lowside_buses = []
    generator_buses = []
    load_buses = []

    # embed network properties
    G.graph['voltage_levels'] = dict()
    for p in network_props:
        if p in data['network']:
            G.graph[p] = data['network'][p]

    # dictionary for mapping voltage point IDs to bus IDs
    vid2bus = {}
    for identifier, component in grg_grgdata.cmd.walk_components(data):
        if component['type'] in ['bus', 'busbar', 'logical_bus']:
            vid2bus[component['link']] = identifier

    for identifier, component in grg_grgdata.cmd.walk_components(data):
        if component['type'] in ['bus', 'busbar', 'logical_bus']:
            G.add_node(identifier, type=component['type'])
            for p in bus_props:
                if p in component:
                    G.node[identifier][p] = component[p]
        elif component['type'] == 'ac_line':
            f, t = vid2bus[component['link_1']], vid2bus[component['link_2']]
            G.add_edge(f,t, type=component['type'])
            for p in line_props:
                if p in component:
                    G[f][t][p] = component[p]
        elif component['type'] == 'two_winding_transformer':
            f, t = vid2bus[component['link_1']], vid2bus[component['link_2']]
            G.add_edge(f,t, type=component['type'])
            transformer_lowside_buses.append(t)
            for p in transformer_props:
                if p in component:
                    G[f][t][p] = component[p]
        elif component['type'] == 'generator':
            generator_buses.append(component['link'])
        elif component['type'] == 'load':
            load_buses.append(component['link'])
        elif component['type'] == 'switch':
            warnings.warn('Switch found; please use the bus-branch form of your network to ensure accuracy.')
        elif component['type'] == 'voltage_level':
            G.graph['voltage_levels'][component['id']] = component['voltage']

    if remove_stepup_transformers:
        degree_one_buses = [k for k, v in nx.degree(G).items() if v == 1]
        stepup_buses = list((set(transformer_lowside_buses) & set(generator_buses) & set(degree_one_buses)) - set(load_buses))
        G.remove_nodes_from(stepup_buses)
    return G

def grg2nx_v1(data, remove_stepup_transformers=False):
    """Given a GRGv1.0 JSON document, return a networkx graph.

    Properties embedded in the graph:
    - network
        - id
        - subtype
        - per_unit
        - description (optional)
        - voltage level information
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
    grg_version = data['grg_version']
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

    # for removing step-up transformers
    transformer_lowside_buses = []
    generator_buses = []
    load_buses = []

    # embed network properties
    G.graph['voltage_levels'] = dict()
    for p in network_props:
        if p in data['network']:
            G.graph[p] = data['network'][p]

    if grg_version == "v.1.5":
        # dictionary for mapping voltage point IDs to bus IDs
        vid2bus = {}
        for identifier, component in walk_components(data['network']['components']):
            if component['type'] in ['bus', 'busbar', 'logical_bus']:
                vid2bus[component['link']] = identifier

    for identifier, component in walk_components(data['network']['components']):
        if component['type'] in ['bus', 'busbar', 'logical_bus']:
            G.add_node(identifier, type=component['type'])
            for p in bus_props:
                if p in component:
                    G.node[identifier][p] = component[p]
        elif component['type'] == 'ac_line':
            if grg_version == 'v.1.5':
                f, t = vid2bus[component['link_1']], vid2bus[component['link_2']]
            else:
                f, t = component['link_1'], component['link_2']
            G.add_edge(f,t, type=component['type'])
            for p in line_props:
                if p in component:
                    G[f][t][p] = component[p]
        elif component['type'] == 'two_winding_transformer':
            if grg_version == 'v.1.5':
                f, t = vid2bus[component['link_1']], vid2bus[component['link_2']]
            else:
                f, t = component['link_1'], component['link_2']
            G.add_edge(f,t, type=component['type'])
            transformer_lowside_buses.append(t)
            for p in transformer_props:
                if p in component:
                    G[f][t][p] = component[p]
        elif component['type'] == 'generator':
            generator_buses.append(component['link'])
        elif component['type'] == 'load':
            load_buses.append(component['link'])
        elif component['type'] == 'switch':
            warnings.warn('Switch found; please use the bus-branch form of your network to ensure accuracy.')
        elif component['type'] == 'voltage_level':
            G.graph['voltage_levels'][component['id']] = component['voltage']

    if remove_stepup_transformers:
        degree_one_buses = [k for k, v in nx.degree(G).items() if v == 1]
        stepup_buses = list((set(transformer_lowside_buses) & set(generator_buses) & set(degree_one_buses)) - set(load_buses))
        G.remove_nodes_from(stepup_buses)
    return G
