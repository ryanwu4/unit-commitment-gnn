import json
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
import pandas as pd
import networkx as nx
from numpy import random

gen_dir = Path("data/case118_data")

# ASSUME TRANSMISSION STATIC FOR NOW
with open(gen_dir / "2017-01-01.json") as f:
    data = json.load(f)

lines = data["Transmission lines"]
buses = data["Buses"]
generators = data["Generators"]

bus2idx = {bus_name: i for i, bus_name in enumerate(buses.keys())}
gen2idx = {gen_name: i for i, gen_name in enumerate(generators.keys())}

bus_edge_index = []
bus_edge_attr = []
for l, props in lines.items():
    src = bus2idx[props["Source bus"]]
    dst = bus2idx[props["Target bus"]]
    bus_edge_index.append([src, dst])
    bus_edge_index.append([dst, src])
    bus_edge_attr.append([props["Reactance (ohms)"], props["Susceptance (S)"]])
    bus_edge_attr.append([props["Reactance (ohms)"], props["Susceptance (S)"]])
bus_edge_index = torch.tensor(bus_edge_index, dtype=torch.long).T
bus_edge_attr = torch.tensor(bus_edge_attr, dtype=torch.float)

graphs = []

dates = pd.date_range("2017-01-01", "2017-12-31", freq="D")
lookahead_hours = 36

for day in dates:
    if day == pd.Timestamp("2017-12-29"):
        continue
    gen_file = gen_dir / f"{day.date()}.json"
    with open(gen_file) as f:
        data = json.load(f)

    generators = data["Generators"]
    buses = data["Buses"]

    data_hetero = HeteroData()

    # --- GENERATOR FEATURES ---
    gen_features = []
    gen_bus_idx = []

    prev_startup_costs = {}
    prev_startup_delays = {}

    for g_name, g in generators.items():
        # --- Fix startup costs ---
        if len(g["Startup costs ($)"]) < 3:
            g["Startup costs ($)"] = prev_startup_costs.get(g_name, [0.0]*3)
        else:
            prev_startup_costs[g_name] = g["Startup costs ($)"]

        # --- Fix startup delays ---
        if len(g["Startup delays (h)"]) < 3:
            g["Startup delays (h)"] = prev_startup_delays.get(g_name, [0.0]*3)
        else:
            prev_startup_delays[g_name] = g["Startup delays (h)"]  

        features = []
        features.extend(g["Production cost curve (MW)"])
        features.extend(g["Production cost curve ($)"])
        features.extend(g["Startup costs ($)"])
        features.extend(g["Startup delays (h)"])
        features.append(g["Ramp up limit (MW)"])
        features.append(g["Ramp down limit (MW)"])
        features.append(g["Startup limit (MW)"])
        features.append(g["Shutdown limit (MW)"])
        features.append(g["Minimum uptime (h)"])
        features.append(g["Minimum downtime (h)"])
        features.append(g["Initial status (h)"])
        features.append(g["Initial power (MW)"])
        gen_features.append(features)

        gen_bus_idx.append(bus2idx[g["Bus"]])
    for i, feats in enumerate(gen_features):
        if len(feats) != 24:
            print(f"Generator {list(generators.keys())[i]} has {len(feats)} features")
            print(generators[list(generators.keys())[i]])
            print(day)
    gen_features = torch.tensor(gen_features, dtype=torch.float)
    # Normalize generator features (per column)
    gen_features = (gen_features - gen_features.mean(dim=0)) / (gen_features.std(dim=0) + 1e-6)

    data_hetero["generator"].x = gen_features
    src = torch.arange(len(gen_bus_idx))
    dst = torch.tensor(gen_bus_idx)
    data_hetero["generator", "produces_at", "bus"].edge_index = torch.stack([src, dst], dim=0)
    data_hetero['bus', 'served_by', 'generator'].edge_index = torch.stack([dst, src], dim=0)

    # --- BUS FEATURES ---
    bus_features = []
    for bus_name, bus_data in buses.items():
        load_seq = bus_data["Load (MW)"]
        if isinstance(load_seq, (int, float)):
            load_seq = [load_seq] * lookahead_hours
        elif len(load_seq) < lookahead_hours:
            load_seq = list(load_seq) + [0.0]*(lookahead_hours - len(load_seq))
        bus_features.append(load_seq)

    bus_features = torch.tensor(bus_features, dtype=torch.float)
    # Normalize bus features
    bus_features = (bus_features - bus_features.mean(dim=0)) / (bus_features.std(dim=0) + 1e-6)

    data_hetero["bus"].x = bus_features
    data_hetero["bus", "transmission", "bus"].edge_index = bus_edge_index
    data_hetero["bus", "transmission", "bus"].edge_attr = bus_edge_attr

    # --- RESERVE FEATURES ---
    reserve_features = []
    reserve_names = list(data.get("Reserves", {}).keys())
    reserve_name2idx = {r: i for i, r in enumerate(reserve_names)}

    for r_name in reserve_names:
        r_data = data["Reserves"][r_name]
        amount_seq = r_data["Amount (MW)"]
        if len(amount_seq) < lookahead_hours:
            amount_seq = list(amount_seq) + [0.0]*(lookahead_hours - len(amount_seq))
        reserve_features.append(amount_seq)

    reserve_features = torch.tensor(reserve_features, dtype=torch.float)
    # Normalize reserve features
    if reserve_features.size(0) == 1:
        reserve_features = (reserve_features - reserve_features.mean()) / (reserve_features.std() + 1e-6)
    else:
        reserve_features = (reserve_features - reserve_features.mean(dim=0)) / (reserve_features.std(dim=0) + 1e-6)

    data_hetero["reserve"].x = reserve_features

    # --- RESERVE-GENERATOR EDGES ---
    reserve_src = []
    reserve_dst = []
    for r_idx, r_name in enumerate(reserve_names):
        for g_idx, g in enumerate(generators.values()):
            if r_name in g.get("Reserve eligibility", []):
                reserve_src.append(r_idx)
                reserve_dst.append(g_idx)

    if reserve_src:
        data_hetero["reserve", "backed_by", "generator"].edge_index = torch.tensor([reserve_src, reserve_dst], dtype=torch.long)
        #data_hetero["generator", "supports", "reserve"].edge_index = torch.tensor([reserve_dst, reserve_src], dtype=torch.long)

    graphs.append(data_hetero)

# --- Step 0: pick anchors ---
num_anchors = 15
bus_nodes = list(bus2idx.values())
anchors = random.choice(bus_nodes, num_anchors, replace=False)
print("Selected anchor bus indices:", anchors)

# --- Step 1: build networkx graph for shortest-path distances ---
G = nx.Graph()
for l, props in lines.items():
    src = bus2idx[props["Source bus"]]
    dst = bus2idx[props["Target bus"]]
    G.add_edge(src, dst)

# --- Step 2: compute shortest-path distances from all nodes to anchors ---
bus_distances = torch.zeros(len(bus_nodes), num_anchors)
for i, a in enumerate(anchors):
    lengths = nx.single_source_shortest_path_length(G, a)
    for b in bus_nodes:
        bus_distances[b, i] = lengths.get(b, float(len(bus_nodes)))  # large number if unreachable

# --- Step 3: attach anchor distances to node features ---
for day_graph in graphs:
    # bus nodes
    day_graph["bus"].x = torch.cat([day_graph["bus"].x, bus_distances], dim=1)

    # generator nodes: use distance of their main bus
    gen_bus_idx = day_graph["generator", "produces_at", "bus"].edge_index[1]
    gen_anchor_feats = bus_distances[gen_bus_idx]
    day_graph["generator"].x = torch.cat([day_graph["generator"].x, gen_anchor_feats], dim=1)

    # reserve nodes: use distance of each reserve’s bus? if you have mapping
    # reserve_name -> eligible generator -> bus, pick first generator’s bus
    if day_graph["reserve"].x.size(0) > 0:
        res_anchor_feats = []
        for r_idx in range(day_graph["reserve"].x.size(0)):
            edge_mask = (day_graph["reserve", "backed_by", "generator"].edge_index[0] == r_idx)
            gen_idx = day_graph["reserve", "backed_by", "generator"].edge_index[1][edge_mask][0]
            bus_idx = gen_bus_idx[gen_idx]
            res_anchor_feats.append(bus_distances[bus_idx])
        res_anchor_feats = torch.stack(res_anchor_feats)
        day_graph["reserve"].x = torch.cat([day_graph["reserve"].x, res_anchor_feats], dim=1)


print(f"Created {len(graphs)} daily heterogeneous graphs with 36-hour lookahead.")
print(graphs[0])
torch.save(graphs, "graphs/all_hetero_graphs_normalized_114.pt")
