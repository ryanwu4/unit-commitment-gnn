import json
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
import pandas as pd
import networkx as nx
from numpy import random

def build_year_graphs(gen_dir: Path, start_date: str, end_date: str, lookahead_hours=36, num_anchors=15):
    """Builds daily HeteroData graphs from JSON data for one year."""
    sample_file = gen_dir / f"{start_date}.json"
    with open(sample_file) as f:
        data = json.load(f)

    lines = data["Transmission lines"]
    buses = data["Buses"]
    generators = data["Generators"]

    bus2idx = {bus_name: i for i, bus_name in enumerate(buses.keys())}
    gen2idx = {gen_name: i for i, gen_name in enumerate(generators.keys())}

    bus_edge_index = []
    bus_edge_attr = []
    for props in lines.values():
        src = bus2idx[props["Source bus"]]
        dst = bus2idx[props["Target bus"]]
        bus_edge_index.append([src, dst])
        bus_edge_index.append([dst, src])
        bus_edge_attr.append([props["Reactance (ohms)"], props["Susceptance (S)"]])
        bus_edge_attr.append([props["Reactance (ohms)"], props["Susceptance (S)"]])
    bus_edge_index = torch.tensor(bus_edge_index, dtype=torch.long).T
    bus_edge_attr = torch.tensor(bus_edge_attr, dtype=torch.float)

    graphs = []
    dates = pd.date_range(start_date, end_date, freq="D")

    prev_startup_costs = {}
    prev_startup_delays = {}

    for day in dates:
        file_path = gen_dir / f"{day.date()}.json"
        if not file_path.exists():
            continue

        with open(file_path) as f:
            data = json.load(f)

        generators = data["Generators"]
        buses = data["Buses"]

        data_hetero = HeteroData()

        gen_features = []
        gen_bus_idx = []
        for g_name, g in generators.items():
            if len(g.get("Startup costs ($)", [])) < 3:
                g["Startup costs ($)"] = prev_startup_costs.get(g_name, [0.0]*3)
            else:
                prev_startup_costs[g_name] = g["Startup costs ($)"]

            if len(g.get("Startup delays (h)", [])) < 3:
                g["Startup delays (h)"] = prev_startup_delays.get(g_name, [0.0]*3)
            else:
                prev_startup_delays[g_name] = g["Startup delays (h)"]


            features = (
                g["Production cost curve (MW)"]
                + g["Production cost curve ($)"]
                + g["Startup costs ($)"]
                + g["Startup delays (h)"]
                + [ 
                    g["Ramp up limit (MW)"],
                    g["Ramp down limit (MW)"],
                    g["Startup limit (MW)"],
                    g["Shutdown limit (MW)"],
                    g["Minimum uptime (h)"],
                    g["Minimum downtime (h)"],
                    g["Initial status (h)"],
                    g["Initial power (MW)"]
                ]
            )
            gen_features.append(features)
            gen_bus_idx.append(bus2idx[g["Bus"]])

        gen_features = torch.tensor(gen_features, dtype=torch.float)
        gen_features = (gen_features - gen_features.mean(dim=0)) / (gen_features.std(dim=0) + 1e-6)

        data_hetero["generator"].x = gen_features
        src = torch.arange(len(gen_bus_idx))
        dst = torch.tensor(gen_bus_idx)
        data_hetero["generator", "produces_at", "bus"].edge_index = torch.stack([src, dst], dim=0)
        data_hetero["bus", "served_by", "generator"].edge_index = torch.stack([dst, src], dim=0)

        bus_features = []
        for bus_data in buses.values():
            load_seq = bus_data["Load (MW)"]
            if isinstance(load_seq, (int, float)):
                load_seq = [load_seq] * lookahead_hours
            elif len(load_seq) < lookahead_hours:
                load_seq = list(load_seq) + [0.0]*(lookahead_hours - len(load_seq))
            bus_features.append(load_seq)

        bus_features = torch.tensor(bus_features, dtype=torch.float)
        #bus_features = (bus_features - bus_features.mean(dim=0)) / (bus_features.std(dim=0) + 1e-6)

        data_hetero["bus"].x = bus_features
        data_hetero["bus", "transmission", "bus"].edge_index = bus_edge_index
        data_hetero["bus", "transmission", "bus"].edge_attr = bus_edge_attr

        # Reserve nodes (if exist)
        reserve_features = []
        reserve_names = list(data.get("Reserves", {}).keys())
        reserve_name2idx = {r: i for i, r in enumerate(reserve_names)}

        for r_name in reserve_names:
            r_data = data["Reserves"][r_name]
            amount_seq = r_data["Amount (MW)"]
            if len(amount_seq) < lookahead_hours:
                amount_seq = list(amount_seq) + [0.0]*(lookahead_hours - len(amount_seq))
            reserve_features.append(amount_seq)

        if reserve_features:
            reserve_features = torch.tensor(reserve_features, dtype=torch.float)
            #reserve_features = (reserve_features - reserve_features.mean(dim=0)) / (reserve_features.std(dim=0) + 1e-6)
            data_hetero["reserve"].x = reserve_features

        reserve_src, reserve_dst = [], []
        for r_idx, r_name in enumerate(reserve_names):
            for g_idx, g in enumerate(generators.values()):
                if r_name in g.get("Reserve eligibility", []):
                    reserve_src.append(r_idx)
                    reserve_dst.append(g_idx)

        if reserve_src:
            data_hetero["reserve", "backed_by", "generator"].edge_index = torch.tensor(
                [reserve_src, reserve_dst], dtype=torch.long
            )

        graphs.append(data_hetero)
    return graphs

graphs_14 = build_year_graphs(Path("data/case14_data"), "2017-01-01", "2017-12-31", num_anchors=0)
graphs_57 = build_year_graphs(Path("data/case57_data"), "2017-01-01", "2017-12-31", num_anchors=0)
graphs_118 = build_year_graphs(Path("data/case118_data"), "2017-01-01", "2017-12-31", num_anchors=0)

all_graphs = graphs_14 + graphs_57 + graphs_118
print(graphs_14[0])
print(graphs_57[0])
print(graphs_118[0])
torch.save(all_graphs, "graphs/mega_train.pt")
