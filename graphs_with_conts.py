import json
import torch
from torch_geometric.data import HeteroData
from pathlib import Path
import pandas as pd
from numpy import random

def build_year_graphs_with_contingencies(gen_dir: Path, contingency_dir: Path,
                                         contingency_solution_dir: Path,
                                         base_solution_dir: Path,
                                         start_date: str, end_date: str, 
                                         lookahead_hours=36, include_contingencies=True,
                                         num_contingencies_per_day=5):
    """
    Builds daily HeteroData graphs from JSON data, including contingency scenarios.
    Only creates graphs where corresponding solution files exist.
    
    Args:
        gen_dir: Directory containing base case JSON files (e.g., data/case118_data)
        contingency_dir: Directory containing contingency data files (e.g., data/contingency_cases/case118_data)
        contingency_solution_dir: Directory with contingency solutions to check existence (e.g., data/contingency_cases/case118_contingency_solutions)
        base_solution_dir: Directory with base case solutions to check existence (e.g., data/case118_solutions)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        lookahead_hours: Time horizon (default 36)
        include_contingencies: Whether to include contingency scenarios
        num_contingencies_per_day: Max contingencies per day (default 5)
    
    Returns:
        List of graphs: [base_day1, cont1_day1, ..., base_day2, ...]
    """
    
    # Load sample file to get static topology
    sample_file = gen_dir / f"{start_date}.json"
    with open(sample_file) as f:
        data = json.load(f)

    lines = data["Transmission lines"]
    buses = data["Buses"]
    generators = data["Generators"]

    bus2idx = {bus_name: i for i, bus_name in enumerate(buses.keys())}
    gen2idx = {gen_name: i for i, gen_name in enumerate(generators.keys())}
    
    # Create line_id to edge mapping for contingencies
    line2edge = {}
    for line_id, props in lines.items():
        src = bus2idx[props["Source bus"]]
        dst = bus2idx[props["Target bus"]]
        line2edge[line_id] = (src, dst)

    graphs = []
    dates = pd.date_range(start_date, end_date, freq="D")

    prev_startup_costs = {}
    prev_startup_delays = {}
    
    total_base = 0
    total_cont = 0
    total_skipped = 0

    for day in dates:
        date_str = str(day.date())
        
        # Build base case graph ONLY if solution exists
        base_file = gen_dir / f"{date_str}.json"
        base_solution_file = Path(base_solution_dir) / f"{date_str}_solution.json"
        
        if not base_file.exists() or not base_solution_file.exists():
            continue
            
        base_graph = build_single_graph(
            base_file, bus2idx, gen2idx, line2edge, 
            lookahead_hours, prev_startup_costs, prev_startup_delays,
            failed_lines=None
        )
        
        if base_graph is not None:
            graphs.append(base_graph)
            total_base += 1
        
        # Build contingency graphs if requested
        if include_contingencies:
            # Find all contingency DATA files for this date
            contingency_files = list(contingency_dir.glob(f"{date_str}_c*.json"))
            
            # For each contingency data file, check if solution exists
            for cont_file in contingency_files[:num_contingencies_per_day]:
                # Extract contingency ID from filename: 2017-01-01_c91.json -> c91
                cont_id = cont_file.stem.split('_c')[-1]
                
                # Check if solution file exists
                solution_file = Path(contingency_solution_dir) / f"{date_str}_c{cont_id}_solution.json"
                
                # DEBUG: Print for first few
                if len(graphs) < 10:
                    print(f"  Checking: {solution_file.name} - Exists: {solution_file.exists()}")
                
                if not solution_file.exists():
                    # Skip this contingency - no solution available
                    total_skipped += 1
                    continue
                
                # Load contingency info to find failed lines
                with open(cont_file) as f:
                    cont_data = json.load(f)
                
                failed_lines = []
                if "Contingencies" in cont_data:
                    for c_name, c_info in cont_data["Contingencies"].items():
                        if c_name == f"c{cont_id}":
                            failed_lines = c_info.get("Affected lines", [])
                            break
                
                # Build graph with failed lines removed
                cont_graph = build_single_graph(
                    cont_file, bus2idx, gen2idx, line2edge,
                    lookahead_hours, prev_startup_costs, prev_startup_delays,
                    failed_lines=failed_lines
                )
                
                if cont_graph is not None:
                    graphs.append(cont_graph)
                    total_cont += 1
    
    print(f"  Built: {total_base} base cases, {total_cont} contingencies, skipped {total_skipped}")
    
    return graphs


def build_single_graph(file_path, bus2idx, gen2idx, line2edge, lookahead_hours,
                      prev_startup_costs, prev_startup_delays, failed_lines=None):
    """
    Builds a single HeteroData graph from a JSON file.
    
    Args:
        failed_lines: List of line IDs to remove (e.g., ['l91', 'l15'])
    """
    if not file_path.exists():
        return None
        
    with open(file_path) as f:
        data = json.load(f)

    generators = data["Generators"]
    buses = data["Buses"]
    lines = data["Transmission lines"]

    data_hetero = HeteroData()

    # Build generator features
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

    # Build bus features
    bus_features = []
    for bus_data in buses.values():
        load_seq = bus_data["Load (MW)"]
        if isinstance(load_seq, (int, float)):
            load_seq = [load_seq] * lookahead_hours
        elif len(load_seq) < lookahead_hours:
            load_seq = list(load_seq) + [0.0]*(lookahead_hours - len(load_seq))
        bus_features.append(load_seq)

    bus_features = torch.tensor(bus_features, dtype=torch.float)
    data_hetero["bus"].x = bus_features

    # Build transmission edges, removing failed lines if specified
    bus_edge_index = []
    bus_edge_attr = []
    failed_line_set = set(failed_lines) if failed_lines else set()
    
    for line_id, props in lines.items():
        # Skip failed lines in contingency scenarios
        if line_id in failed_line_set:
            continue
            
        src = bus2idx[props["Source bus"]]
        dst = bus2idx[props["Target bus"]]
        bus_edge_index.append([src, dst])
        bus_edge_index.append([dst, src])
        bus_edge_attr.append([props["Reactance (ohms)"], props["Susceptance (S)"]])
        bus_edge_attr.append([props["Reactance (ohms)"], props["Susceptance (S)"]])
    
    if bus_edge_index:  # Only add if edges exist
        bus_edge_index = torch.tensor(bus_edge_index, dtype=torch.long).T
        bus_edge_attr = torch.tensor(bus_edge_attr, dtype=torch.float)
        data_hetero["bus", "transmission", "bus"].edge_index = bus_edge_index
        data_hetero["bus", "transmission", "bus"].edge_attr = bus_edge_attr

    # Build reserve nodes
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

    return data_hetero


# Build graphs for all cases with contingencies
print("Building case300 graphs...")
graphs_300 = build_year_graphs_with_contingencies(
    gen_dir=Path("data/case300_data"),
    contingency_dir=Path("data/contingency_cases/case300_data"),
    contingency_solution_dir=Path("data/contingency_cases/case300_contingency_solutions"),
    base_solution_dir=Path("data/case300_solutions"),
    start_date="2017-01-01", 
    end_date="2017-12-31",
    include_contingencies=True,
    num_contingencies_per_day=5
)

print("Building case57 graphs...")
graphs_57 = build_year_graphs_with_contingencies(
    gen_dir=Path("data/case57_data"),
    contingency_dir=Path("data/contingency_cases/case57_data"),
    contingency_solution_dir=Path("data/contingency_cases/case57_contingency_solutions"),
    base_solution_dir=Path("data/case57_solutions"),
    start_date="2017-01-01", 
    end_date="2017-12-31",
    include_contingencies=True,
    num_contingencies_per_day=5
)

print("Building case118 graphs...")
graphs_118 = build_year_graphs_with_contingencies(
    gen_dir=Path("data/case118_data"),
    contingency_dir=Path("data/contingency_cases/case118_data"),
    contingency_solution_dir=Path("data/contingency_cases/case118_contingency_solutions"),
    base_solution_dir=Path("data/case118_solutions"),
    start_date="2017-01-01", 
    end_date="2017-12-31",
    include_contingencies=True,
    num_contingencies_per_day=5
)

# Combine all graphs
all_graphs = graphs_300 + graphs_57 + graphs_118

print(f"\nTotal graphs built: {len(all_graphs)}")
print(f"Expected: ~{(365*1) * 6 * 3} graphs (1 years × 6 scenarios × 3 cases)")
print(f"\nSample graph structures:")
print(f"Case 300: {graphs_300[0]}")
print(f"Case 57: {graphs_57[0]}")
print(f"Case 118: {graphs_118[0]}")

# Save
torch.save(all_graphs, "graphs/mega_train_with_contingencies.pt")
print(f"\nSaved to: graphs/mega_train_with_contingencies.pt")