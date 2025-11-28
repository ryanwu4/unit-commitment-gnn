import os
import json
import torch
from pathlib import Path
import pandas as pd

# Base case solution folders
base_solution_folders = [
    "data/case300_solutions", 
    "data/case57_solutions", 
    "data/case118_solutions"
]

# Contingency solution folders
contingency_data_folders = [
    "data/contingency_cases/case300_contingency_solutions",
    "data/contingency_cases/case57_contingency_solutions", 
    "data/contingency_cases/case118_contingency_solutions"
]

# Date range - MUST MATCH GRAPH BUILDER
START_DATE = "2017-01-01"
END_DATE = "2017-12-31"

all_labels = []

for base_folder, cont_folder in zip(base_solution_folders, contingency_data_folders):
    print(f"Processing {base_folder}...")
    
    base_path = Path(base_folder)
    cont_path = Path(cont_folder)
    
    # Get all base case files sorted by date
    # ONLY process dates in the specified range
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    
    case_labels = []
    
    for day in dates:
        date_str = str(day.date())
        filename = f"{date_str}_solution.json"
        
        base_file_path = base_path / filename
        
        # Skip if base solution doesn't exist
        if not base_file_path.exists():
            continue
        
        # Process base case
        base_file_path = base_path / filename
        with open(base_file_path, 'r') as f:
            data = json.load(f)
        
        thermal = data["Is on"]
        gen_keys = sorted(thermal.keys(), key=lambda x: int(x[1:]))
        
        # Build label for base case
        day_labels = []
        for t in range(len(thermal[gen_keys[0]])):
            timestep_label = [thermal[g][t] for g in gen_keys]
            day_labels.append(timestep_label)
        
        case_labels.append(day_labels)
        
        # Process contingency cases for this date
        # Find all contingency files: 2017-01-01_c*_solution.json
        contingency_files = sorted(cont_path.glob(f"{date_str}_c*_solution.json"))
        
        # Include all available contingencies (limit to 5 to match graph builder)
        for cont_file in contingency_files[:5]:  # Match graph builder limit
            with open(cont_file, 'r') as f:
                cont_data = json.load(f)
            
            # Extract solution from contingency file
            cont_thermal = cont_data["Is on"]
            # Use same generator ordering as base case
            cont_gen_keys = sorted(cont_thermal.keys(), key=lambda x: int(x[1:]))
            
            cont_day_labels = []
            for t in range(len(cont_thermal[cont_gen_keys[0]])):
                timestep_label = [cont_thermal[g][t] for g in cont_gen_keys]
                cont_day_labels.append(timestep_label)
            
            case_labels.append(cont_day_labels)
    
    case_tensor = torch.tensor(case_labels, dtype=torch.float32)
    all_labels.append(case_tensor)
    print(f"  -> {case_tensor.shape} (includes base + all available contingencies)")

# Pad all tensors along generator dimension so they align
max_gens = max(t.shape[2] for t in all_labels)
for i in range(len(all_labels)):
    pad_size = max_gens - all_labels[i].shape[2]
    if pad_size > 0:
        all_labels[i] = torch.nn.functional.pad(all_labels[i], (0, pad_size))

# Concatenate all cases
mega_tensor = torch.cat(all_labels, dim=0)

os.makedirs("labels", exist_ok=True)
torch.save(mega_tensor, "labels/mega_labels_with_contingencies.pt")

print(f"\nSaved labels/mega_labels_with_contingencies.pt with shape {mega_tensor.shape}")
print(f"Expected shape: [~6570, 36, max_generators]")
print(f"  - ~1095 days × 6 scenarios (1 base + 5 contingencies)")
print(f"  - 36 timesteps")
print(f"  - Padded to {max_gens} generators")