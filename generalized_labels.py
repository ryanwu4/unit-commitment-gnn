import os
import json
import torch

base_folders = ["data/case14_solutions", "data/case57_solutions", "data/case118_solutions"]

all_labels = []

for folder_path in base_folders:
    print(f"Processing {folder_path}...")
    
    case_labels = []
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".json"):
            continue
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        thermal = data["Is on"]
        gen_keys = sorted(thermal.keys(), key=lambda x: int(x[1:]))
        
        day_labels = []
        for t in range(len(thermal[gen_keys[0]])):
            timestep_label = [thermal[g][t] for g in gen_keys]
            day_labels.append(timestep_label)
        
        case_labels.append(day_labels)
    
    case_tensor = torch.tensor(case_labels, dtype=torch.float32)
    all_labels.append(case_tensor)
    print(f"  -> {case_tensor.shape}")

# Pad all tensors along generator dimension so they align
max_gens = max(t.shape[2] for t in all_labels)
for i in range(len(all_labels)):
    pad_size = max_gens - all_labels[i].shape[2]
    if pad_size > 0:
        all_labels[i] = torch.nn.functional.pad(all_labels[i], (0, pad_size))

mega_tensor = torch.cat(all_labels, dim=0)

os.makedirs("labels", exist_ok=True)
torch.save(mega_tensor, "labels/mega_labels.pt")

print(f"Saved labels/mega_labels.pt with shape {mega_tensor.shape}")
