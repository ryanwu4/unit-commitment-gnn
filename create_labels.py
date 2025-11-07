import os
import json
import torch

folder_path = "data/case118_solutions"

all_labels = [] 

for filename in sorted(os.listdir(folder_path)):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    thermal = data["Is on"]
    
    gen_keys = sorted(thermal.keys(), key=lambda x: int(x[1:]))
    
    day_labels = []
    for t in range(len(thermal[gen_keys[0]])):
        timestep_label = [thermal[g][t] for g in gen_keys]
        day_labels.append(timestep_label)
    
    all_labels.append(day_labels)

labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
print(labels_tensor.shape) 
torch.save(labels_tensor, "labels/case118_labels.pt")
