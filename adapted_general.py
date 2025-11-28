import torch
from torch import nn
from torch_geometric.nn import HEATConv
from numpy import random
from torch import nn, optim
import time


class HEATEncoder(nn.Module):
    def __init__(self, hidden_dim=128, num_heads=4, edge_emb_dim=16, 
                 dropout=0.4, node_feature_dims={'generator':24,'bus':128,'reserve':128}):
        super().__init__()
        
        # Learnable structural embeddings for bus and reserve nodes
        self.bus_type_embedding = nn.Parameter(torch.randn(hidden_dim))
        self.reserve_type_embedding = nn.Parameter(torch.randn(hidden_dim))
        
        self.node_projections = nn.ModuleDict({
            ntype: nn.Linear(feat_dim, hidden_dim)
            for ntype, feat_dim in node_feature_dims.items()
        })

        self.edge_type_mapping = {
            ('generator', 'produces_at', 'bus'): 0,
            ('bus', 'served_by', 'generator'): 1,
            ('bus', 'transmission', 'bus'): 2,
            ('reserve', 'backed_by', 'generator'): 3
        }
        self.node_type_mapping = {ntype: i for i, ntype in enumerate(['generator', 'bus', 'reserve'])}
        self.num_node_types = len(self.node_type_mapping)
        self.num_edge_types = len(self.edge_type_mapping)

        self.heat1 = HEATConv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            num_node_types=self.num_node_types, num_edge_types=self.num_edge_types,
            edge_type_emb_dim=edge_emb_dim, edge_dim=2, edge_attr_emb_dim=edge_emb_dim,
            heads=num_heads, concat=False
        )
        self.heat2 = HEATConv(
            in_channels=hidden_dim, out_channels=hidden_dim,
            num_node_types=self.num_node_types, num_edge_types=self.num_edge_types,
            edge_type_emb_dim=edge_emb_dim, edge_dim=2, edge_attr_emb_dim=edge_emb_dim,
            heads=num_heads, concat=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_proj = {ntype: self.node_projections[ntype](x) for ntype, x in x_dict.items()}

        node_features_list, node_type_list = [], []
        node_counts, offsets = {}, {}
        current_offset = 0
        
        for ntype in ['generator', 'bus', 'reserve']:
            if ntype in x_proj:
                feats = x_proj[ntype]
                node_features_list.append(feats)
                node_type_list.append(torch.full(
                    (feats.size(0),), self.node_type_mapping[ntype], 
                    dtype=torch.long, device=feats.device
                ))
                node_counts[ntype] = feats.size(0)
                offsets[ntype] = current_offset
                current_offset += feats.size(0)

        x = torch.cat(node_features_list, dim=0)
        node_type = torch.cat(node_type_list, dim=0)

        edge_index_list, edge_attr_list, edge_type_list = [], [], []
        for etype, etype_id in self.edge_type_mapping.items():
            if etype in edge_index_dict:
                e_index = edge_index_dict[etype]
                e_attr = edge_attr_dict.get(etype, torch.zeros(e_index.size(1), 2, device=e_index.device))
                edge_index_list.append(e_index)
                edge_attr_list.append(e_attr)
                edge_type_list.append(torch.full((e_index.size(1),), etype_id, dtype=torch.long, device=e_index.device))

        edge_index = torch.cat(edge_index_list, dim=1)
        edge_attr = torch.cat(edge_attr_list, dim=0)
        edge_type = torch.cat(edge_type_list, dim=0)

        x = self.heat1(x, edge_index, node_type, edge_type, edge_attr)
        x1 = torch.relu(x)
        x = self.heat2(x1, edge_index, node_type, edge_type, edge_attr)
        x_out = torch.relu(x) + x1
        
        if self.dropout.p > 0:
            x_out = self.dropout(x_out)

        return x_out, node_counts, offsets


class FastTemporalModel(nn.Module):
    """
    Two temporal layer(s) options:
    - '1d_conv': Fast - Slightly Weaker
    - 'gru': Medium - Stronger
    """
    def __init__(self, encoder, hidden_dim=128, output_dim=1, T=36, 
                 temporal_method='1d_conv', dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.T = T
        self.temporal_method = temporal_method
        
        # Temporal feature projections (bus load + reserve requirement at each timestep)
        self.bus_temporal_proj = nn.Linear(1, hidden_dim)
        self.reserve_temporal_proj = nn.Linear(1, hidden_dim)
        
        # 1D Conv layers
        if temporal_method == '1d_conv':
            self.temporal = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            
        # GRU layers
        elif temporal_method == 'gru':
            self.temporal = nn.GRU(
                hidden_dim, 
                hidden_dim, 
                num_layers=2,
                batch_first=True,
                dropout=dropout if dropout > 0 else 0,
                bidirectional=False
            )
        
        else:
            raise ValueError(f"temporal_method must be '1d_conv' or 'gru', got {temporal_method}")
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, graph, node_counts):
        """
        Steps:
        1. Get structural embeddings for bus/reserve nodes (no temporal leakage)
        2. Run GNN once on structural features only
        3. Build temporal sequences by combining structural + time-varying features
        4. Apply temporal layers (1D Conv or GRU)
        5. Decode to predictions
        
        Shape flow:
        - After GNN: [total_nodes, hidden_dim]
        - Temporal sequences: [total_nodes, T, hidden_dim]
        - After temporal model: [total_nodes, T, hidden_dim]
        - After decoder: [total_nodes, T, 1]
        - Final output: [T, n_gen, 1]
        """
        device = next(self.parameters()).device
        n_gen = node_counts['generator']
        n_bus = node_counts['bus']
        n_reserve = node_counts['reserve']
        
        # Extract features from graph
        bus_feats = graph.x_dict['bus']      # [n_bus, T]
        res_feats = graph.x_dict['reserve']  # [n_reserve, T]
        gen_feats_raw = graph.x_dict['generator']  # [n_gen, 24]
        
        # Create structural embeddings (no temporal leakage!)
        bus_struct = self.encoder.bus_type_embedding.unsqueeze(0).expand(n_bus, -1)  # [n_bus, hidden_dim]
        res_struct = self.encoder.reserve_type_embedding.unsqueeze(0).expand(n_reserve, -1)  # [n_reserve, hidden_dim]
        
        # Run GNN once on structural features only
        x_dict_static = {
            'generator': gen_feats_raw,  # [n_gen, 24] - rich static features
            'bus': bus_struct,           # [n_bus, hidden_dim] - structural identity
            'reserve': res_struct        # [n_reserve, hidden_dim] - structural identity
        }

        # Get structural embeddings from GNN
        z_graph, _, offsets = self.encoder(x_dict_static, graph.edge_index_dict, graph.edge_attr_dict)
        
        # Extract generator structural embeddings (only computed once!)
        gen_emb_static = z_graph[offsets['generator']:offsets['generator']+n_gen]  # [n_gen, hidden_dim]
        bus_emb_static = z_graph[offsets['bus']:offsets['bus']+n_bus]  # [n_bus, hidden_dim]
        res_emb_static = z_graph[offsets['reserve']:offsets['reserve']+n_reserve]  # [n_reserve, hidden_dim]
        
        # Build temporal sequences efficiently (vectorized over time)
        bus_temporal = bus_feats.unsqueeze(-1)  # [n_bus, T, 1]
        res_temporal = res_feats.unsqueeze(-1)  # [n_reserve, T, 1]
        
        # Project temporal features
        bus_temporal_emb = self.bus_temporal_proj(bus_temporal)  # [n_bus, T, hidden_dim]
        res_temporal_emb = self.reserve_temporal_proj(res_temporal)  # [n_reserve, T, hidden_dim]
        
        # Combine structural + temporal for bus and reserve
        bus_emb_seq = bus_emb_static.unsqueeze(1) + bus_temporal_emb  # [n_bus, T, hidden_dim]
        res_emb_seq = res_emb_static.unsqueeze(1) + res_temporal_emb  # [n_reserve, T, hidden_dim]
        
        # Generators: expand static embedding across time (no temporal features for generators)
        gen_emb_seq = gen_emb_static.unsqueeze(1).expand(-1, self.T, -1)  # [n_gen, T, hidden_dim]
        
        # Concatenate all nodes
        z_seq = torch.cat([gen_emb_seq, bus_emb_seq, res_emb_seq], dim=0)  # [total_nodes, T, hidden_dim]
        
        # Apply temporal model
        if self.temporal_method == '1d_conv':
            # Conv1d expects [batch, channels, length]
            z_seq_t = z_seq.permute(0, 2, 1)  # [total_nodes, hidden_dim, T]
            z_temporal = self.temporal(z_seq_t)  # [total_nodes, hidden_dim, T]
            z_temporal = z_temporal.permute(0, 2, 1)  # [total_nodes, T, hidden_dim]
            
        elif self.temporal_method == 'gru':
            # GRU expects [batch, seq_len, features]
            z_temporal, _ = self.temporal(z_seq)  # [total_nodes, T, hidden_dim]
        
        # Decode to predictions
        out_seq = self.decoder(z_temporal)  # [total_nodes, T, 1]
        out_seq = out_seq.permute(1, 0, 2)  # [T, total_nodes, 1]
        
        # Extract only generator predictions
        gen_predictions = out_seq[:, :n_gen, :]  # [T, n_gen, 1]
        
        return gen_predictions, node_counts


def extract_graph_metadata(graphs):
    metadata = []
    for g in graphs:
        node_counts = {
            'generator': g.x_dict['generator'].size(0),
            'bus': g.x_dict['bus'].size(0),
            'reserve': g.x_dict['reserve'].size(0)
        }
        metadata.append(node_counts)
    return metadata


def move_graph_to_device(graph, device):
    for key in graph.x_dict:
        graph.x_dict[key] = graph.x_dict[key].float().to(device)
    for key in graph.edge_index_dict:
        graph.edge_index_dict[key] = graph.edge_index_dict[key].to(device)
    for key in graph.edge_attr_dict:
        graph.edge_attr_dict[key] = graph.edge_attr_dict[key].float().to(device)
    return graph


def binary_accuracy(preds, targets, threshold=0.5):
    pred_labels = (preds > threshold).float()
    return (pred_labels == targets).float().mean()


# GPU device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load data
graphs = torch.load("graphs/mega_train_with_contingencies.pt")
labels_tensor = torch.load("labels/mega_labels_with_contingencies.pt").float()
num_days = len(graphs)
print(f"Number of graphs: {num_days}")
print(f"Labels tensor shape: {labels_tensor.shape}")

graph_metadata = extract_graph_metadata(graphs)
print(f"Example graph structure: {graph_metadata[0]}")

# GPU transfer
graphs = [move_graph_to_device(g, device) for g in graphs]
labels_tensor = labels_tensor.to(device)

# Random split (better for seasonal patterns than chronological)
indices = list(range(num_days))
random.seed(42)
random.shuffle(indices)
train_ratio = 0.8
train_size = int(train_ratio * num_days)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

train_graphs = [graphs[i] for i in train_idx]
train_metadata = [graph_metadata[i] for i in train_idx]
train_labels = labels_tensor[train_idx]
test_graphs = [graphs[i] for i in test_idx]
test_metadata = [graph_metadata[i] for i in test_idx]
test_labels = labels_tensor[test_idx]

T = 36
output_dim = 1
hidden_dim = 128
num_epochs = 100

# Choose temporal method
TEMPORAL_METHOD = '1d_conv'  # Options: '1d_conv' or 'gru'

print(f"\nTraining with temporal method: {TEMPORAL_METHOD}")

encoder = HEATEncoder(hidden_dim=hidden_dim)
model = FastTemporalModel(
    encoder=encoder,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    T=T,
    temporal_method=TEMPORAL_METHOD
).to(device)

print("\nModel Architecture:")
print(f"  - GNN: 2 HEAT layers")
print(f"  - Structural embeddings: Learnable for bus/reserve nodes")
print(f"  - Temporal: {TEMPORAL_METHOD}")
print(f"  - Decoder: 2-layer MLP")
print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

optimizer = optim.Adam(model.parameters(), lr=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.BCEWithLogitsLoss()

# Training loop
print("Starting training...")
epoch_times = []

for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    model.train()
    total_train_loss = 0.0

    for graph, labels_day, metadata in zip(train_graphs, train_labels, train_metadata):
        optimizer.zero_grad()
        
        preds_gen, _ = model(graph, metadata)
        
        n_gen = metadata['generator']
        labels_gen = labels_day[:, :n_gen].unsqueeze(-1).to(device)
        
        loss = criterion(preds_gen, labels_gen)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_graphs)
    scheduler.step()
    
    epoch_time = time.time() - start_time
    epoch_times.append(epoch_time)

    # Validation
    model.eval()
    total_val_loss = 0.0
    val_accs = []

    with torch.no_grad():
        for val_graph, val_labels_day, val_metadata in zip(test_graphs, test_labels, test_metadata):
            val_preds_gen, _ = model(val_graph, val_metadata)
            
            n_gen = val_metadata['generator']
            val_labels_gen = val_labels_day[:, :n_gen].unsqueeze(-1).to(device)
            
            val_loss = criterion(val_preds_gen, val_labels_gen)
            total_val_loss += val_loss.item()
            
            val_accs.append(binary_accuracy(torch.sigmoid(val_preds_gen), val_labels_gen))

    avg_val_loss = total_val_loss / len(test_graphs)
    avg_val_acc = sum(val_accs) / len(val_accs)

    if epoch % 10 == 0 or epoch == 1:
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        est_total_time = avg_epoch_time * num_epochs / 60
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s | Est Total: {est_total_time:.1f}min")

# Final evaluation
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

model.eval()
with torch.no_grad():
    test_losses = []
    test_accs = []

    for g, labels_day, metadata in zip(test_graphs, test_labels, test_metadata):
        preds_gen, _ = model(g, metadata)
        n_gen = metadata['generator']
        labels_gen = labels_day[:, :n_gen].unsqueeze(-1).to(device)

        test_loss = criterion(preds_gen, labels_gen)
        test_losses.append(test_loss.item())

        test_accs.append(binary_accuracy(torch.sigmoid(preds_gen), labels_gen))

    final_bce_loss = sum(test_losses) / len(test_losses)
    final_acc = sum(test_accs) / len(test_accs)

print(f"\nMethod: {TEMPORAL_METHOD}")
print(f"Test BCE Loss: {final_bce_loss:.4f}")
print(f"Test Accuracy: {final_acc:.4f}")
print(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.2f}s")
print(f"Total training time: {sum(epoch_times)/60:.2f} minutes")

torch.save(model.state_dict(), f"trained_{TEMPORAL_METHOD}_model_conts_case.pt")
print(f"\nModel saved to trained_{TEMPORAL_METHOD}_model_conts_case.pt")