import torch
from torch import nn
from torch_geometric.nn import HEATConv
from numpy import random
from torch import nn, optim


class HEATEncoder(nn.Module):
    def __init__(self, 
                 hidden_dim=128, 
                 num_heads=4, 
                 edge_emb_dim=16, 
                 dropout=0.4, 
                 node_feature_dims={'generator':39,'bus':16,'reserve':16}):
        """
        node_feature_dims: dict of input feature sizes per node type
        """
        super().__init__()

        # Node projections
        self.node_projections = nn.ModuleDict({
            ntype: nn.Linear(feat_dim, hidden_dim)
            for ntype, feat_dim in node_feature_dims.items()
        })

        # Mappings
        self.edge_type_mapping = {
            ('generator', 'produces_at', 'bus'): 0,
            ('bus', 'served_by', 'generator'): 1,
            ('bus', 'transmission', 'bus'): 2,
            ('reserve', 'backed_by', 'generator'): 3
            #('generator', 'supports', 'reserve'): 4
        }
        self.node_type_mapping = {ntype: i for i, ntype in enumerate(node_feature_dims.keys())}

        self.num_node_types = len(self.node_type_mapping)
        self.num_edge_types = len(self.edge_type_mapping)

        self.heat1 = HEATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_node_types=self.num_node_types,
            num_edge_types=self.num_edge_types,
            edge_type_emb_dim=edge_emb_dim,
            edge_dim=2,
            edge_attr_emb_dim=edge_emb_dim,
            heads=num_heads,
            concat=False
        )
        self.heat2 = HEATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_node_types=self.num_node_types,
            num_edge_types=self.num_edge_types,
            edge_type_emb_dim=edge_emb_dim,
            edge_dim=2,
            edge_attr_emb_dim=edge_emb_dim,
            heads=num_heads,
            concat = False
        )
        self.heat3 = HEATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            num_node_types=self.num_node_types,
            num_edge_types=self.num_edge_types,
            edge_type_emb_dim=edge_emb_dim,
            edge_dim=2,
            edge_attr_emb_dim=edge_emb_dim,
            heads=num_heads,
            concat = False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_proj = {ntype: self.node_projections[ntype](x) for ntype, x in x_dict.items()}

        node_features_list, node_type_list, node_counts = [], [], {}
        for ntype in x_proj:
            feats = x_proj[ntype]
            node_features_list.append(feats)
            node_type_list.append(torch.full(
                (feats.size(0),), self.node_type_mapping[ntype], dtype=torch.long, device=feats.device
            ))
            node_counts[ntype] = feats.size(0)

        x = torch.cat(node_features_list, dim=0)
        node_type = torch.cat(node_type_list, dim=0)

        edge_index_list, edge_attr_list, edge_type_list = [], [], []
        for etype, etype_id in self.edge_type_mapping.items():
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
        x2 = self.heat2(x1, edge_index, node_type, edge_type, edge_attr)
        x2 = torch.relu(x2) + x1  
        x3 = self.heat3(x2, edge_index, node_type, edge_type, edge_attr)
        x_out = torch.relu(x3) + x2 
        if self.dropout.p > 0:
            x_out = self.dropout(x_out)

        return x_out, node_counts, node_type
    

class HEATTemporalFast(nn.Module):
    def __init__(self, encoder, hidden_dim=128, output_dim=1, T=36, time_emb_dim=8, num_heads=4, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.T = T

        self.time_embed = nn.Embedding(T, time_emb_dim)

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim + time_emb_dim,
                                          num_heads=num_heads,
                                          batch_first=True,
                                          dropout=dropout)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + time_emb_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        device = next(self.parameters()).device

        gen_feats = self.encoder.node_projections['generator'](graph.x_dict['generator'])  # [num_gen, hidden_dim]

        bus_feats = graph.x_dict['bus']      
        res_feats = graph.x_dict['reserve']  

        bus_static = bus_feats[:, self.T:]    # anchor features
        res_static = res_feats[:, self.T:]

        z_seq_list = []
        for t in range(self.T):
            bus_hour = bus_feats[:, t].unsqueeze(-1)
            res_hour = res_feats[:, t].unsqueeze(-1)

            bus_input = torch.cat([bus_hour, bus_static], dim=-1)
            res_input = torch.cat([res_hour, res_static], dim=-1)

            bus_emb = self.encoder.node_projections['bus'](bus_input)
            res_emb = self.encoder.node_projections['reserve'](res_input)

            z_t = torch.cat([gen_feats, bus_emb, res_emb], dim=0)
            z_seq_list.append(z_t.unsqueeze(1))  
 
        z_seq = torch.cat(z_seq_list, dim=1)  

        # time embeddings
        hour_embs = self.time_embed(torch.arange(self.T, device=device))
        hour_embs = hour_embs.unsqueeze(0).expand(z_seq.size(0), -1, -1)
        z_seq = torch.cat([z_seq, hour_embs], dim=-1) 

        # --- temporal attention
        attn_out, _ = self.attn(z_seq, z_seq, z_seq)
        attn_out = self.dropout(attn_out)

        out_seq = self.decoder(attn_out).permute(1, 0, 2)  
        return out_seq


# --- Set device ---
device = torch.device("cuda" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

graphs = torch.load("graphs/all_hetero_graphs_normalized_114.pt")
labels_tensor = torch.load("labels/case118_labels.pt").float()
num_days = len(graphs)
print(f"Number of time steps: {num_days}")

def move_graph_to_device(graph, device):
    for key in graph.x_dict:
        graph.x_dict[key] = graph.x_dict[key].float().to(device)
    for key in graph.edge_index_dict:
        graph.edge_index_dict[key] = graph.edge_index_dict[key].to(device)
    for key in graph.edge_attr_dict:
        graph.edge_attr_dict[key] = graph.edge_attr_dict[key].float().to(device)
    return graph

graphs = [move_graph_to_device(g, device) for g in graphs]
labels_tensor = labels_tensor.to(device)

indices = list(range(num_days))
random.seed(42)
random.shuffle(indices)
train_ratio = 0.8
train_size = int(train_ratio * num_days)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

train_graphs = [graphs[i] for i in train_idx]
train_labels = labels_tensor[train_idx]
test_graphs = [graphs[i] for i in test_idx]
test_labels = labels_tensor[test_idx]

T = 36
N_gen = 54
output_dim = 1
hidden_dim = 128
time_emb_dim = 8
num_epochs = 200

def build_sequences(graphs, labels, T=T, N_gen=N_gen):
    sequences, labels_seq = [], []
    for day_start in range(0, len(graphs), T):
        if day_start + T > len(graphs):
            break
        day_graphs = graphs[day_start: day_start + T]
        day_labels = labels[day_start: day_start + T, -1, :N_gen].unsqueeze(-1)
        sequences.append(day_graphs)
        labels_seq.append(day_labels)
    return sequences, torch.stack(labels_seq)

train_sequences, train_labels_seq = build_sequences(train_graphs, train_labels)
test_sequences, test_labels_seq = build_sequences(test_graphs, test_labels)

def binary_accuracy(preds, targets, threshold=0.5):
    pred_labels = (preds > threshold).float()
    return (pred_labels == targets).float().mean()

encoder = HEATEncoder(hidden_dim=hidden_dim)
model = HEATTemporalFast(encoder=encoder, hidden_dim=hidden_dim, output_dim=output_dim,
                         T=T, time_emb_dim=time_emb_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(1, num_epochs + 1):
    model.train()
    total_train_loss = 0.0

    for graph, labels_day in zip(train_graphs, train_labels):
        optimizer.zero_grad()
        preds = model(graph)
        labels_gen = labels_day[:, :N_gen].unsqueeze(-1).to(device)
        preds_gen = preds[:, :N_gen, :]
        loss = criterion(preds_gen, labels_gen)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_graphs)
    scheduler.step()

    model.eval()
    total_val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for val_graph, val_labels_day in zip(test_graphs, test_labels):
            val_preds = model(val_graph)
            val_labels_gen = val_labels_day[:, :N_gen].unsqueeze(-1).to(device)
            val_preds_gen = val_preds[:, :N_gen, :]
            val_loss = criterion(val_preds_gen, val_labels_gen)
            total_val_loss += val_loss.item()
            all_preds.append(torch.sigmoid(val_preds_gen))
            all_labels.append(val_labels_gen)

    avg_val_loss = total_val_loss / len(test_graphs)
    val_acc = binary_accuracy(torch.cat(all_preds, dim=0), torch.cat(all_labels, dim=0))

    print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

model.load_state_dict(torch.load("best_model118.pt", map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    test_preds_list = []
    for g in test_sequences:
        test_preds_list.append(model(g).unsqueeze(0))
    test_preds = torch.cat(test_preds_list, dim=0)
    test_labels_gen = test_labels_seq[:, :, :N_gen, 0].unsqueeze(-1).to(device)

    final_bce_loss = criterion(test_preds, test_labels_gen)
    final_acc = binary_accuracy(torch.sigmoid(test_preds), test_labels_gen)

print(f"Final Test BCE Loss: {final_bce_loss:.4f}")
print(f"Final Test Accuracy: {final_acc:.4f}")
