import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

graphs = torch.load("graphs/all_hetero_graphs_normalized_114.pt")
labels = torch.load("labels/case118_labels.pt").float()

def flatten_graph_nodes(g):
    return np.concatenate([x.cpu().numpy().flatten() for x in g.x_dict.values()])

X = np.stack([flatten_graph_nodes(g) for g in graphs])
y = labels.reshape(len(labels), -1).cpu().numpy()  # flatten labels

X = SimpleImputer(strategy='constant', fill_value=0).fit_transform(X)
X = StandardScaler().fit_transform(X)

pca_components = 50  # reduce features for speed
pca = PCA(n_components=pca_components)
X_reduced = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42
)

def fit_ridge_column(X, y_col, alpha=1.0):
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y_col)
    return model

n_outputs = y_train.shape[1]
models = Parallel(n_jobs=-1)(
    delayed(fit_ridge_column)(X_train, y_train[:, i]) for i in range(n_outputs)
)

train_preds = np.column_stack([m.predict(X_train) for m in models])
test_preds  = np.column_stack([m.predict(X_test) for m in models])

train_mse = np.mean((train_preds - y_train)**2)
test_mse  = np.mean((test_preds - y_test)**2)

train_acc = ((train_preds > 0.5) == (y_train > 0.5)).mean()
test_acc  = ((test_preds > 0.5) == (y_test > 0.5)).mean()

print(f"Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
print(f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
