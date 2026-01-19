# geometry.py (simplified)

import torch
from torch_geometric.nn import knn_graph
import torch_geometric.utils as PyGu
import scipy.sparse as sp


def fit_graph(x, k=20):
    edge_index = knn_graph(x, k=k)
    edge_index = PyGu.to_undirected(edge_index)
    return edge_index


def compute_laplacian(data):
    edge_index, edge_weight = PyGu.get_laplacian(
        data.edge_index,
        normalization="rw",
        num_nodes=data.num_nodes,
    )
    return torch.sparse_coo_tensor(edge_index, edge_weight).coalesce()


def compute_eigendecomposition(L, k=32):
    indices, values, size = L.indices(), L.values(), L.size()
    A = sp.coo_array((values, (indices[0], indices[1])), shape=size)
    evals, evecs = sp.linalg.eigsh(A, k=k, which="SM")
    return torch.tensor(evals), torch.tensor(evecs)
