"""Preprocessing module."""

import torch
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

from MARBLE import geometry as g
from MARBLE import utils


def construct_dataset(
    anchor,
    vector,
    label=None,
    mask=None,
    k=20,
    spacing=0.0,
    number_of_resamples=1,
    seed=None,
    number_of_eigenvectors=None,
):

    """Construct PyG dataset from node positions and features.

    Args:
        anchor: matrix or list of matrices with positions of points
        vector: matrix or list of matrices with feature values for each point
        label: any additional data labels used for plotting only
        mask: boolean array, that will be forced to be close (default is None)
        graph_type: type of nearest-neighbours graph: cknn (default), knn or radius
        k: number of nearest-neighbours to construct the graph
        delta: argument for cknn graph construction to decide the radius for each points.
        frac_geodesic_nb: number of geodesic neighbours to fit the gauges to
        to map to tangent space k*frac_geodesic_nb
        spacing: furthest point sampling spacing (0 disables sampling)
        number_of_resamples: number of furthest point sampling runs to prevent bias (experimental)
        var_explained: fraction of variance explained by the local gauges
        local_gauges: if True, tries to compute local gauges if it can
        seed: Specify for reproducibility in the furthest point sampling.
        metric: metric used to fit proximity graph
        number_of_eigenvectors: int number of eigenvectors to use. Default: None, meaning use all.
    """

    print("\n================ construct_dataset =================")

    # Normalize inputs to lists
    anchor_list = utils.to_list(anchor)
    vector_list = utils.to_list(vector)

    print(f"Number of conditions: {len(anchor_list)}")
    print("\n[Input data]")
    print(" Anchor shapes:", [getattr(a, "shape", None) for a in anchor_list])
    print(" Vector shapes:", [getattr(v, "shape", None) for v in vector_list])

    # Convert to torch tensors
    anchor = [torch.tensor(a).float() for a in anchor_list]
    vector = [torch.tensor(v).float() for v in vector_list]
    num_node_features = vector[0].shape[1] if len(vector) > 0 else 0

    # Labels
    if label is None:
        label = [torch.arange(len(a)) for a in anchor]
    else:
        label = [torch.tensor(lab).float() for lab in utils.to_list(label)]

    # Masks
    if mask is None:
        mask = [torch.zeros(len(a), dtype=torch.bool) for a in anchor]
    else:
        mask = [torch.tensor(m) for m in utils.to_list(mask)]

    if spacing == 0.0:
        number_of_resamples = 1
        print("\n[Sampling]")
        print(" spacing=0.0 => furthest point sampling disabled; number_of_resamples forced to 1")
    else:
        print("\n[Sampling]")
        print(f" spacing={spacing} => furthest point sampling enabled; number_of_resamples={number_of_resamples}")

    print("\n[Graph parameters]")
    print(f" k={k}")

    data_list = []
    total_raw_points = 0
    total_sampled_points = 0

    for i, (a, v, l, m) in enumerate(zip(anchor, vector, label, mask)):
        print(f"\n--- Condition {i}")
        print(f" Raw points: {a.shape[0]} | dim: {a.shape[1] if a.ndim > 1 else 1}")
        total_raw_points += int(a.shape[0])

        for r in range(number_of_resamples):
            if len(a) == 0:
                print(f"  Resample {r}: skipped (empty condition)")
                continue

            # Choose start index
            if seed is None:
                start_idx = torch.randint(low=0, high=len(a), size=(1,))
                start_idx_print = int(start_idx.item())
            else:
                start_idx = 0
                start_idx_print = 0

            print(f"  Resample {r}: start_idx={start_idx_print}")

           # Sampling (disabled when spacing == 0.0)
            
            sample_ind = torch.arange(len(a), device=a.device)
        

            a_, v_, l_, m_ = (
                a[sample_ind],
                v[sample_ind],
                l[sample_ind],
                m[sample_ind],
            )

            print(f"  After sampling: {a_.shape[0]} points")

            total_sampled_points += int(a_.shape[0])

            # Fit graph to point cloud (STEP 1: kNN)
            print("  [Step 1: Graph] fitting graph ...", end="")
            edge_index = g.fit_graph(a_, k=k)
            print(" done")
            print(f"   edges: {edge_index.shape[1]} | edge_weight: None")


            # Define data object
            data_ = Data(
            pos=a_,
            x=v_,
            label=l_,
            mask=m_,
            edge_index=edge_index,
            num_nodes=len(a_),
            num_node_features=num_node_features,
            y=torch.ones(len(a_), dtype=int) * i,
            sample_ind=sample_ind,
        )


            data_list.append(data_)

    # Collate datasets
    print("\n[Batch]")
    print(f" Total raw points: {total_raw_points}")
    print(f" Total sampled points (sum over resamples): {total_sampled_points}")
    print(f" Number of Data objects: {len(data_list)}")

    batch = Batch.from_data_list(data_list)
    batch.degree = k
    batch.number_of_resamples = number_of_resamples

    print(f" Batch nodes: {batch.num_nodes}")
    print(f" Batch edges: {batch.edge_index.shape[1] if hasattr(batch, 'edge_index') else None}")

    # Split into training/validation/test datasets
    print("\n[Split]")
    split = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.1)
    split(batch)
    # Note: RandomNodeSplit adds boolean masks (train/val/test) to batch
    if hasattr(batch, "train_mask"):
        print(f" train_mask: {int(batch.train_mask.sum())} / {batch.train_mask.numel()}")
    if hasattr(batch, "val_mask"):
        print(f" val_mask:   {int(batch.val_mask.sum())} / {batch.val_mask.numel()}")
    if hasattr(batch, "test_mask"):
        print(f" test_mask:  {int(batch.test_mask.sum())} / {batch.test_mask.numel()}")

    print("\n[Geometric objects]")
    print(f" number_of_eigenvectors={number_of_eigenvectors}")

    return _compute_geometric_objects(
        batch,
        number_of_eigenvectors=number_of_eigenvectors,
    )



def _compute_geometric_objects(data, number_of_eigenvectors=None):

    """
    Compute geometric objects used later: local gauges, Levi-Civita connections
    gradient kernels, scalar and connection laplacians.
    """

    print("\n================ geometric objects =================")

    n, dim_emb = data.pos.shape
    dim_signal = data.x.shape[1]

    print("\n[Geometry]")
    print(f" Nodes: {n}")
    print(f" Embedding dimension: {dim_emb}")
    print(f" Signal dimension: {dim_signal}")

    print("\n[Local gauges] disabled -> using global identity gauges")
    gauges = torch.eye(dim_emb).repeat(n, 1, 1)

    print("\n[Step 4: Diffusion] computing Laplacian ...", end="")
    L = g.compute_laplacian(data)
    print(" done")

    print("\n[Step 3: Kernels/Connections] skipped (simplified geometry)")
    data.kernels = []
    data.Lc = None

    print("\n[Step 2: PCA / spectral]")
    if number_of_eigenvectors is None:
        print(" Computing FULL spectrum (may take long)")
    else:
        print(f" Computing top-{number_of_eigenvectors} eigenvectors")

    data.L = g.compute_eigendecomposition(L, k=number_of_eigenvectors)
    data.gauges = gauges
    data.local_gauges = False


    print("\n================ dataset ready =================")
    print(f" data.pos: {tuple(data.pos.shape)}")
    print(f" data.x: {tuple(data.x.shape)}")
    print(f" data.edge_index: {tuple(data.edge_index.shape)}")
    print(f" has Laplacian: {data.L is not None}")
    print(f" local_gauges: {data.local_gauges}")
    if isinstance(data.L, tuple) and len(data.L) == 2:
        print(f" L eigenvectors: {tuple(data.L[1].shape)}")
    print("=================================================\n")

    return data
