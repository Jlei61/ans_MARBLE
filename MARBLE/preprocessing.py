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
    graph_type="cknn",
    k=20,
    delta=1.0,
    frac_geodesic_nb=1.5,
    spacing=0.0,
    number_of_resamples=1,
    var_explained=0.9,
    local_gauges=False,
    seed=None,
    metric="euclidean",
    number_of_eigenvectors=None,
    memory_efficient=False,
    use_cuda=False,
):
    """Construct PyG dataset from node positions and features.

    Args:
        pos: matrix with position of points
        features: matrix with feature values for each point
        labels: any additional data labels used for plotting only
        mask: boolean array, that will be forced to be close (default is None)
        graph_type: type of nearest-neighbours graph: cknn (default), knn or radius
        k: number of nearest-neighbours to construct the graph
        delta: argument for cknn graph construction to decide the radius for each points.
        frac_geodesic_nb: number of geodesic neighbours to fit the gauges to
        to map to tangent space k*frac_geodesic_nb
        stop_crit: stopping criterion for furthest point sampling
        number_of_resamples: number of furthest point sampling runs to prevent bias (experimental)
        var_explained: fraction of variance explained by the local gauges
        local_gauges: is True, it will try to compute local gauges if it can (signal dim is > 2,
            embedding dimension is > 2 or dim embedding is not dim of manifold)
        seed: Specify for reproducibility in the furthest point sampling.
              The default is None, which means a random starting vertex.
        metric: metric used to fit proximity graph
        number_of_eigenvectors: int number of eigenvectors to use. Default: None, meaning use all.
        memory_efficient: Whether to use memory-efficient furthest point sampling. Default: False.
              If True, uses O(N) memory algorithm instead of O(N²) for large datasets.
              The classic algorithm computes the full pairwise distance matrix while
              the memory-efficient version processes distances in batches.
              Both algorithms should yield equivalent results, but the memory-efficient
              version is recommended for large datasets.
        use_cuda: Whether to use CUDA GPU acceleration when available. Default: False.
              This can significantly speed up computations for large datasets,
              especially when combined with memory_efficient=True.
    """
    # Check CUDA availability
    cuda_available = torch.cuda.is_available() if use_cuda else False
    device = torch.device("cuda" if cuda_available else "cpu")
    
    # Function to move tensors to the appropriate device
    def to_device(tensor_list):
        return [t.to(device) for t in tensor_list]

    anchor = [torch.tensor(a, dtype=torch.float32) for a in utils.to_list(anchor)]
    vector = [torch.tensor(v, dtype=torch.float32) for v in utils.to_list(vector)]
    
    if cuda_available:
        anchor = to_device(anchor)
        vector = to_device(vector)
    
    num_node_features = vector[0].shape[1]

    if label is None:
        label = [torch.arange(len(a), device=a.device) for a in anchor]
    else:
        label = [torch.tensor(lab, dtype=torch.float32) for lab in utils.to_list(label)]
        if cuda_available:
            label = to_device(label)

    if mask is None:
        mask = [torch.zeros(len(a), dtype=torch.bool, device=a.device) for a in anchor]
    else:
        mask = [torch.tensor(m, dtype=torch.bool) for m in utils.to_list(mask)]
        if cuda_available:
            mask = to_device(mask)

    if spacing == 0.0:
        number_of_resamples = 1

    data_list = []
    for i, (a, v, l, m) in enumerate(zip(anchor, vector, label, mask)):
        for _ in range(number_of_resamples):
            if len(a) != 0:
                # even sampling of points
                if seed is None:
                    start_idx = torch.randint(low=0, high=len(a), size=(1,), device=a.device)
                else:
                    start_idx = 0

                # Choose appropriate sampling method based on parameters
                if cuda_available and (memory_efficient or len(a) > 10000):
                    # Use CUDA-accelerated version for large datasets
                    sample_ind, _ = g.cuda_memory_efficient_fps(a, spacing=spacing, start_idx=start_idx)
                elif memory_efficient:
                    sample_ind, _ = g.memory_efficient_fps(a, spacing=spacing, start_idx=start_idx)
                else:
                    sample_ind, _ = g.furthest_point_sampling(a, spacing=spacing, start_idx=start_idx)
                    
                sample_ind, _ = torch.sort(sample_ind)  # this will make postprocessing easier
                a_, v_, l_, m_ = (
                    a[sample_ind],
                    v[sample_ind],
                    l[sample_ind],
                    m[sample_ind],
                )

                # fit graph to point cloud
                edge_index, edge_weight = g.fit_graph(
                    a_, graph_type=graph_type, par=k, delta=delta, metric=metric
                )

                # define data object
                data_ = Data(
                    pos=a_,
                    x=v_,
                    label=l_,
                    mask=m_,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    num_nodes=len(a_),
                    num_node_features=num_node_features,
                    y=torch.ones(len(a_), dtype=int, device=a_.device) * i,
                    sample_ind=sample_ind,
                )

                data_list.append(data_)

    # collate datasets
    batch = Batch.from_data_list(data_list)
    batch.degree = k
    batch.number_of_resamples = number_of_resamples

    # split into training/validation/test datasets
    split = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.1)
    split(batch)

    return _compute_geometric_objects(
        batch,
        local_gauges=local_gauges,
        n_geodesic_nb=k * frac_geodesic_nb,
        var_explained=var_explained,
        number_of_eigenvectors=number_of_eigenvectors,
        use_cuda=cuda_available,
    )


def _compute_geometric_objects(
    data,
    n_geodesic_nb=10,
    var_explained=0.9,
    local_gauges=False,
    number_of_eigenvectors=None,
    use_cuda=False,
):
    """
    Compute geometric objects used later: local gauges, Levi-Civita connections
    gradient kernels, scalar and connection laplacians.

    Args:
        data: pytorch geometric data object
        n_geodesic_nb: number of geodesic neighbours to fit the tangent spaces to
        var_explained: fraction of variance explained by the local gauges
        local_gauges: whether to use local or global gauges
        number_of_eigenvectors: int number of eigenvectors to use. Default: None, meaning use all.
        use_cuda: Whether to use CUDA GPU acceleration when available. Default: False.

    Returns:
        data: pytorch geometric data object with the following new attributes
        kernels (list of d (nxn) matrices): directional kernels
        L (nxn matrix): scalar laplacian
        Lc (ndxnd matrix): connection laplacian
        gauges (nxdxd): local gauges at all points
        par (dict): updated dictionary of parameters
        local_gauges: whether to use local gauges

    """
    # Determine device based on use_cuda parameter and data location
    cuda_available = torch.cuda.is_available() and use_cuda
    device = data.x.device
    
    # If data is not on CUDA but CUDA is available and requested, move it
    if cuda_available and device.type != 'cuda':
        device = torch.device('cuda')
        data = data.to(device)
    
    n, dim_emb = data.pos.shape
    dim_signal = data.x.shape[1]
    print(f"\n---- Embedding dimension: {dim_emb}", end="")
    print(f"\n---- Signal dimension: {dim_signal}", end="")
    print(f"\n---- Using device: {device}", end="")

    # disable vector computations if 1) signal is scalar or 2) embedding dimension
    # is <= 2. In case 2), either M=R^2 (manifold is whole space) or case 1).
    if dim_signal == 1:
        print("\nSignal dimension is 1, so manifold computations are disabled!")
        local_gauges = False
    if dim_emb <= 2:
        print("\nEmbedding dimension <= 2, so manifold computations are disabled!")
        local_gauges = False
    if dim_emb != dim_signal:
        print("\nEmbedding dimension /= signal dimension, so manifold computations are disabled!")

    if local_gauges:
        try:
            # Use GPU for gauge computation if available
            if cuda_available:
                print("\n---- Computing tangent spaces with CUDA acceleration...", end="")
                # Note: Some parts of gauge computation might still use CPU due to implementation specifics
                gauges, Sigma = g.compute_gauges(data, n_geodesic_nb=n_geodesic_nb, use_cuda=cuda_available)
            else:
                gauges, Sigma = g.compute_gauges(data, n_geodesic_nb=n_geodesic_nb)
        except Exception as exc:
            raise Exception(
                "\nCould not compute gauges (possibly data is too sparse or the \
                  number of neighbours is too small)"
            ) from exc
    else:
        gauges = torch.eye(dim_emb, device=device).repeat(n, 1, 1)

    # Compute Laplacian with GPU acceleration when available
    print("\n---- Computing Laplacian operator...", end="")
    L = g.compute_laplacian(data)

    if local_gauges:
        data.dim_man = g.manifold_dimension(Sigma, frac_explained=var_explained)
        print(f"---- Manifold dimension: {data.dim_man}")

        gauges = gauges[:, :, : data.dim_man]
        
        print("\n---- Computing connections...", end="")
        R = g.compute_connections(data, gauges, use_cuda=cuda_available)

        print("\n---- Computing kernels...", end="")
        # Gradient operator computation can be accelerated on GPU
        kernels = g.gradient_op(data.pos, data.edge_index, gauges)
        kernels = [utils.tile_tensor(K, data.dim_man) for K in kernels]
        kernels = [K * R for K in kernels]

        Lc = g.compute_connection_laplacian(data, R)

    else:
        print("\n---- Computing kernels...", end="")
        kernels = g.gradient_op(data.pos, data.edge_index, gauges)
        Lc = None

    if number_of_eigenvectors is None:
        print(
            """\n---- Computing full spectrum...
              (if this takes too long, then run construct_dataset()
              with number_of_eigenvectors specified) """,
            end="",
        )
    else:
        print(
            f"\n---- Computing spectrum with {number_of_eigenvectors} eigenvectors...",
            end="",
        )
    
    # Eigendecomposition can benefit greatly from CUDA acceleration
    L = g.compute_eigendecomposition(L, k=number_of_eigenvectors, use_cuda=cuda_available)
    Lc = g.compute_eigendecomposition(Lc, k=number_of_eigenvectors, use_cuda=cuda_available)

    data.kernels = [
        utils.to_SparseTensor(K.coalesce().indices(), value=K.coalesce().values()) for K in kernels
    ]
    data.L, data.Lc, data.gauges, data.local_gauges = L, Lc, gauges, local_gauges

    return data
