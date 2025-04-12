"""Preprocessing module."""

import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from torch_geometric.data import Batch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit

from MARBLE import geometry as g
from MARBLE import utils


def _process_data_item(
    item_tuple, 
    number_of_resamples, 
    spacing, 
    Sampling, 
    graph_type, 
    k, 
    delta, 
    metric, 
    use_parallel, 
    seed=None
):
    """Process a single data item for parallel execution.
    
    Args:
        item_tuple: Tuple of (index, anchor, vector, label, mask)
        number_of_resamples: Number of resampling runs
        spacing: Stopping criterion for furthest point sampling
        Sampling: Whether to use sampling
        graph_type: Type of graph to construct
        k: Number of neighbors
        delta: Delta parameter for graph construction
        metric: Distance metric
        use_parallel: Whether to use parallel graph fitting
        seed: Random seed
        
    Returns:
        List of processed Data objects
    """
    i, a, v, l, m = item_tuple
    result_data = []
    
    for _ in range(number_of_resamples):
        if len(a) != 0:
            # Apply sampling only if enabled
            if Sampling:
                # even sampling of points
                if seed is None:
                    start_idx = torch.randint(low=0, high=len(a), size=(1,))
                else:
                    start_idx = 0

                sample_ind, _ = g.furthest_point_sampling_memory_efficient(a, spacing=spacing, start_idx=start_idx)
                sample_ind, _ = torch.sort(sample_ind)  # this will make postprocessing easier
                a_, v_, l_, m_ = (
                    a[sample_ind],
                    v[sample_ind],
                    l[sample_ind],
                    m[sample_ind],
                )
            else:
                # Use all points without sampling
                a_, v_, l_, m_ = a, v, l, m
                sample_ind = torch.arange(len(a))

            # fit graph to point cloud
            edge_index, edge_weight = g.fit_graph(
                a_, graph_type=graph_type, par=k, delta=delta, metric=metric, use_parallel=use_parallel
            )

            # define data object
            num_node_features = v.shape[1]
            data_ = Data(
                pos=a_,
                x=v_,
                label=l_,
                mask=m_,
                edge_index=edge_index,
                edge_weight=edge_weight,
                num_nodes=len(a_),
                num_node_features=num_node_features,
                y=torch.ones(len(a_), dtype=int) * i,
                sample_ind=sample_ind,
            )

            result_data.append(data_)
    
    return result_data


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
    use_parallel=True,
    Sampling=False,
    parallel_processing=False,
    n_jobs=None,
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
        spacing: stopping criterion for furthest point sampling
        number_of_resamples: number of furthest point sampling runs to prevent bias (experimental)
                            if set to 0, no resampling will be performed
        var_explained: fraction of variance explained by the local gauges
        local_gauges: is True, it will try to compute local gauges if it can (signal dim is > 2,
            embedding dimension is > 2 or dim embedding is not dim of manifold)
        seed: Specify for reproducibility in the furthest point sampling.
              The default is None, which means a random starting vertex.
        metric: metric used to fit proximity graph
        number_of_eigenvectors: int number of eigenvectors to use. Default: None, meaning use all.
        Sampling: whether to use sampling strategy. Default: False
        parallel_processing: whether to use parallel processing for data preparation. Default: False
        n_jobs: number of parallel jobs. Default: None (use all available cores)
    """

    anchor = [torch.tensor(a).float() for a in utils.to_list(anchor)]
    vector = [torch.tensor(v).float() for v in utils.to_list(vector)]
    num_node_features = vector[0].shape[1]

    if label is None:
        label = [torch.arange(len(a)) for a in utils.to_list(anchor)]
    else:
        label = [torch.tensor(lab).float() for lab in utils.to_list(label)]

    if mask is None:
        mask = [torch.zeros(len(a), dtype=torch.bool) for a in utils.to_list(anchor)]
    else:
        mask = [torch.tensor(m) for m in utils.to_list(mask)]

    # Set number_of_resamples to 1 if Sampling is False
    if not Sampling:
        number_of_resamples = 1

    data_list = []
    
    if parallel_processing and len(anchor) > 1:
        print(f"Using parallel processing with {n_jobs or multiprocessing.cpu_count()} workers")
        # Prepare data for parallel processing
        items = [(i, a, v, l, m) for i, (a, v, l, m) in enumerate(zip(anchor, vector, label, mask))]
        
        # Create a partial function with fixed parameters
        process_func = partial(
            _process_data_item,
            number_of_resamples=number_of_resamples,
            spacing=spacing,
            Sampling=Sampling,
            graph_type=graph_type,
            k=k,
            delta=delta,
            metric=metric,
            use_parallel=use_parallel,
            seed=seed
        )
        
        # Process data in parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(process_func, item) for item in items]
            for future in as_completed(futures):
                try:
                    result = future.result()
                    data_list.extend(result)
                except Exception as exc:
                    print(f"An exception occurred during parallel processing: {exc}")
    else:
        # Original sequential processing
        for i, (a, v, l, m) in enumerate(zip(anchor, vector, label, mask)):
            for _ in range(number_of_resamples):
                if len(a) != 0:
                    # Apply sampling only if enabled
                    if Sampling:
                        # even sampling of points
                        if seed is None:
                            start_idx = torch.randint(low=0, high=len(a), size=(1,))
                        else:
                            start_idx = 0

                        sample_ind, _ = g.furthest_point_sampling_memory_efficient(a, spacing=spacing, start_idx=start_idx)
                        sample_ind, _ = torch.sort(sample_ind)  # this will make postprocessing easier
                        a_, v_, l_, m_ = (
                            a[sample_ind],
                            v[sample_ind],
                            l[sample_ind],
                            m[sample_ind],
                        )
                    else:
                        # Use all points without sampling
                        a_, v_, l_, m_ = a, v, l, m
                        sample_ind = torch.arange(len(a))

                    # fit graph to point cloud
                    edge_index, edge_weight = g.fit_graph(
                        a_, graph_type=graph_type, par=k, delta=delta, metric=metric, use_parallel=use_parallel
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
                        y=torch.ones(len(a_), dtype=int) * i,
                        sample_ind=sample_ind,
                    )

                    data_list.append(data_)

    # collate datasets
    batch = Batch.from_data_list(data_list)
    batch.degree = k
    batch.number_of_resamples = number_of_resamples if Sampling else 0

    # split into training/validation/test datasets
    split = RandomNodeSplit(split="train_rest", num_val=0.1, num_test=0.1)
    split(batch)

    return _compute_geometric_objects(
        batch,
        local_gauges=local_gauges,
        n_geodesic_nb=k * frac_geodesic_nb,
        var_explained=var_explained,
        number_of_eigenvectors=number_of_eigenvectors,
    )


def _compute_geometric_objects(
    data,
    n_geodesic_nb=10,
    var_explained=0.9,
    local_gauges=False,
    number_of_eigenvectors=None,
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

    Returns:
        data: pytorch geometric data object with the following new attributes
        kernels (list of d (nxn) matrices): directional kernels
        L (nxn matrix): scalar laplacian
        Lc (ndxnd matrix): connection laplacian
        gauges (nxdxd): local gauges at all points
        par (dict): updated dictionary of parameters
        local_gauges: whether to use local gauges

    """
    n, dim_emb = data.pos.shape
    dim_signal = data.x.shape[1]
    print(f"\n---- Embedding dimension: {dim_emb}", end="")
    print(f"\n---- Signal dimension: {dim_signal}", end="")

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
            gauges, Sigma = g.compute_gauges(data, n_geodesic_nb=n_geodesic_nb)
        except Exception as exc:
            raise Exception(
                "\nCould not compute gauges (possibly data is too sparse or the \
                  number of neighbours is too small)"
            ) from exc
    else:
        gauges = torch.eye(dim_emb).repeat(n, 1, 1)

    L = g.compute_laplacian(data)

    if local_gauges:
        data.dim_man = g.manifold_dimension(Sigma, frac_explained=var_explained)
        print(f"---- Manifold dimension: {data.dim_man}")

        gauges = gauges[:, :, : data.dim_man]
        R = g.compute_connections(data, gauges)

        print("\n---- Computing kernels ... ", end="")
        kernels = g.gradient_op(data.pos, data.edge_index, gauges)
        kernels = [utils.tile_tensor(K, data.dim_man) for K in kernels]
        kernels = [K * R for K in kernels]

        Lc = g.compute_connection_laplacian(data, R)

    else:
        print("\n---- Computing kernels ... ", end="")
        kernels = g.gradient_op(data.pos, data.edge_index, gauges)
        Lc = None

    if number_of_eigenvectors is None:
        print(
            """\n---- Computing full spectrum ...
              (if this takes too long, then run construct_dataset()
              with number_of_eigenvectors specified) """,
            end="",
        )
    else:
        print(
            f"\n---- Computing spectrum with {number_of_eigenvectors} eigenvectors...",
            end="",
        )
    L = g.compute_eigendecomposition(L, k=number_of_eigenvectors)
    Lc = g.compute_eigendecomposition(Lc, k=number_of_eigenvectors)

    data.kernels = [
        utils.to_SparseTensor(K.coalesce().indices(), value=K.coalesce().values()) for K in kernels
    ]
    data.L, data.Lc, data.gauges, data.local_gauges = L, Lc, gauges, local_gauges

    return data
