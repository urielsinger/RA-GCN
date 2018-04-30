from __future__ import print_function

from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from itertools import islice, takewhile, repeat
import scipy
import scipy.sparse as sp
import pickle
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import bisect
from scipy.spatial.distance import cosine as cosine_dist
import numpy as np
import pickle as pkl
import sys
import networkx as nx
from tqdm import tqdm
import os
from scipy.sparse.linalg.eigen.arpack import eigsh
import tensorflow as tf

"""
All functions are taken verbatim from https://github.com/tkipf/keras-gcn
or https://github.com/tkipf/gcn
"""


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """Load data."""
    print('Loading {} dataset...'.format(dataset_str))

    FILE_PATH = os.path.abspath(__file__)
    DIR_PATH = os.path.dirname(FILE_PATH)
    DATA_PATH = os.path.join(DIR_PATH, f'data/{dataset_str}/')

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("{}ind.{}.{}".format(DATA_PATH, dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("{}ind.{}.test.index".format(DATA_PATH, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, idx_test, idx_val, idx_train


def categorical_crossentropy(preds, labels):
    return np.mean(-np.log(np.extract(labels, preds)))


def accuracy(preds, labels):
    return np.mean(np.equal(np.argmax(labels, 1), np.argmax(preds, 1)))


def evaluate_preds(preds, labels, indices):
    split_loss = list()
    split_acc = list()

    for y_split, idx_split in zip(labels, indices):
        split_loss.append(categorical_crossentropy(preds[idx_split], y_split[idx_split]))
        split_acc.append(accuracy(preds[idx_split], y_split[idx_split]))

    return split_loss, split_acc


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data_cora(path="data", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    try:
        data_path = fr"../RA-GCN/{path}/{dataset}/{dataset}"
        idx_features_labels = np.genfromtxt(data_path +".content", dtype=np.dtype(str))
        edges_unordered = np.genfromtxt(data_path +".cites", dtype=np.int32)

    except OSError as ex:
        data_path = "{}\{}\{}".format(path, dataset, dataset)
        idx_features_labels = np.genfromtxt(data_path+".content", dtype=np.dtype(str))
        edges_unordered = np.genfromtxt(data_path+".cites", dtype=np.int32)

    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features.todense(), adj, labels


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def get_nonego_simmilarity_vec(source_node, data , ind_list, cutoff
                               , simm_fun = lambda x,y: 1 - cosine_dist(x,y)
                               , sparseflag = False):
    """
    returns a vector with most similar values of nodes from @ind_list
    :param source_node:
    :param data:
    :param ind_list:
    :param cutoff:
    :param simm_fun:
    :return:
    """
    f_ref = data[source_node]
    out = np.zeros((1,data.shape[0]))
    out = scipy.sparse.dok_matrix(out) if sparseflag else out
    rank = []
    for n in ind_list:
        bisect.insort(rank,(simm_fun(f_ref, data[n,:]), n))
    for r,n in rank[::-1][:cutoff]:
        out[0,n] = r
    return out

def tqdm_parallel_map(executor, fn, *iterables, **kwargs):
    """
    Equivalent to executor.map(fn, *iterables),
    but displays a tqdm-based progress bar.

    Does not support timeout or chunksize as executor.submit is used internally

    **kwargs is passed to tqdm.
    """
    futures_list = []
    for iterable in iterables:
        futures_list += [executor.submit(fn, i) for i in iterable]
    for f in tqdm(concurrent.futures.as_completed(futures_list), total=len(futures_list), **kwargs):
        yield f.result()


def get_adjointed_l_bias(X, A, l = None, radius = 5, sparseflag = False, n_jobs = 1, cache=None):
    """
    Produces a similarity matrix like @A (Adjacency), with similarity values in
    the @l-most similar nodes which are @radius steps away from source   
    :param X: data
    :param A: Adjacency matrix
    :param l: cutoff value for most similar adjointed nodes
    :param radius: 
    :return: 
    """
    if l == None:
        return []
    else:
        FILE_PATH = os.path.abspath(__file__)
        DIR_PATH = os.path.dirname(FILE_PATH)
        DATA_PATH = os.path.join(DIR_PATH, fr'data/{cache}/cache')
        if not os.path.isdir(DATA_PATH):
            os.mkdir(DATA_PATH)
        FILE_NAME = os.path.join(DATA_PATH, f'bias_cache_L{l}_R{radius}.pkl')
        if cache:
            if os.path.isfile(FILE_NAME):
                print(f"Bias mat :: Loading bias matrix from cache...")
                with open(FILE_NAME, 'rb') as f:
                    return pickle.load(f)

        def get_n_bias(node, callback = None):
            """get bias vector for single node"""
            if callback:
                callback()
            ego_n = nx.ego_graph(G, node, radius=radius, center=True)
            non_ego_n = nodes_set - set(ego_n.nodes)
            return get_nonego_simmilarity_vec(node, X, non_ego_n, l, sparseflag=sparseflag)

        print(f'Bias mat :: Building {l}_order bias matrix...')
        G = nx.from_scipy_sparse_matrix(A) if isinstance(A, scipy.sparse.spmatrix) else \
            nx.from_numpy_array(A)
        nodes_set = set(G.nodes)
        b = []
        # TODO: cache output and parralelize function with plain joblib
        if n_jobs<=1:
            # no-parallel -     takes 100 sec (1.52 min) for 1000 nodes 2708 iter
            with tqdm(total = len(nodes_set)) as pbar:
                for n in G.nodes:
                    b_itr = get_n_bias(n)
                    b.append(b_itr)
                    pbar.set_description(desc = f"node {n}..")
                    pbar.update()
            out = scipy.sparse.vstack(b) if sparseflag else np.vstack(b).squeeze()
        else:
            n_chunks = round(len(G.nodes)/n_jobs)
            t = tqdm(total=len(G.nodes))

            def update():
                t.update()

            def get_nlist_bias(nlist, callback):
                b_parall = []
                for n in nlist:
                    b_itr = get_n_bias(n, callback = callback)
                    b_parall.append(b_itr)
                return b_parall

            mapped_process = partial(get_nlist_bias, callback = update)

            # parallel -        takes 4:47 min with 2708 iter
            split_every = (lambda n, it: takewhile(bool, (list(islice(it, n)) for _ in repeat(None))))
            nodes_chunks = list(split_every(n_chunks , iter(G.nodes)))
            with ThreadPoolExecutor(max_workers = n_jobs) as p:
                b = p.map(mapped_process, nodes_chunks, chunksize = n_chunks)

            out = scipy.sparse.vstack(list(b)) if sparseflag else np.vstack(list(b)).squeeze()
        if cache:
            print(f"Bias mat :: Caching bias matrix...")
            with open(FILE_NAME,'wb') as f:
                pickle.dump(out,f)
        return out


def get_splits(y):
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask


def normalized_laplacian(adj, symmetric=True):
    adj_normalized = normalize_adj(adj, symmetric)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    return laplacian


def rescale_laplacian(laplacian):
    try:
        print('Calculating largest eigenvalue of normalized graph Laplacian...')
        largest_eigval = eigsh(laplacian, 1, which='LM', return_eigenvectors=False)[0]
    except ArpackNoConvergence:
        print('Eigenvalue calculation did not converge! Using largest_eigval=2 instead.')
        largest_eigval = 2

    scaled_laplacian = (2. / largest_eigval) * laplacian - sp.eye(laplacian.shape[0])
    return scaled_laplacian


def chebyshev_polynomial(X, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    T_k = list()
    T_k.append(sp.eye(X.shape[0]).tocsr())
    T_k.append(X)

    def chebyshev_recurrence(T_k_minus_one, T_k_minus_two, X):
        X_ = sp.csr_matrix(X, copy=True)
        return 2 * X_.dot(T_k_minus_one) - T_k_minus_two

    for i in range(2, k+1):
        T_k.append(chebyshev_recurrence(T_k[-1], T_k[-2], X))

    return T_k

def noamuriel_polynomial(A, k, to_tensor=False):
    """Calculate noamuriel polynomials up to order k. Return a list of sparse matrices."""
    print("Calculating noamuriel polynomials up to order {}...".format(k))

    A_k = list()
    A_k.append(sp.eye(A.shape[0]).tocsr())
    A_k.append(A)

    def noamuriel_recurrence(T_k_minus_one, X):
        X_ = sp.csr_matrix(X, copy=True)
        return X_.dot(T_k_minus_one)

    for i in range(2, k+1):
        A_k.append(noamuriel_recurrence(A_k[-1], A))

    if to_tensor:
        # def convert_sparse_matrix_to_sparse_tensor(X):
        #     coo = X.tocoo().astype(np.float32)
        #     indices = np.mat([coo.row, coo.col]).transpose()
        #     return tf.SparseTensor(indices, coo.data, coo.shape)
        # A_k = [convert_sparse_matrix_to_sparse_tensor(A) for A in A_k]
        # A_k = tf.sparse_reshape(tf.sparse_concat(axis=1, sp_inputs=A_k), (A.shape[0], A.shape[1], k + 1))

        A_k = np.transpose(np.array([A.todense() for A in A_k]), [1,2,0])
        # A_k = tf.convert_to_tensor(A_k)

        A_k = [A_k]

    return A_k


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


## functions taken from https://github.com/PetarV-/GAT/
def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)