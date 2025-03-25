"""
Helper functions of HyperNIA
root: ../code/Attack/HyperNIA
@author: 

It contains the source code for the paper [_UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks_](https://arxiv.org/abs/2105.00956), accepted by IJCAI 2021.
"""
from model import *
import torch, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
import pdb
import torch_sparse as tsp
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE

def accuracy(Z, Y):
    
    return 100 * Z.argmax(1).eq(Y).float().mean().item()


import torch_sparse

def compute_node_node_adjacency_matrix(G, N): # A
    """
    Compute the node-node adjacency matrix from the hypergraph dictionary G.

    Args:
        G (dict): Hypergraph dictionary where keys are edges and values are sets of nodes.
        N (int): Number of nodes.
    
    Returns:
        sp.csc_matrix: The node-node adjacency matrix.
    """
    row = []
    col = []
    data = []

    # Iterate over each edge in the hypergraph
    for nodes in G.values():
        nodes = list(nodes)  # Convert set to list if necessary
        # Create all pairs of nodes in this edge
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                row.append(nodes[i])
                col.append(nodes[j])
                data.append(1)
                row.append(nodes[j])
                col.append(nodes[i])
                data.append(1)

    # Create a COO matrix to ensure no duplicate entries
    A_coo = sp.coo_matrix((data, (row, col)), shape=(N, N), dtype=int)
    
    # Convert to CSR format to remove duplicate entries
    A_csr = sp.csr_matrix(A_coo)

    # Ensure the matrix contains only 0s and 1s
    A_csr.data = np.clip(A_csr.data, 0, 1)

    return A_csr


def fetch_data(args):
    from data import data 
    dataset, _, _ = data.load(args)
    args.dataset_dict = dataset 

    X, Y, G = dataset['features'], dataset['labels'], dataset['hypergraph']
   
    # node features in sparse representation
    X = sp.csr_matrix(normalise(np.array(X)), dtype=np.float32)
    X = torch.FloatTensor(np.array(X.todense()))
    
    # labels
    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])

    X, Y = X.cuda(), Y.cuda()
    
    # # Compute node-node adjacency matrix
    # N = X.shape[0]
    # A = compute_node_node_adjacency_matrix(G, N)
    
    return X, Y, G

def initialise(X, Y, G, args, X_injected=None, H_injected=None, unseen=None):
    """
    initialises model, optimiser, normalises graph, and features
    
    arguments:
    X, Y, G: the entire dataset (with graph, features, labels)
    args: arguments
    unseen: if not None, remove these nodes from hypergraphs

    returns:
    a tuple with model details (UniGNN, optimiser)    
    """
        
    G = G.copy()
    
    if unseen is not None:
        unseen = set(unseen)
        # remove unseen nodes
        for e, vs in G.items():
            G[e] =  list(set(vs) - unseen)

    if args.add_self_loop:
        Vs = set(range(X.shape[0]))

        # only add self-loop to those are orginally un-self-looped
        # TODO:maybe we should remove some repeated self-loops?
        for edge, nodes in G.items():
            if len(nodes) == 1 and nodes[0] in Vs:
                Vs.remove(nodes[0])

        for v in Vs:
            G[f'self-loop-{v}'] = [v]

    # if args.model_name == 'HyperGCN':
    #     num_nodes = X.shape[0]
    #     Graph = args.dataset_dict['hypergraph']
    # Injection !!!
    if X_injected is not None and H_injected is not None:
        X = X_injected.cpu().numpy()
        H = H_injected.cpu().numpy()
        H = sp.coo_matrix(H).tocsr()
        # if args.model_name == 'HyperGCN':
        #     num_nodes = X_injected.shape[0]
        #     Graph = hypergraph_to_graph(X_injected, H_injected)
        #     Graph = csr_to_dict(Graph)

    N, M = X.shape[0], len(G)
    indptr, indices, data = [0], [], []
    for e, vs in G.items():
        indices += vs 
        data += [1] * len(vs)
        indptr.append(len(indices))
    H = sp.csc_matrix((data, indices, indptr), shape=(N, M), dtype=int).tocsr() # V x E
    # pdb.set_trace()
   
    
    degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    degE2 = torch.from_numpy(H.sum(0)).view(-1, 1).float()

    (row, col), value = torch_sparse.from_scipy(H)
    V, E = row, col
    from torch_scatter import scatter
    assert args.first_aggregate in ('mean', 'sum'), 'use `mean` or `sum` for first-stage aggregation'
    degE = scatter(degV[V], E, dim=0, reduce=args.first_aggregate)
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[degV.isinf()] = 1 # when not added self-loop, some nodes might not be connected with any edge


    V, E = V.cuda(), E.cuda()
    args.degV = degV.cuda()
    args.degE = degE.cuda()
    args.degE2 = degE2.pow(-1.).cuda()


    nfeat, nclass = X.shape[1], len(Y.unique())
    nlayer = args.nlayer
    nhid = args.nhid
    nhead = args.nhead

    # UniGNN and optimiser
    if args.model_name == 'UniGCNII':
        model = UniGCNII(args, nfeat, nhid, nclass, nlayer, nhead, V, E)
        optimiser = torch.optim.Adam([
            dict(params=model.reg_params, weight_decay=0.01),
            dict(params=model.non_reg_params, weight_decay=5e-4)
        ], lr=0.01)
    elif args.model_name == 'HyperGCN':
        args.fast = True
        dataset = args.dataset_dict
        # if X_injected is not None and H_injected is not None:
        #     model = HyperGCN(args, nfeat, nhid, nclass, nlayer, num_nodes, Graph, X_injected)
        # else:
        # model = HyperGCN(args, nfeat, nhid, nclass, nlayer, dataset['n'], dataset['hypergraph'], X.cpu())
        model = HyperGCN(args, nfeat, nhid, nclass, nlayer, dataset['n'], dataset['hypergraph'], X)
        optimiser = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    else:
        model = UniGNN(args, nfeat, nhid, nclass, nlayer, nhead, V, E)
        optimiser = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


    model.cuda()
   
    return model, optimiser, X, H



def normalise(M):
    """
    row-normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1} M  
    where D is the diagonal node-degree matrix 
    """
    
    d = np.array(M.sum(1))
    
    di = np.power(d, -1).flatten()
    di[np.isinf(di)] = 0.
    DI = sp.diags(di)    # D inverse i.e. D^{-1}
    
    return DI.dot(M)



"""
2024/8/4 
Compute Hypernode-Centric Homophily by features, i.e., 
HNCH = Similar(Original node feature, Node feature after a hypergraph convolution)
"""
def hypergraph_norm(H, X):
    """
    Hypergraph 1-hop neighbor aggregate.
    """
    if isinstance(H, torch.Tensor):
        H = H.cpu().numpy()  # 转换为 numpy 数组
        H = sp.csr_matrix(H)  # 转换为 scipy.sparse.csr_matrix

    
    degV = torch.from_numpy(H.sum(1)).view(-1, 1).float()
    degE2 = torch.from_numpy(H.sum(0)).view(-1, 1).float()
    N = X.shape[0]

    (row, col), value = torch_sparse.from_scipy(H)
    V, E = row, col
    from torch_scatter import scatter
    degE = scatter(degV[V], E, dim=0, reduce='mean')
    degE = degE.pow(-0.5)
    degV = degV.pow(-0.5)
    degV[degV.isinf()] = 1 # when not added self-loop, some nodes might not be connected with any edge

    V, E = V.cuda(), E.cuda()
     
    Xve = X[V].cuda() # [nnz, C]
    Xe = scatter(Xve, E, dim=0, reduce='mean') # [E, C]
    Xe = Xe.cuda()

    Xe = Xe * degE.cuda() 

    Xev = Xe[E] # [nnz, C]
    Xv = scatter(Xev, V, dim=0, reduce='sum', dim_size=N) # [N, C]

    Xv = Xv * degV.cuda()
    return Xv

# def compute_homophily(H, X, data, dataset, save_plot=True):
#     """
#     Compute the homophily of the hypergraph.

#     Parameters:
#     - H: Hypergraph incidence matrix
#     - X: Node feature matrix
#     - data: Data name or identifier
#     - dataset: Dataset name or identifier
#     - save_plot: Whether to save the cosine similarity distribution plot
#     """
#     # Normalize hypergraph (assuming hypergraph_norm returns normalized node features)
#     X_neg = hypergraph_norm(H, X)

#     # Compute cosine similarity between original and negative sampled features
#     node_sims = np.array([F.cosine_similarity(xn.unsqueeze(0), xx.unsqueeze(0)).item() for (xn, xx) in zip(X_neg, X)])
#     # node_sims = np.array([F.cosine_similarity(xn.unsqueeze(0), xx.unsqueeze(0)).item() for (xn, xx) in zip(X, X)])
#     # Filter out zero values
#     node_sims_nonzero = node_sims[node_sims > 0]
#     # Compute the mean of non-zero values
#     homophily_score = np.mean(node_sims_nonzero) if len(node_sims_nonzero) > 0 else 0.0
#     # homophily_score = np.mean(node_sims)
#     if save_plot:
#         plt.figure(figsize=(8, 6))  # Set the figure size
#         plt.hist(node_sims_nonzero, bins=100, density=True, alpha=0.75, color='dodgerblue', edgecolor='black')
        
#         # Add grid and customize axes
#         plt.grid(True, linestyle='--', alpha=0.7)

#         # Customize axis labels with larger font size and bold font
#         plt.xlabel("$FHH$", fontsize=20)
#         plt.ylabel("Density", fontsize=20)
        
#         # # Add a title with a larger font size
#         # plt.title("Cosine Similarity Distribution", fontsize=16, fontweight='bold', color='navy')

#         # Customize ticks font size
#         plt.xticks(fontsize=16)
#         plt.yticks(fontsize=16)

#         # Save the plot
#         plt.savefig(f"homo_{data}_{dataset}.png", dpi=300, bbox_inches='tight')
    
#     # plt.savefig("coci_cora_node_sims.pdf")
#     # pdb.set_trace()
#     # plt.close() # 画一张图上对比更直观
#     return homophily_score
def compute_homophily(H, X, data, dataset, is_first_run=False, is_last_run=False):
    """
    Compute the homophily of the hypergraph.
    Plot the homophily distribution only for the first and last run.
    
    Parameters:
    - H: Hypergraph incidence matrix
    - X: Node feature matrix
    - data: Data name or identifier (for saving plot)
    - dataset: Dataset name or identifier (for saving plot)
    - is_first_run: Boolean, True if it's the first run (pre-attack)
    - is_last_run: Boolean, True if it's the last run (post-attack)
    """
    global pre_attack_sims, post_attack_sims
    
    # Normalize hypergraph (assuming hypergraph_norm returns normalized node features)
    X_neg = hypergraph_norm(H, X)

    # Compute cosine similarity between original and negative sampled features (vectorized for speed)
    node_sims = np.array([F.cosine_similarity(xn.unsqueeze(0), xx.unsqueeze(0)).item() for (xn, xx) in zip(X_neg, X)])
    # Filter out zero values
    node_sims_nonzero = node_sims[node_sims > 0]

    # Compute the mean of non-zero values
    homophily_score = np.mean(node_sims_nonzero) if len(node_sims_nonzero) > 0 else 0.0

    # 如果是第一次运行，记录 pre_attack_sims
    if is_first_run:
        pre_attack_sims = node_sims_nonzero

    # 如果是最后一次运行，记录 post_attack_sims 并进行绘图
    if is_last_run:
        post_attack_sims = node_sims_nonzero
        # 绘制两次的相似度分布
        plt.figure(figsize=(8, 6))
        
        # 绘制首次的分布
        plt.hist(pre_attack_sims, bins=100, density=True, alpha=0.6, color='dodgerblue', edgecolor='black', label='Before Attack')

        # 绘制最终的分布
        plt.hist(post_attack_sims, bins=100, density=True, alpha=0.6, color='tomato', edgecolor='black', label='After Attack')

        # Customize the plot
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel("Homophily", fontsize=20)
        plt.ylabel("Density", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc='upper left', fontsize=16)

        # Save plot
        plt.savefig(f"homo_{data}_{dataset}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    return homophily_score

def homophily_loss(homophily_before, homophily_after, tolerance):
    """
    Mean Squared Error
    MSE = 1/N SUM_{i=1}^n (a_i - b_i)^2
    """
    # return F.mse_loss(homophily_after, homophily_before, reduction='mean') / tolerance
    mse_loss = (homophily_after - homophily_before) ** 2 / tolerance
    return mse_loss

    
def compare_feature_statistics(X_before, X_after):
    X_before_np = X_before.cpu().detach().numpy()
    X_after_np = X_after.cpu().detach().numpy()

    mean_before = np.mean(X_before_np, axis=0)
    mean_after = np.mean(X_after_np, axis=0)
    var_before = np.var(X_before_np, axis=0)
    var_after = np.var(X_after_np, axis=0)

    print("Mean before injection:", mean_before)
    print("Mean after injection:", mean_after)
    print("Variance before injection:", var_before)
    print("Variance after injection:", var_after)
    
# def visualize_distribution(X_before, X_after, injected_indices, data, dataset):
#     # Convert to numpy arrays
#     X_before_np = X_before.cpu().detach().numpy()
#     X_after_np = X_after.cpu().detach().numpy()

#     # Apply t-SNE on the original data
#     tsne = TSNE(n_components=2, random_state=42)
#     X_before_2d = tsne.fit_transform(X_before_np)

#     # Apply the same t-SNE transformation to the data after injection
#     X_after_2d = tsne.fit_transform(X_after_np)

#     # Create labels for visualization
#     labels_before = np.array(['Original'] * len(X_before_np))
#     labels_after = np.array(['Original'] * len(X_after_np))
#     labels_after[injected_indices] = 'Injected'

#     plt.savefig(f"{data}_{dataset}_embedding_distribution.png")
#     plt.close() 

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np

# def visualize_distribution(X_before, X_after, injected_indices, data, dataset):
#     # Convert to numpy arrays
#     X_before_np = X_before.cpu().detach().numpy()
#     X_after_np = X_after.cpu().detach().numpy()

#     # Apply t-SNE on the original data
#     tsne = TSNE(n_components=2, random_state=42)
#     X_before_2d = tsne.fit_transform(X_before_np)

#     # Apply the same t-SNE transformation to the data after injection
#     X_after_2d = tsne.fit_transform(X_after_np)

#     # Create labels for visualization
#     labels_before = np.array(['Original'] * len(X_before_np))
#     labels_after = np.array(['Original'] * len(X_after_np))
#     labels_after[injected_indices] = 'Injected'

#     # Font size setting
#     font_size = 20

#     # Plotting original data distribution
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X_before_2d[:, 0], X_before_2d[:, 1], c='dodgerblue', label='Original', s=10, alpha=0.75)
#     plt.title('Data Distribution Before Attack', fontsize=font_size)
#     plt.legend(fontsize=font_size)
#     plt.xlabel('t-SNE Dimension 1', fontsize=font_size)
#     plt.ylabel('t-SNE Dimension 2', fontsize=font_size)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.grid(True, alpha=0.5)
#     plt.savefig(f"{data}_{dataset}_before_attack_distribution.png")
#     plt.close()

#     # Plotting data distribution after attack
#     plt.figure(figsize=(8, 6))
#     plt.scatter(X_after_2d[labels_after == 'Original', 0], X_after_2d[labels_after == 'Original', 1], 
#                 c='#1e90ff', label='Original', s=10, alpha=0.75)  # Lighter blue than dodgerblue
#     plt.scatter(X_after_2d[labels_after == 'Injected', 0], X_after_2d[labels_after == 'Injected', 1], 
#                 c='tomato', label='Injected', s=10, alpha=0.75)  # Lighter tomato
#     # plt.title('Data Distribution After Attack', fontsize=font_size)
#     plt.legend(fontsize=font_size)
#     # plt.xlabel('t-SNE Dimension 1', fontsize=font_size)
#     # plt.ylabel('t-SNE Dimension 2', fontsize=font_size)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)
#     plt.grid(True, alpha=0.5)
#     plt.savefig(f"{data}_{dataset}_after_attack_distribution.png")
#     plt.close()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def visualize_distribution(X_before, X_after, injected_indices, data, dataset):
    # Convert to numpy arrays
    X_after_np = X_after.cpu().detach().numpy()

    # Apply t-SNE on the data after injection
    tsne = TSNE(n_components=2, random_state=42)
    X_after_2d = tsne.fit_transform(X_after_np)

    # Create labels for visualization
    labels_after = np.array(['Original'] * len(X_after_np))
    labels_after[injected_indices] = 'Injected'

    # Font size setting
    font_size = 20

    # Plotting data distribution after attack (original and injected)
    plt.figure(figsize=(8, 6))
    
    # Original nodes after attack
    plt.scatter(X_after_2d[labels_after == 'Original', 0], X_after_2d[labels_after == 'Original', 1], 
                c='dodgerblue', label='Original Nodes', s=14, alpha=0.7)
    
    # Injected nodes after attack
    plt.scatter(X_after_2d[labels_after == 'Injected', 0], X_after_2d[labels_after == 'Injected', 1], 
                c='tomato', label='Injected Nodes', s=14, alpha=0.7)
    
    # Plot settings
    # plt.title('Data Distribution After Attack', fontsize=font_size)
    plt.legend(fontsize=font_size)
    # plt.xlabel('t-SNE Dimension 1', fontsize=font_size)
    # plt.ylabel('t-SNE Dimension 2', fontsize=font_size)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, alpha=0.5)
    
    # Save the figure
    plt.savefig(f"{data}_{dataset}_after_attack_distribution.png")
    plt.close()
