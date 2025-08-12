"""
Helper functions of HyperNIA(HyperNear)
root: ../code/Attack/HyperNIA
@author: Tingyi
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


import seaborn as sns
def compute_homophily(H, X, data, dataset, is_first_run=False, is_last_run=False):
    """
    Compute the homophily of the hypergraph and plot distributions with KDE.
    
    Parameters:
    - H: Hypergraph incidence matrix
    - X: Node feature matrix
    - data: Data name or identifier (for saving plot)
    - dataset: Dataset name or identifier (for saving plot)
    - is_first_run: Boolean, True if it's the first run (pre-attack)
    - is_last_run: Boolean, True if it's the last run (post-attack)
    """
    global pre_attack_sims, post_attack_sims
    
    # Normalize hypergraph
    X_neg = hypergraph_norm(H, X)

    # Compute cosine similarity between original and normalized features
    node_sims = np.array([F.cosine_similarity(xn.unsqueeze(0), xx.unsqueeze(0)).item() for (xn, xx) in zip(X_neg, X)])
    node_sims_nonzero = node_sims[node_sims > 0]
    homophily_score = np.mean(node_sims_nonzero) if len(node_sims_nonzero) > 0 else 0.0

    # Record pre-attack similarities if it's the first run
    if is_first_run:
        pre_attack_sims = node_sims_nonzero

    # Record post-attack similarities if it's the last run and plot
    if is_last_run:
        post_attack_sims = node_sims_nonzero
        
        # KDE Plot
        plt.figure(figsize=(10, 6))
        sns.kdeplot(pre_attack_sims, color="dodgerblue", label="Before Attack", fill=True)
        sns.kdeplot(post_attack_sims, color="tomato", label="After Attack", fill=True)

        # Mark mean and std
        plt.axvline(np.mean(pre_attack_sims), color='blue', linestyle='--', label=f'Pre-attack Mean: {np.mean(pre_attack_sims):.2f}')
        plt.axvline(np.mean(post_attack_sims), color='red', linestyle='--', label=f'Post-attack Mean: {np.mean(post_attack_sims):.2f}')
        plt.fill_betweenx([0, plt.ylim()[1]], np.mean(pre_attack_sims)-np.std(pre_attack_sims), np.mean(pre_attack_sims)+np.std(pre_attack_sims), color='blue', alpha=0.2)
        plt.fill_betweenx([0, plt.ylim()[1]], np.mean(post_attack_sims)-np.std(post_attack_sims), np.mean(post_attack_sims)+np.std(post_attack_sims), color='red', alpha=0.2)

        # Customize plot
        plt.xlabel("Homophily", fontsize=20)
        plt.ylabel("Density", fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(loc='upper left', fontsize=14)
        

        # Difference Heatmap
        # Align lengths by padding the shorter array with NaN values
        max_len = max(len(pre_attack_sims), len(post_attack_sims))
        pre_attack_sims_padded = np.pad(pre_attack_sims, (0, max_len - len(pre_attack_sims)), 'constant', constant_values=np.nan)
        post_attack_sims_padded = np.pad(post_attack_sims, (0, max_len - len(post_attack_sims)), 'constant', constant_values=np.nan)
        diff = post_attack_sims_padded - pre_attack_sims_padded
        # diff = post_attack_sims- pre_attack_sims
        plt.figure(figsize=(10, 6))
        heatmap = sns.heatmap(diff.reshape(1, -1), cmap="RdYlGn", cbar_kws={'label': 'Homophily Change'})
        # 调整颜色条标签字体大小
        colorbar = heatmap.collections[0].colorbar
        colorbar.set_label('Homophily Change', size=14)  # 设置标签字体大小
        colorbar.ax.tick_params(labelsize=14)  # 调整颜色条刻度字体大小
        plt.xlabel("Node Index", fontsize=20)
        plt.ylabel("Homophily Change", fontsize=20)
        plt.xticks([])
        
        # Save heatmap
        plt.savefig(f"NDA_homo_{data}_{dataset}.png", dpi=300, bbox_inches='tight')
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
    plt.savefig(f"{data}_{dataset}_embedding_0.1.png")
    plt.close()
