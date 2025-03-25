"""
Node Injection Attack
@author: caiman
date: 2024/7/31
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse import lil_matrix
import pdb
import torch
import random

class HypergraphAttack:
    def __init__(self, adj_matrix, features, device='cpu', seed=4202):
        self.adj_matrix = adj_matrix
        self.features = features
        self.device = device
        self.seed = seed
        
        # 确保 self.features 是 numpy 数组
        if not isinstance(self.features, np.ndarray):
            self.features = self.features.cpu().numpy()
            
        # Set the seed for reproducibility
        self.set_seed(self.seed)
        
        
        # Initialize class-related attributes
        self.features_dim = features.shape[1]
        self.major_features_candidates = self.identify_major_features()
        self.feature_avg = np.mean(self.features, axis=0, keepdims=True)

    def set_seed(self, seed):
        """Set the random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def identify_major_features(self, num_features=5):
        """Identify major features based on variance."""
        feature_variances = np.var(self.features, axis=0)
        major_features = np.argsort(feature_variances)[-num_features:]  # Select top features by variance
        return major_features
    
    def make_statistic_features(self, n_added):
        n_added_features = np.zeros((n_added, self.features_dim))
        for i in range(n_added):
            n_added_features[i][self.major_features_candidates] = self.feature_avg[0][self.major_features_candidates]
        return n_added_features
    

    def inject_nodes(self, k=10, injection_ratio=0.05, nodes_per_hyperedge=5, perturbation_std=2, budget=0.5):
        """
        Node injection 
        Parameters:
        X (torch.Tensor): Node features matrix
        H (torch.Tensor): Hypergraph incidence matrix
        k (int): Number of top hyperedges to target for injection (default is 10)
        nodes_per_hyperedge (int): Number of nodes to inject into each hyperedge (default is 3)
        injection_ratio (float): Ratio of injected nodes to the original nodes (default is 0.05)

        Returns:
        modified_X (torch.Tensor): Modified node features matrix with injected nodes
        modified_H (torch.Tensor): Modified hypergraph incidence matrix with injected nodes
        """
  
        # Convert scipy sparse matrix to dense tensor
        H = torch.tensor(self.adj_matrix.todense(), dtype=torch.float32)
        X = torch.tensor(self.features, dtype=torch.float32)
        
        
        
        # Randomly select k hyperedges 
        num_nodes = X.shape[0]
        num_hyperedges = H.shape[1]
        num_injected_nodes = int(num_nodes * injection_ratio)
        k = min(num_injected_nodes // nodes_per_hyperedge, k)
        # topk_indices = random.sample(range(num_hyperedges), k)

        # Calculate the degree of each hyperedge
        degrees = torch.sum(H, dim=0)
        # Sort hyperedges indices based on degree in descending order
        sorted_indices = torch.argsort(degrees, descending=True)
        # Select top-k hyperedges
        topk_indices = sorted_indices[:k]
        
        # Calculate the number of injected nodes
        num_nodes = X.shape[0]
        num_hyperedges = H.shape[1]
        num_injected_nodes = int(num_nodes * injection_ratio)

        # # Randomly select features for injected nodes
        # injected_features = X[torch.randperm(num_nodes)[:num_injected_nodes]]

        # Random generation node features
        feature_dim = X.shape[1]
        injected_features = torch.rand(num_injected_nodes, feature_dim, device=self.device)
        
        # Generate the list of indices for injected nodes
        injected_indices = list(range(num_nodes, num_nodes + num_injected_nodes))
        # Initialize modified node features matrix and hypergraph incidence matrix
        modified_X = torch.cat((X, injected_features), dim=0)

        # Calculate the number of new hyperedges needed for connecting injected nodes
        num_new_hyperedges = (num_injected_nodes * (num_injected_nodes - 1)) // 2

        # Expand hypergraph dimension to accommodate injected nodes and new hyperedges
        new_columns = torch.zeros(num_nodes + num_injected_nodes, num_new_hyperedges)
        modified_H = torch.cat((H, new_columns[:num_nodes, :]), dim=1)  # Add columns for injected nodes and new hyperedges

        new_rows = torch.zeros(num_injected_nodes, modified_H.shape[1])
        modified_H = torch.cat((modified_H, new_rows), dim=0)  # Add rows for injected nodes

        # Inject nodes into top-k hyperedges
        for i, idx in enumerate(topk_indices):
            for j in range(nodes_per_hyperedge):
                new_node_index = num_nodes + i * nodes_per_hyperedge + j
                modified_H[new_node_index, idx] = 1


        # Connect injected nodes with themselves by creating new hyperedges
        edge_idx = 0
        for i in range(num_injected_nodes):
            for j in range(i + 1, num_injected_nodes):
                new_hyperedge_index = num_hyperedges + edge_idx
                modified_H[num_nodes + i, new_hyperedge_index] = 1
                modified_H[num_nodes + j, new_hyperedge_index] = 1
                edge_idx += 1
        
        self.original_edges = modified_H[:, : num_hyperedges]
        self.injected_edges = modified_H[:, num_hyperedges:]
        
        
        return modified_X, modified_H, k, injected_indices

        # # Convert scipy sparse matrix to dense tensor
        # H = torch.tensor(self.adj_matrix.todense(), dtype=torch.float32, device=self.device)
        # X = torch.tensor(self.features, dtype=torch.float32).to(self.device).clone().detach()
        
        # self.set_seed(self.seed)
        
        # # Calculate the degree of each hyperedge
        # degrees = torch.sum(H, dim=0)
        
        # # Calculate the degree of each node
        # node_degrees = torch.sum(H, dim=1)
        # # Calculate the average node degree for each hyperedge
        # hyperedge_node_degrees = torch.matmul(H.t(), node_degrees)
        # hyperedge_sizes = torch.sum(H, dim=0)
        # average_node_degrees = hyperedge_node_degrees / hyperedge_sizes

        # # Sort hyperedges indices based on degree in descending order
        # # sorted_indices = torch.argsort(degrees, descending=True) # 超边大度
        # # # Sort hyperedges indices based on degree in ascending order
        # # sorted_indices = torch.argsort(degrees) # 超边小度
        
        # # Sort hyperedges indices based on average node degree in ascending order
        # # sorted_indices = torch.argsort(average_node_degrees) # 小度
        # # sorted_indices = torch.argsort(average_node_degrees, descending=True) # 大度
        

        # # Calculate the number of injected nodes
        # num_nodes = X.shape[0]
        # num_hyperedges = H.shape[1]
        # num_injected_nodes = int(num_nodes * injection_ratio)
        # # pdb.set_trace()
        # # k <= num_injected_nodes
        # # k = min(num_injected_nodes, k)
        # k = min(num_injected_nodes // nodes_per_hyperedge, k)
        # # # Select top-k hyperedges
        # # topk_indices = sorted_indices[:k]
        
        # # Randomly select k hyperedges 
        # # self.set_seed(4202)
        # topk_indices = random.sample(range(num_hyperedges), k)
        
        # # torch.set_printoptions(profile="full") # 完整打印tensor

        # # # Random extraction from the original node features
        # # injected_features = X[torch.randperm(num_nodes)[:num_injected_nodes]] 
       
        # # Random generation
        # feature_dim = X.shape[1]
        # # self.set_seed(4202)
        # injected_features = torch.rand(num_injected_nodes, feature_dim, device=self.device)
        
        # # # 使用 make_statistic_features 函数生成注入节点特征
        # # injected_features = self.make_statistic_features(num_injected_nodes)
        # # injected_features = torch.tensor(injected_features, dtype=torch.float32, device=self.device)
        
        # # # Generate features for injected nodes by adding perturbation to original features
        # # original_features = X[torch.randperm(num_nodes)[:num_injected_nodes]]
        # # perturbation = torch.randn_like(original_features) * perturbation_std
        # # injected_features = original_features + perturbation

        # # Initialize modified node features matrix and hypergraph incidence matrix
        # modified_X = torch.cat((X, injected_features), dim=0)

        # # Calculate the number of new hyperedges needed for connecting injected nodes i.e., $C_k^2)$
        # # num_new_hyperedges = (num_injected_nodes * (num_injected_nodes - 1)) // 2
        # num_new_hyperedges = 1        
               
        # # Expand hypergraph dimension to accommodate injected nodes and new hyperedges
        # new_columns = torch.zeros(num_nodes + num_injected_nodes, num_new_hyperedges, device=self.device)
        # modified_H = torch.cat((H, new_columns[:num_nodes, :]), dim=1)  # Add columns for injected nodes and new hyperedges
        # new_rows = torch.zeros(num_injected_nodes, modified_H.shape[1], device=self.device)
        # modified_H = torch.cat((modified_H, new_rows), dim=0)  # Add rows for injected nodes

        # # Inject nodes into top-k hyperedges
        # for i, idx in enumerate(topk_indices):
        #     for j in range(nodes_per_hyperedge):
        #         new_node_index = num_nodes + i * nodes_per_hyperedge + j
        #         modified_H[new_node_index, idx] = 1

        # # Connect injected nodes with themselves by creating new hyperedges
        # edge_idx = 0
        # for i in range(num_injected_nodes):
        #     for j in range(i + 1, num_injected_nodes):
        #         new_hyperedge_index = num_hyperedges + edge_idx
        #         modified_H[num_nodes + i, new_hyperedge_index] = 1
        #         modified_H[num_nodes + j, new_hyperedge_index] = 1
        #         edge_idx += 1
        
        # # # Connect all injected nodes into the same new hyperedge
        # # new_hyperedge_index = num_hyperedges  # Use the first new hyperedge index
        # # for i in range(num_injected_nodes):
        # #     modified_H[num_nodes + i, new_hyperedge_index] = 1

        # # self.original_edges = modified_H[:, : num_hyperedges]
        # # self.injected_edges = modified_H[:, num_hyperedges:]
        
        # return modified_X, modified_H, k

    def random_split_hyperedge(self, edges):
        """
        Randomly split one hyperedge.

        Args:
        edges (torch.Tensor): Tensor representing the hyperedges. i.e., H

        Returns:
        torch.Tensor: Tensor representing the hyperedges after random splitting.
        """

        num_nodes, num_hyperedges = edges.size()
        edges = edges.clone()
        
        hyperedge_idx = random.randint(0, num_hyperedges - 1)
        hyperedge_nodes = torch.nonzero(edges[:, hyperedge_idx]).squeeze()  
        # Ensure hyperedge_nodes is a list
        if hyperedge_nodes.dim() == 0:  # This handles the case where squeeze results in a scalar
            hyperedge_nodes = [hyperedge_nodes.item()]
        else:
            hyperedge_nodes = hyperedge_nodes.tolist()
        if len(hyperedge_nodes) <= 1:
            return edges
        # self.set_seed(4202)
        split_point = random.randint(1, len(hyperedge_nodes) - 1)
        random.shuffle(hyperedge_nodes)
        new_hyperedge1 = hyperedge_nodes[:split_point]
        new_hyperedge2 = hyperedge_nodes[split_point:]
        
        new_edges = torch.cat((edges, torch.zeros(edges.shape[0], 1, device=edges.device)), dim=1)
        new_edges[:, hyperedge_idx] = 0
        new_edges[new_hyperedge1, hyperedge_idx] = 1
        new_edges[new_hyperedge2, -1] = 1
        # pdb.set_trace()
        
        return new_edges
