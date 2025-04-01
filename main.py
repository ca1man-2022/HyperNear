import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer
import os
import copy
import numpy as np
import time
import datetime
import path
import shutil
import pdb
import config
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

args = config.parse()

# GPU, seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)

use_norm = 'use-norm' if args.use_norm else 'no-norm'
add_self_loop = 'add-self-loop' if args.add_self_loop else 'no-self-loop'

#### Configure output directory

dataname = f'{args.data}_{args.dataset}'
model_name = args.model_name
nlayer = args.nlayer
dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
out_dir = path.Path(f'./{args.out_dir}/2-runs/{model_name}_{nlayer}_{dataname}/seed_{args.seed}')

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()

### Configure logger
from logger import get_logger

baselogger = get_logger('base logger', f'{out_dir}/logging.log', not args.nostdout)
resultlogger = get_logger('result logger', f'{out_dir}/result.log', not args.nostdout)
baselogger.info(args)

# Load data
from data import data
from prepare import *

test_accs = []
best_val_accs, best_test_accs = [], []
new_test_accs = []
new_best_val_accs, new_best_test_accs = [], []
update_test_accs = []
update_best_val_accs, update_best_test_accs = [], []

resultlogger.info(args)


# Load data
X, Y, G = fetch_data(args)


for run in range(1, args.n_runs+1):
    run_dir = out_dir / f'{run}'
    run_dir.makedirs_p()

    # Load data
    args.split = run
    _, train_idx, test_idx = data.load(args)
    train_idx = torch.LongTensor(train_idx).cuda()
    test_idx = torch.LongTensor(test_idx).cuda()

    # Model 
    model, optimizer, X, H = initialise(X, Y, G, args)
    # 计算同配率
    homophily_score_before = compute_homophily(H, X, args.data, args.dataset)#, True, False)
    # print(f'Average homophily score : {homophily_score_before}')
    
    # pdb.set_trace()

    baselogger.info(f'Run {run}/{args.n_runs}, Total Epochs: {args.epochs}')
    baselogger.info(model)
    baselogger.info(f'total_params:{sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    tic_run = time.time()

    from collections import Counter
    counter = Counter(Y[train_idx].tolist())
    baselogger.info(counter)
    label_rate = len(train_idx) / X.shape[0]
    baselogger.info(f'label rate: {label_rate}')

    best_test_acc, test_acc, Z = 0, 0, None    
    for epoch in range(args.epochs):
        # Train
        tic_epoch = time.time()
        model.train()

        optimizer.zero_grad()
        Z = model(X)
        loss = F.nll_loss(Z[train_idx], Y[train_idx])

        loss.backward()
        optimizer.step()

        train_time = time.time() - tic_epoch 
        
        # Eval
        model.eval()
        Z = model(X)
        train_acc = accuracy(Z[train_idx], Y[train_idx])
        test_acc = accuracy(Z[test_idx], Y[test_idx])

        # Log acc
        best_test_acc = max(best_test_acc, test_acc)
        baselogger.info(f'epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.2f} | test acc:{test_acc:.2f} | time:{train_time*1000:.1f}ms')
        
    resultlogger.info(f"Run {run}/{args.n_runs}, best test accuracy: {best_test_acc:.2f}, acc(last): {test_acc:.2f}, total time: {time.time()-tic_run:.2f}s")
    test_accs.append(test_acc)
    best_test_accs.append(best_test_acc)

    # Node Inject Attack
    from NIA import HypergraphAttack
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hypergraph_attack = HypergraphAttack(H, X)

    # Select k hyperedges to inject
    k_values = args.k
    k = int(H.shape[1] * 0.01) * k_values  # 10% hyperedges

    # Inject nodes
    modified_X, modified_H, ground_k, injected_indices = hypergraph_attack.inject_nodes(k, injection_ratio=0.05)
    
    # homophily_score_after_injection = compute_homophily(modified_H, modified_X, args.data, args.dataset)
    # homophily_injection_loss = homophily_loss(homophily_score_before, homophily_score_after_injection, tolerance=0.1)
   
    new_best_test_acc, new_test_acc, Z = 0, 0, None    
    update_best_test_acc, update_test_acc = 0, 0 

    # Model initialization after injection
    model, optimizer, new_X, new_H = initialise(X, Y, G, args, modified_X, modified_H)
    # new_X = torch.from_numpy(new_X) # new_X是计算同配率用的

    torch.set_printoptions(profile="full") 
    baselogger.info(f'--------------begin---NIA------k={ground_k}----------')
    
    lambda_h = args.lambda_h  # Weight for the homophily regularization term <Injection>
    # homo_budget = 0.1  # Budget for the homophily difference <Update>
    
    for epoch in range(args.epochs):    
        # Train
        tic_epoch = time.time()
        model.train()
        optimizer.zero_grad()
        modified_X = modified_X.to(device)
        Z = model(modified_X)
        loss = F.nll_loss(Z[train_idx], Y[train_idx])
        loss_value = loss.item()
        loss.backward()
        optimizer.step()
        train_time = time.time() - tic_epoch
        
        # Eval
        model.eval()
        Z = model(modified_X)
        new_train_acc = accuracy(Z[train_idx], Y[train_idx])
        new_test_acc = accuracy(Z[test_idx], Y[test_idx])

        # Log acc
        new_best_test_acc = max(new_best_test_acc, new_test_acc)
        baselogger.info(f'epoch:{epoch} | loss after NIA:{loss:.4f} | train acc after NIA:{new_train_acc:.2f} | test acc after NIA:{new_test_acc:.2f} | time:{train_time*1000:.1f}ms')
            
        # Save current model and optimizer state
        current_model_state = copy.deepcopy(model.state_dict())
        current_optimizer_state = copy.deepcopy(optimizer.state_dict())    
            
        # Perform random splitting of injected hyperedge
        split_attempts = 0
        injected_edges = hypergraph_attack.injected_edges
        original_edges = hypergraph_attack.original_edges
        flag = False

        while split_attempts < 2:    
            modified_injected_edges = hypergraph_attack.random_split_hyperedge(injected_edges)
            combined_edges = torch.cat((original_edges, modified_injected_edges), dim=1)
            # Update model with new hyperedges
            model, optimizer, _, _ = initialise(X, Y, G, args, modified_X, combined_edges)
            
            with torch.no_grad():
                # Calculate loss after the splitting
                Z = model(modified_X)
                modified_loss = F.nll_loss(Z[train_idx], Y[train_idx]).item()
                
            homophily_score_after_update = compute_homophily(combined_edges, modified_X, args.data, args.dataset)
            homophily_update_loss = homophily_loss(homophily_score_before, homophily_score_after_update, tolerance=0.1)
            
            # total_loss = modified_loss + lambda_h * homophily_update_loss
            total_loss = modified_loss - lambda_h * homophily_update_loss
            # If loss increases, accept the splitting and update the model parameters
            # if modified_loss > loss_value and homophily_update_loss < homo_budget:
            if total_loss > loss_value:
                modified_H = combined_edges  # Accept the splitting
                injected_edges = modified_injected_edges  
                loss_value = total_loss
                current_model_state = copy.deepcopy(model.state_dict())
                current_optimizer_state = copy.deepcopy(optimizer.state_dict())
                split_attempts = 0
                flag = True
                print("Splitting accepted, Loss increased to:", loss_value)
                # print(modified_H.size())
            else:
                print("Splitting rejected, Loss remains:", loss_value)
                # Restore model and optimizer state
                model.load_state_dict(current_model_state)
                optimizer.load_state_dict(current_optimizer_state)
                # model.train()
                # optimizer.zero_grad()
                # modified_X = modified_X.to(device)
                # Z = model(modified_X)
                # loss = F.nll_loss(Z[train_idx], Y[train_idx])
                # loss_value = loss.item()
                # loss.backward()
                # optimizer.step()
                split_attempts += 1
                
        if flag:
            model, optimizer, _, _ = initialise(X, Y, G, args, modified_X, modified_H)
            # Train the updated model
            tic_epoch = time.time()
            model.train()
            optimizer.zero_grad()
            modified_X = modified_X.to(device)
            Z = model(modified_X)
            classification_loss = F.nll_loss(Z[train_idx], Y[train_idx])
            # Calculate the homophily loss
            homophily_score_after_train = compute_homophily(modified_H, modified_X, args.data, args.dataset)
            homophily_train_loss = homophily_loss(homophily_score_before, homophily_score_after_train, tolerance=0.1)

            # Combine the classification loss and homophily loss
            # total_loss = classification_loss + lambda_h * homophily_train_loss
            total_loss = classification_loss - lambda_h * homophily_train_loss
            
            total_loss.backward()
            optimizer.step()
            
            train_time = time.time() - tic_epoch

            # Eval
            model.eval()
            Z = model(modified_X)
            update_train_acc = accuracy(Z[train_idx], Y[train_idx])
            update_test_acc = accuracy(Z[test_idx], Y[test_idx])
            update_best_test_acc = max(update_best_test_acc, update_test_acc)
            baselogger.info(f'epoch:{epoch} | loss after update:{loss:.4f} | train acc after update:{update_train_acc:.2f} | test acc after update:{update_test_acc:.2f} | time:{train_time*1000:.1f}ms')


    # visualize_distribution(X, modified_X, injected_indices, args.data, args.dataset)
    # pdb.set_trace()
    
    
    resultlogger.info(f"Run {run}/{args.n_runs}, best test accuracy after NIA: {new_best_test_acc:.2f}, acc(last) after NIA: {new_test_acc:.2f}, total time: {time.time()-tic_run:.2f}s")
        
    # Update the adjacency matrix with modified hyperedges
    G_modified = {i: set() for i in range(modified_H.shape[1])}
    for edge_idx in range(modified_H.shape[1]):
        nodes = [idx for idx, val in enumerate(modified_H[:, edge_idx]) if val > 0]
        G_modified[edge_idx] = set(nodes)

    # A_modified = compute_node_node_adjacency_matrix(G_modified, modified_X.shape[0])
    # Compute homophily score
    # new_X = new_X.cuda()
    # homophily_score_after = compute_homophily(new_H, new_X, args.data, args.dataset)
    
    homophily_score_update = compute_homophily(modified_H, modified_X, args.data, args.dataset)#, False, True)

    print(f'Average homophily score before attack: {homophily_score_before}')
    # print(f'Average homophily score after attack: {homophily_score_after}') # 注入攻击后没有更新超边，注入节点全连接一条超边
    print(f'Average homophily score after update: {homophily_score_update}')
    
    new_test_accs.append(new_test_acc)
    new_best_test_accs.append(new_best_test_acc)
    update_test_accs.append(update_test_acc)
    update_best_test_accs.append(update_best_test_acc)

baselogger.info(f'--------------Attack--ending-----k={ground_k}----------')
resultlogger.info(f"Average best test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")
# resultlogger.info(f"Average best test accuracy After NI: {np.mean(new_best_test_accs)} ± {np.std(new_best_test_accs)}")
resultlogger.info(f"Average best test accuracy After UpdateEdge: {np.mean(update_best_test_accs)} ± {np.std(update_best_test_accs)}")
print(update_best_test_accs) 
