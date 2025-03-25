import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import datetime
import shutil
import path
import config
from logger import get_logger
from data import data
from prepare import *
from NIA import HypergraphAttack
import random
import os
import time
import copy
import torch.nn.functional as F

criterion = nn.CrossEntropyLoss()

# Parse configuration arguments
args = config.parse()

# GPU and seed setup
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)

# Configure output directory
dataname = f'{args.data}_{args.dataset}'
model_name = args.model_name
nlayer = args.nlayer
dirname = f'{datetime.datetime.now()}'.replace(' ', '_').replace(':', '.')
out_dir = path.Path(f'./{args.out_dir}/{model_name}_{nlayer}_{dataname}/seed_{args.seed}')

if out_dir.exists():
    shutil.rmtree(out_dir)
out_dir.makedirs_p()

# Configure logger
baselogger = get_logger('base logger', f'{out_dir}/logging.log', not args.nostdout)
resultlogger = get_logger('result logger', f'{out_dir}/result.log', not args.nostdout)
baselogger.info(args)

# Load data
X, Y, G = fetch_data(args)
test_accs, best_val_accs, best_test_accs = [], [], []
new_test_accs, new_best_val_accs, new_best_test_accs = [], [], []
update_test_accs, update_best_val_accs, update_best_test_accs = [], [], []


resultlogger.info(args)

for run in range(1, args.n_runs + 1):
    run_dir = out_dir / f'{run}'
    run_dir.makedirs_p()

    # Load data
    args.split = run
    _, train_idx, test_idx = data.load(args)
    train_idx = torch.LongTensor(train_idx).cuda()
    test_idx = torch.LongTensor(test_idx).cuda()

    # Model initialization
    model, optimizer, X, H = initialise(X, Y, G, args)

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
        model.train()
        optimizer.zero_grad()
        Z = model(X)
        loss = F.nll_loss(Z[train_idx], Y[train_idx])
        loss.backward()
        optimizer.step()

        train_acc = accuracy(Z[train_idx], Y[train_idx])
        test_acc = accuracy(Z[test_idx], Y[test_idx])

        best_test_acc = max(best_test_acc, test_acc)
        baselogger.info(f'epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.2f} | test acc:{test_acc:.2f} | time:{(time.time() - tic_run) * 1000:.1f}ms')

    resultlogger.info(f"Run {run}/{args.n_runs}, best test accuracy: {best_test_acc:.2f}, acc(last): {test_acc:.2f}, total time: {time.time() - tic_run:.2f}s")
    test_accs.append(test_acc)
    best_test_accs.append(best_test_acc)

    # Node Injection Attack
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hypergraph_attack = HypergraphAttack(H, X, device=device)

    # Initialize RL environment parameters
    state_dim = X.shape[1] + H.shape[1]  # Node features + hyperedge features
    action_dim = int(X.shape[0] * 0.05)
   
   
    modified_X = None
    modified_H = None

    for epoch in range(args.epochs):
        
        if modified_X is not None and modified_H is not None:
            # 使用修改后的图结构初始化模型
            model, optimizer, _, _ = initialise(X, Y, G, args, modified_X, modified_H)

        else:
            # 使用原始图结构初始化模型
            model, optimizer, _, _ = initialise(X, Y, G, args)

        # Train
        model.train()
        optimizer.zero_grad()
        Z = model(X if modified_X is None else modified_X)
        loss = F.nll_loss(Z[train_idx], Y[train_idx])
        loss.backward()
        optimizer.step()
        
        for param in model.parameters():
            print(f"Gradient norm: {param.grad.norm()}")

        train_acc = accuracy(Z[train_idx], Y[train_idx])
        test_acc = accuracy(Z[test_idx], Y[test_idx])
        new_best_test_acc = 0
        baselogger.info(f'epoch:{epoch} | loss:{loss:.4f} | train acc:{train_acc:.2f} | test acc:{test_acc:.2f} | time:{(time.time() - tic_run) * 1000:.1f}ms')
        # Node Injection Attack
        k = int(H.shape[1] * 0.01) * 10  # 10% hyperedges
        modified_X, modified_H, ground_k = hypergraph_attack.inject_nodes(k, injection_ratio=0.05)
        
        # Define the state as the concatenation of modified_X and modified_H
        state = torch.cat((modified_X, modified_H), dim=1).to(device)
        # pdb.set_trace()
        model, optimizer, _, _ = initialise(X, Y, G, args, modified_X, modified_H)

        # Use the current model to predict actions (action probabilities)
        action_probs = model(modified_X)  # Ensure input shape matches the model's expected input
        # action_probs = model(state)
        action = torch.argmax(action_probs, dim=1)
        # 如果 action 只有一个元素，使用 .item() 获取标量值
        if action.numel() == 1:
            action = action.item()
        else:
            # 处理多个元素的情况，例如，取第一个元素
            action = action[0].item()
        # if len(action_probs.shape) == 2 and action_probs.shape[1] == action_dim:
        #     # 获取最大值的索引
        #     action = torch.argmax(action_probs, dim=1)
    
        #     # 如果 batch_size 为 1，使用 .item() 获取标量值
        #     if action.numel() == 1:
        #         action = action.item()
        #     else:
        #         # 处理多个元素的情况，例如，取第一个元素
        #         action = action[0].item()
        # else:
        #     raise ValueError(f"Unexpected shape for action_probs: {action_probs.shape}, expected [batch_size, {action_dim}]")

        print(f"Action probs: {action_probs}")
        print(f"Action: {action}")
    
        # Perform the action
        modified_injected_edges = hypergraph_attack.random_split_hyperedge(modified_H)
        modified_H = torch.cat((modified_H[:, :ground_k], modified_injected_edges), dim=1)
        
        # Evaluate the new model
        model, optimizer, new_X, new_H = initialise(X, Y, G, args, modified_X, modified_H)
        Z = model(modified_X)
        new_train_acc = accuracy(Z[train_idx], Y[train_idx])
        new_test_acc = accuracy(Z[test_idx], Y[test_idx])

        new_best_test_acc = max(new_best_test_acc, new_test_acc)
        baselogger.info(f'epoch:{epoch} | loss after NIA:{loss:.4f} | train acc after NIA:{new_train_acc:.2f} | test acc after NIA:{new_test_acc:.2f} | time:{(time.time() - tic_run) * 1000:.1f}ms')
        
        # Update RL policy
        optimizer.zero_grad()
        action_probs = torch.clamp(action_probs, min=1e-10, max=1 - 1e-10)
        reward = best_test_acc - new_test_acc
        print(f"Reward: {reward}")

        policy_loss = -torch.mean(torch.log(action_probs) * reward)
        # pdb.set_trace()
        policy_loss.backward()
        optimizer.step()

        print(f"Policy loss: {policy_loss.item()}")
        
        # Save model and optimizer state
        current_model_state = copy.deepcopy(model.state_dict())
        current_optimizer_state = copy.deepcopy(optimizer.state_dict())
        
        # Restore model and optimizer state
        model.load_state_dict(current_model_state)
        optimizer.load_state_dict(current_optimizer_state)

    resultlogger.info(f"Run {run}/{args.n_runs}, best test accuracy after NIA: {new_best_test_acc:.2f}, acc(last) after NIA: {new_test_acc:.2f}, total time: {time.time() - tic_run:.2f}s")
    new_test_accs.append(new_test_acc)
    new_best_test_accs.append(new_best_test_acc)
    update_test_accs.append(new_test_acc)
    update_best_test_accs.append(new_best_test_acc)

resultlogger.info(f"Average best test accuracy: {np.mean(best_test_accs)} ± {np.std(best_test_accs)}")
resultlogger.info(f"Average best test accuracy After NI: {np.mean(new_best_test_accs)} ± {np.std(new_best_test_accs)}")
resultlogger.info(f"Average best test accuracy After UpdateEdge: {np.mean(update_best_test_accs)} ± {np.std(update_best_test_accs)}")
