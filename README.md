# HyperNear

## Train

```python
python train_NIA.py --method EDGNN --dname coauthor_dblp --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 2 --MLP_hidden 128 --Classifier_hidden 96 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 200 --runs 10 --cuda 1
```

```python
python train_NIA.py --method HGNN --HCHA_symdegnorm --cuda 2
```

```python
python train_NIA.py --method UniGCNII --cuda 0
```

```python
python train_NIA.py --method HyperGCN --cuda 0
```
