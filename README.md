# HyperNear

The code we currently have on GitHub is the version we used for the experiments. We will release a cleaner, easier-to-run version of the code later on. 🙂

## Train

```bash
python main.py --data cocitation --dataset cora --model-name UniGCN --add-self-loop --n-runs 5
```

```python
python train_NIA.py --method EDGNN --dname coauthor_dblp --All_num_layers 1 --MLP_num_layers 0 --MLP2_num_layers 0 --MLP3_num_layers 1 --Classifier_num_layers 2 --MLP_hidden 128 --Classifier_hidden 96 --aggregate mean --restart_alpha 0.0 --lr 0.001 --wd 0 --epochs 200 --runs 10 --n-runs 5
```

```python
python train_NIA.py --method HGNN --HCHA_symdegnorm --n-runs 5
```

```python
python train_NIA.py --method UniGCNII --n-runs 5
```

```python
python train_NIA.py --method HyperGCN --n-runs 5
```
