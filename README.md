# HyperNear
This is the official pytorch implementation of the paper "HyperNear: Unnoticeable Node Injection Attacks on Hypergraph Neural Networks, ICML2025"

![model](https://github.com/ca1man-2022/HyperNear/blob/main/model.png)

Also, we provide an appendix [here](https://github.com/ca1man-2022/HyperNear/blob/main/Appendix_HyperNear.pdf).

*Thank you for your interest in our work! The authors are currently engaged in other projects but are working to tidy up and release a more readable and fully reproducible version of the code by December.*

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
