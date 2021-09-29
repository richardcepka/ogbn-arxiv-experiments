import torch
from torch_geometric.nn import ChebConv, SAGEConv

config = {
    # model parameters
    "model": {
        'input_dim': 128,  # ogbn-arxiv 128
        'output_dim': 40,  # ogbn-arxiv 40
        'hidden_dim': 128,
        'mlp_dim': 256,
        'dropout': 0.2,
        'num_layers': 10,
        'heads': [{'layer': ChebConv, 'params': {"K": 5},
                   'layer': SAGEConv, 'params': {}}]
    },
    # optimizer
    "opt": "sgd",  # adam, sgd, rmsprop, adagrad
    "lr": 0.1,
    "momentun": 0.9,
    "weight_decay": 0.0005,
    # scheduler
    "opt_scheduler": "step",  # step, cos, None
    "step_size": 200//3,
    "gamma": 0.1,
    # trainloader
    "train_batch": "cluster",  # cluster, walk, None
    # cluster
    "cluster": {
        "num_parts": 10,
        "batch_size": 1,
    },
    # walk
    "walk": {
        "batch_size": 10000,
        "walk_length": 2,
        "num_steps": 50
    },
    # train config
    "epochs": 200,
    "eval_every": 5,
    "device": torch.device('cuda')
}
