import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.loader import GraphSAINTRandomWalkSampler


def load_ogbn_arxiv():

    dataset_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name=dataset_name,
                                     transform=T.ToSparseTensor())
    data = dataset[0]

    split_idx = dataset.get_idx_split()

    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        split_idx[key] = mask

    # change to the Data torch_geometric object
    row, col, edge_attr = data['adj_t'].t().coo()
    edge_index = torch.stack([row, col], dim=0).long()

    data = Data(x=data['x'],
                edge_index=edge_index,
                y=data['y'],
                train_mask=split_idx['train'],
                val_mask=split_idx['valid'],
                test_mask=split_idx['test'])

    evaluator = Evaluator(name='ogbn-arxiv')

    return data, evaluator


def load_loader(data, config):
    if config['train_batch'] == 'cluster':
        cluster_data = ClusterData(data,
                                   num_parts=config['cluster']['num_parts'],
                                   recursive=False)

        train_loader = ClusterLoader(cluster_data,
                                     batch_size=config['cluster']['batch_size'],
                                     shuffle=True)

    elif config['train_batch'] == 'walk':
        train_loader = GraphSAINTRandomWalkSampler(data,
                                                   batch_size=config['walk']['batch_size'],
                                                   walk_length=config['walk']['walk_length'],
                                                   num_steps=config['walk']['num_steps'])
    else:
        train_loader = None
    return train_loader


def train_bach(model, config, train_loader, optimizer, loss_fn):

    model.train()
    total_loss = 0
    total_nodes = 0
    for batch in train_loader:
        batch = batch.to(config["device"])
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)

        loss = loss_fn(out[batch.train_mask],
                       batch.y[batch.train_mask].squeeze(1))

        loss.backward()
        optimizer.step()

        nodes = batch.train_mask.sum().item()
        total_loss += loss.item() * nodes
        total_nodes += nodes

    return total_loss / total_nodes


def train_full(model, data, optimizer, loss_fn):

    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = loss_fn(out[data.train_mask],
                   data.y[data.train_mask].squeeze(1))

    loss.backward()
    optimizer.step()

    return loss.item()


@ torch.no_grad()
def test(model, config, data, evaluator):
    model.to('cpu')

    model.eval()
    # The output of model on all data
    logits = model(data.x,  data.edge_index)
    out = F.log_softmax(logits, dim=1)

    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': data.y[data.train_mask],
        'y_pred': y_pred[data.train_mask],
    })['acc']

    valid_acc = evaluator.eval({
        'y_true': data.y[data.val_mask],
        'y_pred': y_pred[data.val_mask],
    })['acc']

    test_acc = evaluator.eval({
        'y_true': data.y[data.test_mask],
        'y_pred': y_pred[data.test_mask],
    })['acc']

    model.to(config["device"])
    return train_acc, valid_acc, test_acc


def build_optimizer(config, params):

    if config['opt'] == 'adam':
        optimizer = optim.Adam(params,
                               lr=config['lr'],
                               weight_decay=config['weight_decay'])

    elif config['opt'] == 'sgd':
        optimizer = optim.SGD(params,
                              lr=config['lr'],
                              momentum=config['momentun'],
                              weight_decay=config['weight_decay'])

    elif config['opt'] == 'rmsprop':
        optimizer = optim.RMSprop(params,
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])

    elif config['opt'] == 'adagrad':
        optimizer = optim.Adagrad(params,
                                  lr=config['lr'],
                                  weight_decay=config['weight_decay'])

    if config['opt_scheduler'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=config['step_size'],
                                              gamma=config['gamma'])

    elif config['opt_scheduler'] == 'cos':
        scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                    num_warmup_steps=config["warmup_epochs"],
                                                    num_training_steps=config["epochs"])
    else:
        scheduler = None

    return scheduler, optimizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
