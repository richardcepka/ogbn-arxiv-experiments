from utils import load_ogbn_arxiv, build_optimizer, train_bach, train_full, test, load_loader, count_parameters
from config import config
import torch
import torch.nn as nn
from MultiGNN import GNN


seed = 1297978
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    data, evaluator = load_ogbn_arxiv()
    train_loader = load_loader(data, config)

    if train_loader is None:
        data = data.to(config["device"])

    model = GNN(**config["model"]).to(config["device"])
    print(f'number of trainable parameters: {count_parameters(model)}')

    scheduler, optimizer = build_optimizer(config, model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 1 + config["epochs"]):
        # train
        if train_loader is None:
            loss = train_full(model, data, optimizer, loss_fn)
        else:
            loss = train_bach(model, config, train_loader, optimizer, loss_fn)

        # scheduler
        if scheduler is not None:
            scheduler.step()

        # eval
        if epoch % config["eval_every"] == 0:
            train_acc, valid_acc, test_acc = test(
                model, config, data, evaluator)
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}% '
                  f'Test: {100 * test_acc:.2f}%')
