import torch
from utils.metrics import calc_acc

def baseline_test(dataset, model):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset.length)
    _, (X, Y) = next(enumerate(test_loader))
    predict, ans, loss = model(X, Y)
    return calc_acc(predict, ans)
