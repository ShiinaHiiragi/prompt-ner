import torch
from utils.metrics import calc_acc

LABEL_ENTITY = { "LOC": "地点", "ORG": "组织", "PER": "人名", "GPE": "地缘政治实体" }

def baseline_test(dataset, model):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset.length)
    _, (X, Y) = next(enumerate(test_loader))
    predict, ans, loss = model(X, Y)
    return calc_acc(predict, ans)
