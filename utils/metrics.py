import torch

def flatten_2D(nested_list):
    return [item for sub_list in nested_list for item in sub_list]

def calc_acc(predict, ans):
    predict = torch.tensor(flatten_2D(predict))
    assert predict.shape == ans.shape

    correct, total = 0, predict.shape[0]
    for index in range(total):
        if predict[index]:
            correct += 1

    return correct / total
