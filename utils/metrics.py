import torch

def unify_format(predict, ans):
    predict = torch.tensor([item for sub_list in predict for item in sub_list])

    assert predict.shape == ans.shape
    return predict, ans

def calc_acc(predict, ans):
    predict, ans = unify_format(predict, ans)

    correct, total = 0, predict.shape[0]
    for index in range(total):
        if predict[index]:
            correct += 1

    return correct / total

def bart_calc_acc(predict, ans):
    correct, total = 0, 0
    for index in range(len(predict)):
        for sub_index in range(len(predict[index])):
            correct += 1 if predict[index][sub_index] == ans[index][sub_index] else 0
            total += 1

    return correct / total
