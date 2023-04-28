import torch

def bart_calc_acc(predict, ans):
    correct, total = 0, 0
    for index in range(len(predict)):
        for sub_index in range(len(predict[index])):
            correct += 1 if predict[index][sub_index] == ans[index][sub_index] else 0
            total += 1

    return correct / total
