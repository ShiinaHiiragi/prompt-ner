import torch

def calc_acc(predict, ans):
    correct, total = 0, 0
    for index in range(len(predict)):
        for sub_index in range(len(predict[index])):
            correct += 1 if predict[index][sub_index] == ans[index][sub_index] else 0
            total += 1

    return correct / total

def calc_f1(predict, ans):
    tp, tn, fp, fn = 0, 0, 0, 0
    for index in range(len(predict)):
        for sub_index in range(len(predict[index])):
            p = predict[index][sub_index]
            a = ans[index][sub_index]
            if p > 0 and a > 0:
                tp += 1
            elif p == 0 and a > 0:
                fn += 1
            elif p > 0 and a == 0:
                fp += 1
            else:
                tn += 1
    return (2 * tp) / (2 * tp + fp + fn)
