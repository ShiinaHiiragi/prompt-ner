import torch
from utils.metrics import calc_acc
from utils.constants import DEVICE, LABEL_ENTITY
from PromptWeaver import BartPromptOperator

GRAM = 4

def baseline_test(dataset, model):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=dataset.length)
    _, (X, Y) = next(enumerate(test_loader))
    predict, ans, loss = model(X, Y)
    return calc_acc(predict, ans)

def calc_labels_entity(dataset):
    labels = list(set(
        map(
            lambda item: item[2:],
            filter(
                lambda item: item != "O",
                dataset.id_label
            )
        )
    ))

    return { item: LABEL_ENTITY[item] for item in labels }

def generate_template(sentence_str, start_point, part_labels_entity):
    result = []
    for span_size in range(1, GRAM + 1):
        span = sentence_str[start_point:start_point + span_size]
        result.append(BartPromptOperator.NEGATIVE_TEMPLATE.format(candidate_span=span))
        for entity in part_labels_entity.keys():
            result.append(BartPromptOperator.POSITIVE_TEMPLATE.format(
                candidate_span=span,
                entity_type=part_labels_entity[entity]
            ))

    def find_tag(index):
        part_labels = list(part_labels_entity.keys())
        part_labels.insert(0, "O")
        label_size = len(part_labels)

        span_size = index // label_size + 1
        span_type = index % label_size
        return span_size, part_labels[span_type]

    return result, find_tag

def calc_max_possible(model, tokenizer, sentence_str, templates):
    batch_size = len(templates)
    inputs_id = tokenizer([sentence_str] * batch_size, return_tensors="pt")["input_ids"]
    outputs = tokenizer(templates, return_tensors="pt", padding=True)
    outputs_id = outputs["input_ids"]

    outputs_id_length = torch.sum(outputs["attention_mask"], axis=1) - 2
    max_outputs_id_length = int(max(outputs_id_length))

    score = [1] * batch_size
    with torch.no_grad():
        logits = model(input_ids=inputs_id.to(DEVICE), decoder_input_ids=outputs_id.to(DEVICE))[0]
        for token_index in range(max_outputs_id_length):
            single_logits = logits[:, token_index, :].softmax(dim=1).to('cpu').numpy()
            for sentence_index in range(batch_size):
                if token_index < outputs_id_length[sentence_index]:
                    next_token_id = int(outputs_id[sentence_index, token_index + 1])
                    score[sentence_index] *= single_logits[sentence_index][next_token_id]

    max_score = max(score)
    return score.index(max_score), max_score

def mark_label(predict, sentence_size):
    def is_intersect(left, right):
        if left[1] < right[0] or left[0] > right[1]:
            return False
        return True

    left = 0
    while left < len(predict):
        right = left + 1
        while right < len(predict):
            if is_intersect(predict[left]["interval"], predict[right]["interval"]):
                if predict[left]["score"] < predict[right]["score"]:
                    predict[left], predict[right] = predict[right], predict[left]
                predict.pop(right)
            else:
                right += 1
        left += 1

    labels = ["O"] * sentence_size
    for item in predict:
        left, right = item["interval"]
        labels[left:right] = [f"I-{item['type']}"] * (right - left)
        labels[left] = f"B-{item['type']}"

    return labels

def predict_labels(model, dataset, sentence_str):
    right = len(sentence_str) - GRAM + 1
    part_labels_entity = calc_labels_entity(dataset)

    predict = []
    for start_point in range(0, right):
        templates, find_tag = generate_template(sentence_str, start_point, part_labels_entity)
        max_index, max_score = calc_max_possible(model, dataset.tokenizer, sentence_str, templates)
        span_size, span_type = find_tag(max_index)
        if span_type != "O":
            predict.append({
                "interval": (start_point, start_point + span_size),
                "type": span_type,
                "score": max_score
            })

    labels = mark_label(predict, len(sentence_str))
    return labels
