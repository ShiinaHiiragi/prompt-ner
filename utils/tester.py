import torch
from tqdm import tqdm
from utils.segment import cut
from utils.constants import DEVICE, LABEL_ENTITY, NULL_LABEL, MASK_TOKEN, GRAM
from utils.metrics import calc_acc, calc_f1
from PromptWeaver import BartPromptOperator, EntailPromptOperator

def baseline_test(dataset, model):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    with torch.no_grad():
        all_predict, all_ans = [], []
        for batch_index, (batch_X, batch_Y) in enumerate(tqdm(test_loader)):
            predict, ans, loss = model(batch_X, batch_Y)
            all_predict.append(predict[0])
            all_ans.append(ans.to("cpu").tolist())
        return calc_acc(all_predict, all_ans), calc_f1(all_predict, all_ans)

def find_token(tokenizer):
    positive_token = tokenizer.convert_tokens_to_ids(EntailPromptOperator.POSITIVE_FLAG)
    negative_token = tokenizer.convert_tokens_to_ids(EntailPromptOperator.NEGATIVE_FLAG)

    return {
        EntailPromptOperator.POSITIVE_FLAG: positive_token,
        EntailPromptOperator.NEGATIVE_FLAG: negative_token
    }, {
        positive_token: EntailPromptOperator.POSITIVE_FLAG,
        negative_token: EntailPromptOperator.NEGATIVE_FLAG
    }

def calc_labels_entity(dataset):
    labels = list(set(
        map(
            lambda item: item[2:],
            filter(
                lambda item: item != NULL_LABEL,
                dataset.id_label
            )
        )
    ))

    return { item: LABEL_ENTITY[item] for item in labels }

def calc_prob_score(positive, negative):
    return positive - negative

def predict_word_cut(model, tokenizer, sentence_str, word, flag_token):
    label_entity_keys = list(LABEL_ENTITY.keys())

    test_positive = list(map(
        lambda key: sentence_str + EntailPromptOperator.TRUE_TEMPLATE["test_positive"].format(
            candidate_span=word,
            entity_type=LABEL_ENTITY[key]
        ),
        label_entity_keys
    ))
    test_negative = sentence_str + EntailPromptOperator.TRUE_TEMPLATE["test_negative"].format(word_span=word)

    positive_inputs = tokenizer(
        test_positive,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    negative_input = tokenizer(test_negative, return_tensors="pt")
    positive_mask_index = (positive_inputs["input_ids"] == MASK_TOKEN).nonzero()
    negative_mask_index = (negative_input["input_ids"] == MASK_TOKEN).nonzero()

    result = []
    with torch.no_grad():
        for key in positive_inputs.keys():
            positive_inputs[key].to(DEVICE)

        for key in negative_input.keys():
            negative_input[key].to(DEVICE)

        positive_outputs = model(**positive_inputs)[0]
        negative_output = model(**negative_input)[0]

        positive_token = flag_token[EntailPromptOperator.POSITIVE_FLAG]
        negative_token = flag_token[EntailPromptOperator.NEGATIVE_FLAG]

        for batch, index in enumerate(positive_mask_index):
            positive_prob_vector = positive_outputs[index[0]][index[1]]
            result.append((
                label_entity_keys[batch],
                calc_prob_score(
                    float(positive_prob_vector[positive_token]),
                    float(positive_prob_vector[negative_token])
                )
            ))

        negative_prob_vector = negative_output[negative_mask_index[0, 0]][negative_mask_index[0, 1]]
        result.append((
            NULL_LABEL,
            calc_prob_score(
                float(negative_prob_vector[positive_token]),
                float(negative_prob_vector[negative_token])
            )
        ))

    return max(result, key=lambda item: item[1])

def entail_test(model, tokenizer, reader):
    flag_token, _ = find_token(tokenizer)

    predicts = []
    for sentence in reader.sentences:
        sentence_str = "".join(sentence)
        words = cut(sentence_str)
        predict = []
        for word in words:
            word_result = predict_word_cut(model, tokenizer, sentence_str, word, flag_token)[0]
            for idx, ch in enumerate(word):
                if word_result != NULL_LABEL:
                    predict.append(f"I-{word_result}" if idx else f"B-{word_result}")
                else:
                    predict.append(word_result)
        predicts.append(predict)

    return predicts

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
        part_labels.insert(0, NULL_LABEL)
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

    labels = [NULL_LABEL] * sentence_size
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
        if span_type != NULL_LABEL:
            predict.append({
                "interval": (start_point, start_point + span_size),
                "type": span_type,
                "score": max_score
            })

    labels = mark_label(predict, len(sentence_str))
    return labels
