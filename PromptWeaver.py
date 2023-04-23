import re
from random import random, randint
from utils import han
from utils.segment import cut
from utils.constants import LABEL_ENTITY
from operators.PromptOperator import PromptOperator

class BartPromptOperator(PromptOperator):
    POSITIVE_TEMPLATE = "“{candidate_span}”是一个{entity_type}实体"
    NEGATIVE_TEMPLATE = "“{candidate_span}”不是一个命名实体"

    COUNTER_MAX = 100
    NEGATIVE_LOWER_BOND = 2

    def __init__(self, reader, ratio=1.5):
        super().__init__(reader)
        self.ratio = ratio

    def __generate_golden_entity(self, sentence, label):
        golden_entity = []
        new_item, new_tag = None, None
        for item, tag in zip(sentence, label):
            if re.match(r"B-", tag):
                golden_entity.append([item, tag[2:]])
            elif re.match(r"I-", tag):
                golden_entity[-1][0] += item
        return list(map(lambda lst: (lst[0], lst[1]), golden_entity))

    def __clear_invalid_words(self, words, entities):
        # clear punctuations
        right = len(words) - 1
        for index, item in enumerate(reversed(words)):
            if item in han.punctuation:
                words.pop(right - index)

        # clear positive entities
        for item in entities:
            if item in words:
                words.pop(words.index(item))

    def __generate_negative_entity(self, sentence, label, golden_entity):
        sentence_str = "".join(sentence)
        words = cut(sentence_str)
        segments = re.findall(han.segments, sentence_str)

        entities = list(map(lambda item: item[0], golden_entity))
        self.__clear_invalid_words(words, entities)

        negative_count = max(self.NEGATIVE_LOWER_BOND, round(len(entities) * self.ratio))
        while negative_count < len(words):
            words.pop(randint(0, len(words) - 1))

        counter = 0
        while negative_count > len(words):
            rand_segment = segments[randint(0, len(segments) - 1)]
            rand_range_left = randint(0, len(rand_segment) - 1)
            rand_range_right = randint(rand_range_left + 1, len(rand_segment))
            rand_span = rand_segment[rand_range_left:rand_range_right]
            if not rand_span in words:
                words.append(rand_span)

            if counter < self.COUNTER_MAX:
                counter += 1
            else:
                break

        return words

    def _format(self, sentence, label):
        result = []
        sentence_str = "".join(sentence)
        golden_entity = self.__generate_golden_entity(sentence, label)
        negative_entity = self.__generate_negative_entity(sentence, label, golden_entity)

        for item, tag in golden_entity:
            positive_format = self.POSITIVE_TEMPLATE.format(candidate_span=item, entity_type=LABEL_ENTITY[tag])
            result.append(f"{sentence_str}\t{positive_format}")

        for item in negative_entity:
            negative_format = self.NEGATIVE_TEMPLATE.format(candidate_span=item)
            result.append(f"{sentence_str}\t{negative_format}")

        return result

class EntailPromptOperator(PromptOperator):
    POSITIVE_FLAG = "对"
    NEGATIVE_FLAG = "错"

    # true positive:  「韩国」是地名实体
    # true negative:  「城市」不是命名实体
    TRUE_TEMPLATE = {
        "true_positive": "“{candidate_span}”是一个{entity_type}实体。" + POSITIVE_FLAG,
        "true_negative": "“{word_span}”不是一个命名实体。" + POSITIVE_FLAG
    }

    # false positive: 「城市」是地名实体
    # false positive: 「韩国」是人名实体
    # false negative: 「韩国」不是命名实体
    FALSE_TEMPLATE = {
        "false_positive_non": "“{word_span}”是一个{random_entity_type}实体。" + NEGATIVE_FLAG,
        "false_positive_entity": "“{candidate_span}”是一个{another_entity_type}实体。" + NEGATIVE_FLAG,
        "false_negative": "“{candidate_span}”不是一个命名实体。" + NEGATIVE_FLAG
    }

    COUNTER_MAX = 100
    NEGATIVE_LOWER_BOND = 1

    TRUE_POSITIVE_RATIO = 1
    FALSE_POSITIVE_RATIO = 0.75
    FALSE_NEGATIVE_RATIO = 0.75

    # F/T = 1 + 1/ratio
    def __init__(self, reader, ratio=1):
        super().__init__(reader)
        self.ratio = ratio

    def __generate_golden_entity(self, sentence, label):
        golden_entity = []
        new_item, new_tag = None, None
        for item, tag in zip(sentence, label):
            if re.match(r"B-", tag):
                golden_entity.append([item, tag[2:]])
            elif re.match(r"I-", tag):
                golden_entity[-1][0] += item
        return list(map(lambda lst: (lst[0], lst[1]), golden_entity))

    def __clear_invalid_words(self, words, entities):
        # clear punctuations
        right = len(words) - 1
        for index, item in enumerate(reversed(words)):
            if item in han.punctuation:
                words.pop(right - index)

        # clear positive entities
        for item in entities:
            if item in words:
                words.pop(words.index(item))

    def __generate_negative_entity(self, sentence, label, golden_entity):
        sentence_str = "".join(sentence)
        words = cut(sentence_str)
        segments = re.findall(han.segments, sentence_str)

        entities = list(map(lambda item: item[0], golden_entity))
        self.__clear_invalid_words(words, entities)

        negative_count = max(self.NEGATIVE_LOWER_BOND, round(len(entities) * self.ratio))
        while negative_count < len(words):
            words.pop(randint(0, len(words) - 1))

        counter = 0
        while negative_count > len(words):
            rand_segment = segments[randint(0, len(segments) - 1)]
            rand_range_left = randint(0, len(rand_segment) - 1)
            rand_range_right = randint(rand_range_left + 1, len(rand_segment))
            rand_span = rand_segment[rand_range_left:rand_range_right]
            if not rand_span in words:
                words.append(rand_span)

            if counter < self.COUNTER_MAX:
                counter += 1
            else:
                break

        return words

    def _format(self, sentence, label):
        result = []
        sentence_str = "".join(sentence)

        golden_entity = self.__generate_golden_entity(sentence, label)
        true_negative_entity = self.__generate_negative_entity(sentence, label, golden_entity)
        false_positive_entity = self.__generate_negative_entity(sentence, label, golden_entity)

        # true positive
        for item, tag in golden_entity:
            if random() > self.TRUE_POSITIVE_RATIO:
                continue
            positive_format = self.TRUE_TEMPLATE["true_positive"].format(
                candidate_span=item,
                entity_type=LABEL_ENTITY[tag]
            )
            result.append(f"{sentence_str}{positive_format}")

        # true negative
        for item in true_negative_entity:
            negative_format = self.TRUE_TEMPLATE["true_negative"].format(word_span=item)
            result.append(f"{sentence_str}{negative_format}")

        # false positive for non-entity
        entities_size = len(LABEL_ENTITY)
        entities_name = list(LABEL_ENTITY.values())
        for item in false_positive_entity:
            rand_index = randint(0, entities_size - 1)
            positive_format = self.FALSE_TEMPLATE["false_positive_non"].format(
                word_span=item,
                random_entity_type=entities_name[rand_index]
            )
            result.append(f"{sentence_str}{positive_format}")

        # false positive for entity
        for item, tag in golden_entity:
            if random() > self.FALSE_POSITIVE_RATIO:
                continue

            copy_entities_name = entities_name.copy()
            copy_entities_name.pop(copy_entities_name.index(LABEL_ENTITY[tag]))

            rand_index = randint(0, entities_size - 2)
            positive_format = self.FALSE_TEMPLATE["false_positive_entity"].format(
                candidate_span=item,
                another_entity_type=copy_entities_name[rand_index]
            )
            result.append(f"{sentence_str}{positive_format}")

        # false negative
        for item, _ in golden_entity:
            if random() > self.FALSE_NEGATIVE_RATIO:
                continue
            negative_format = self.FALSE_TEMPLATE["false_negative"].format(candidate_span=item)
            result.append(f"{sentence_str}{negative_format}")

        return result

if __name__ == "__main__":
    INPUT_DATASET_NAME = "msra.min"
    OUTPUT_DATASET_NAME = "msra.entail.min"

    prompt_op = EntailPromptOperator(f"./data/{INPUT_DATASET_NAME}.dev")
    prompt_op.dump(f"./prompts/{OUTPUT_DATASET_NAME}.dev.tsv")
