import re
from random import random, randint
from utils import han
from utils.segment import cut
from utils.constants import LABEL_ENTITY, LOG
from operators.PromptOperator import PromptOperator

class BartPromptOperator(PromptOperator):
    POSITIVE_TEMPLATE = "“{candidate_span}”是一个{entity_type}实体"
    NEGATIVE_TEMPLATE = "“{candidate_span}”不是一个命名实体"

    # 实体：POSITIVE_RATIO 概率可以保留
    # 非实体：NEGATIVE_RATIO 概率可以保留
    # 　　　　非实体是实体的 POSITIVE_NEGATIVE_RATIO 倍，最低为 NEGATIVE_LOWER_BOND(ratio) 个
    # 　　　　该函数有 ratio 概率输出 1，有 1 - ratio 概率输出 0，默认为 DEFAULT_LOWER_BOND
    POSITIVE_RATIO = 1
    NEGATIVE_RATIO = 1
    POSITIVE_NEGATIVE_RATIO = 1
    DEFAULT_LOWER_BOND = 0.25
    NEGATIVE_LOWER_BOND = lambda self, ratio: 0 if random() > ratio else 1

    def __init__(self, reader, dropout=1):
        super().__init__(reader)
        self.dropout = dropout

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

        negative_count = max(
            self.NEGATIVE_LOWER_BOND(self.DEFAULT_LOWER_BOND),
            round(len(entities) * self.POSITIVE_NEGATIVE_RATIO)
        )

        while negative_count < len(words):
            words.pop(randint(0, len(words) - 1))
        return words

    def _format(self, sentence, label):
        if random() > self.dropout:
            return []

        result = []
        sentence_str = "".join(sentence)
        golden_entity = self.__generate_golden_entity(sentence, label)
        negative_entity = self.__generate_negative_entity(sentence, label, golden_entity)

        for item, tag in golden_entity:
            if random() > self.POSITIVE_RATIO:
                continue
            positive_format = self.POSITIVE_TEMPLATE.format(candidate_span=item, entity_type=LABEL_ENTITY[tag])
            result.append(f"{sentence_str}\t{positive_format}")

        for item in negative_entity:
            if random() > self.NEGATIVE_RATIO:
                continue
            negative_format = self.NEGATIVE_TEMPLATE.format(candidate_span=item)
            result.append(f"{sentence_str}\t{negative_format}")

        return result

class EntailPromptOperator(PromptOperator):
    POSITIVE_FLAG = "对"
    NEGATIVE_FLAG = "错"
    MASK_FLAG = "[MASK]"


    TRUE_TEMPLATE = {
        "test_positive": "“{candidate_span}”是一个{entity_type}实体。" + MASK_FLAG,
        "true_positive": "“{candidate_span}”是一个{entity_type}实体。" + POSITIVE_FLAG,
        "test_negative": "“{word_span}”不是一个命名实体。" + MASK_FLAG,
        "true_negative": "“{word_span}”不是一个命名实体。" + POSITIVE_FLAG
    }


    FALSE_TEMPLATE = {
        "false_positive_non": "“{word_span}”是一个{random_entity_type}实体。" + NEGATIVE_FLAG,
        "false_positive_entity": "“{candidate_span}”是一个{another_entity_type}实体。" + NEGATIVE_FLAG,
        "false_negative": "“{candidate_span}”不是一个命名实体。" + NEGATIVE_FLAG
    }

    # true positive:  「韩国」是地名实体
    # true negative:  「城市」不是命名实体
    # false positive: 「城市」是地名实体
    # false positive: 「韩国」是人名实体
    # false negative: 「韩国」不是命名实体

    # TP：实体，TRUE_POSITIVE_RATIO 概率保留
    # TN：非实体，TRUE_NEGATIVE_RATIO 概率保留
    # FP：分为实体与非实体（与 TN 独立筛选）部分，两者均有 FALSE_POSITIVE_RATIO 概率保留
    # FN：实体，FALSE_NEGATIVE_RATIO 概率保留
    # 非实体：非实体是实体的 POSITIVE_NEGATIVE_RATIO 倍，最低为 NEGATIVE_LOWER_BOND(ratio) 个
    # 　　　　该函数有 ratio 概率输出 1，有 1 - ratio 概率输出 0，默认为 DEFAULT_LOWER_BOND
    TRUE_POSITIVE_RATIO = 1
    TRUE_NEGATIVE_RATIO = 0.25
    FALSE_POSITIVE_RATIO = 0.25
    FALSE_NEGATIVE_RATIO = 0.25

    POSITIVE_NEGATIVE_RATIO = 0.5
    DEFAULT_LOWER_BOND = 0.25
    NEGATIVE_LOWER_BOND = lambda self, ratio: 0 if random() > ratio else 1

    def __init__(self, reader, dropout=1):
        super().__init__(reader)
        self.dropout = dropout

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

        negative_count = max(
            self.NEGATIVE_LOWER_BOND(self.DEFAULT_LOWER_BOND),
            round(len(entities) * self.POSITIVE_NEGATIVE_RATIO)
        )

        while negative_count < len(words):
            words.pop(randint(0, len(words) - 1))
        return words

    def _format(self, sentence, label):
        if random() > self.dropout:
            return []

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
            if random() > self.TRUE_NEGATIVE_RATIO:
                continue
            negative_format = self.TRUE_TEMPLATE["true_negative"].format(word_span=item)
            result.append(f"{sentence_str}{negative_format}")

        # false positive for non-entity
        entities_size = len(LABEL_ENTITY)
        entities_name = list(LABEL_ENTITY.values())
        for item in false_positive_entity:
            if random() > self.FALSE_POSITIVE_RATIO:
                continue
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
    OperatorClass = {
        "bart": BartPromptOperator,
        "entail": EntailPromptOperator
    }

    for dataset_name in ["min", "msra", "weibo"]:
        for prompt_class in OperatorClass.keys():
            for suffix in ["train", "dev", "lite.dev"]:
                LOG(f"WRITING prompts/{dataset_name}.{prompt_class}.{suffix}.tsv")
                prompt_op = OperatorClass[prompt_class](f"./data/{dataset_name}.{suffix}")
                prompt_op.dump(f"./prompts/{dataset_name}.{prompt_class}.{suffix}.tsv")
