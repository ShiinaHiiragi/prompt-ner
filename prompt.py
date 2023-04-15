import re
from random import randint
from utils.lib import han
from utils.PromptOperator import PromptOperator
from utils.segment import cut

class BartPromptOperator(PromptOperator):
    POSITIVE_TEMPLATE = "“{candidate_span}”是一个{entity_type}实体"
    NEGATIVE_TEMPLATE = "“{candidate_span}”不是一个命名实体"
    LABEL_ENTITY = { "LOC": "地点", "ORG": "组织", "PER": "人名", "GPE": "地缘政治实体" }

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
            result.append(
                sentence_str + 
                self.POSITIVE_TEMPLATE.format(
                    candidate_span=item,
                    entity_type=self.LABEL_ENTITY[tag]
                )
            )

        for item in negative_entity:
            result.append(sentence_str + self.NEGATIVE_TEMPLATE.format(candidate_span=item))

        return result

if __name__ == "__main__":
    prompt_op = BartPromptOperator("./data/msra.min.dev")
    prompt_op.dump("./test")