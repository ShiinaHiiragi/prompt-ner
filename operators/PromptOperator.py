import random
from operators.CONLLReader import CONLLReader

class PromptOperator:
    def __init__(self, reader=None):
        assert reader != None
        if type(reader) == str:
            reader = CONLLReader(reader)

        self.reader = reader

    def _format(self, sentence, label):
        raise NotImplementedError()

    def dump(self, filename):
        index_list = list(range(self.reader.length))
        random.shuffle(index_list)
        sentences_labels = list(zip(self.reader.sentences, self.reader.labels))
        shuffled = map(
            lambda index: sentences_labels[index],
            index_list
        )

        self.__container = []
        for sentence, label in shuffled:
            self.__container += self._format(sentence, label)

        with open(filename, "w") as f:
            for line in self.__container:
                f.write(line)
                f.write("\n")
