import numpy as np

class CONLLReader:
    def __init__(self, filename):
        self.filename = filename

        self.__calculate_info()
        self.__calculate_domain()

    def __calculate_info(self):
        sentences = [[]]
        labels = [[]]
        size, max_size = 0, 0
        with open(self.filename, "r") as f:
            for line in f:
                items = line.strip().split("\t")
                if len(items) > 1:
                    sentences[-1].append(items[0])
                    labels[-1].append(items[1])
                    size += 1
                else:
                    sentences.append([])
                    labels.append([])
                    max_size = max(size, max_size)
                    size = 0

        self.sentences = sentences
        self.labels = labels

        assert len(self.sentences) == len(self.labels)
        self.length = len(self.sentences)
        self.max_size = max_size

    def __calculate_domain(self):
        self.domain = set()
        for line in self.labels:
            for item in line:
                self.domain.add(item)

    def dump(self, filename):
        outer_size = self.length()
        with open(filename, "w") as f:
            for index in range(outer_size):
                inner_size = len(self.sentences[index])
                for sub_index in range(inner_size):
                    f.write(f"{self.sentences[index][sub_index]}\t{self.labels[index][sub_index]}\n")
                f.write("\n")
