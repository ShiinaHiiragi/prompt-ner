import torch
from transformers import AutoTokenizer

from .CONLLReader import CONLLReader
from .constants import SPECIAL_TOKENS, SPECIAL_ID
from .saver import tokenizer_loader

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, model=None, tokenizer=None, filename=None, reader=None):
        assert model != None or tokenizer != None
        if tokenizer == None:
            tokenizer = tokenizer_loader(AutoTokenizer, model)

        assert filename != None or reader != None
        if reader == None:
            reader = CONLLReader(filename)

        self.reader = reader
        self.max_size = self.reader.max_size + 2
        self.length = reader.length

        self.id_label = list(self.reader.domain)
        self.id_label.sort()
        self.num_labels = len(self.id_label)
        self.label_id = { item : index for index, item in enumerate(self.id_label) }

        text_tokenized = tokenizer(
            ["".join(line) for line in self.reader.sentences],
            padding='max_length',
            max_length=self.max_size,
            truncation=True,
            return_tensors="pt"
        )

        self.shape = text_tokenized["input_ids"].shape
        self.inputs = text_tokenized

        self.labels = torch.zeros(self.shape, dtype=torch.int64)
        for index in range(self.shape[0]):
            for sub_index in range(self.shape[1]):
                if int(self.inputs["input_ids"][index][sub_index]) in SPECIAL_TOKENS:
                    self.labels[index][sub_index] = SPECIAL_ID
                else:
                    self.labels[index][sub_index] = self.label_id[self.reader.labels[index][sub_index - 1]]

    def __getitem__(self, index):
        return { key: self.inputs[key][index] for key in self.inputs.keys() }, self.labels[index]

    def __len__(self):
        return self.length

def crf_mask(input_tensor):
    output_tensor = torch.ones_like(input_tensor, dtype=torch.bool)

    for index in range(input_tensor.shape[0]):
        for sub_index in range(input_tensor.shape[1]):
            if int(input_tensor[index][sub_index]) in SPECIAL_TOKENS:
                output_tensor[index][sub_index] = torch.tensor(0)

    return output_tensor
