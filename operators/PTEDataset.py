import torch
from transformers import AutoTokenizer
from utils.saver import tokenizer_loader
from utils.constants import MASK_TOKEN, SEP_TOKEN

class PTEDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer=None, reader=None):
        assert tokenizer != None
        if type(tokenizer) == str:
            tokenizer = tokenizer_loader(AutoTokenizer, tokenizer)

        assert reader != None
        if type(reader) == str:
            reader = open(reader, mode="r", encoding="utf-8")

        self.tokenizer = tokenizer
        self.prompts = reader.read().strip("\n").split("\n")
        self.length = len(self.prompts)
        reader.close()

        text_tokenized = self.tokenizer(
            self.prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        self.shape = text_tokenized["input_ids"].shape
        self.inputs = text_tokenized
        self.labels = self.inputs["input_ids"].detach().clone()

        self.flag = set()
        for index in range(self.shape[0]):
            for sub_index in range(self.shape[1]):
                if int(self.inputs["input_ids"][index][sub_index]) == SEP_TOKEN:
                    self.flag.add(int(self.inputs["input_ids"][index][sub_index - 1]))
                    self.inputs["input_ids"][index][sub_index - 1] = MASK_TOKEN

        self.flag = tokenizer.convert_ids_to_tokens(list(self.flag))

    def __getitem__(self, index):
        return { key: self.inputs[key][index] for key in self.inputs.keys() }, self.labels[index]

    def __len__(self):
        return self.length
