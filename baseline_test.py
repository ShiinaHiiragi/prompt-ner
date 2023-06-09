import torch
from transformers import AutoTokenizer
from utils.saver import tokenizer_loader
from utils.tester import baseline_test
from utils.constants import LOG
from operators.NERDataset import NERDataset
from operators.NERModel import NERModel

DATASET_NAME = "msra"
EPOCH_INDEX = 0

tokenizer = tokenizer_loader(AutoTokenizer, "bert-base-chinese")
test_dataset = NERDataset(tokenizer=tokenizer, reader=f"./data/{DATASET_NAME}.test")
model = torch.load(f"./pretrained/model/fine-tune/baseline-{DATASET_NAME}-epoch-{EPOCH_INDEX:02d}.pt")

LOG("DATASET & MODEL LOADED")
test_acc = baseline_test(test_dataset, model)
LOG(f"TEST ACC: {test_acc}")
