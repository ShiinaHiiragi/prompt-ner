import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from utils.constants import DEVICE, LOG
from utils.saver import tokenizer_loader
from utils.tester import baseline_test
from operators.CONLLReader import CONLLReader
from operators.NERDataset import NERDataset
from operators.NERModel import NERModel

LEARNING_RATE = 1e-5
EPOCH = 1
BATCH_SIZE = 4
DATASET_NAME = "msra"

train_reader = CONLLReader(f"./data/{DATASET_NAME}.train")
dev_lite_reader = CONLLReader(f"./data/{DATASET_NAME}.lite.dev")
tokenizer = tokenizer_loader(AutoTokenizer, "bert-base-chinese")

# test_reader = CONLLReader(f"./data/{DATASET_NAME}.test")
# assert train_reader.domain == dev_lite_reader.domain
# assert train_reader.domain == test_reader.domain

LOG("READER & TOKENIZER LOADED")
train_dataset = NERDataset(tokenizer=tokenizer, reader=train_reader)
dev_lite_dataset = NERDataset(tokenizer=tokenizer, reader=dev_lite_reader)
model = NERModel(train_dataset.num_labels, bert_model="bert-base-chinese")

def train_loop(train_dataset, dev_lite_dataset, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for index in range(EPOCH):
        train(train_dataset, dev_lite_dataset, model, optimizer)
        model.save_pretrained(f"./pretrained/model/fine-tune/baseline-{DATASET_NAME}-epoch{index:02d}.pt")

def train(train_dataset, dev_lite_dataset, model, optimizer):
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    for batch_index, (batch_X, batch_Y) in enumerate(tqdm(dataloader)):
        _, __, loss = model(batch_X, batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index > 0 and batch_index % 100 == 0:
            dev_acc = baseline_test(dev_lite_dataset, model)
            LOG(f"\nDEV ACC: {dev_acc}")

LOG("DATASET & MODEL LOADED")
train_loop(train_dataset, dev_lite_dataset, model)
