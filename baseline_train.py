import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from utils.constants import DEVICE, LOG
from utils.saver import tokenizer_loader
from utils.tester import baseline_test
from operators.NERDataset import NERDataset
from operators.NERModel import NERModel

LEARNING_RATE = 5e-3
EPOCH = 4
BATCH_SIZE = 32
DATASET_NAME = "msra"
MODEL_NAME = "baseline-msra"

tokenizer = tokenizer_loader(AutoTokenizer, "bert-base-chinese")
train_dataset = NERDataset(tokenizer=tokenizer, reader=f"./data/{DATASET_NAME}.train")
dev_dataset = NERDataset(tokenizer=tokenizer, reader=f"./data/{DATASET_NAME}.dev")
test_dataset = NERDataset(tokenizer=tokenizer, reader=f"./data/{DATASET_NAME}.test")
model = NERModel(train_dataset.num_labels, bert_model="bert-base-chinese")

LOG("LOADED")
assert train_dataset.id_label == dev_dataset.id_label
assert train_dataset.id_label == test_dataset.id_label

def train_loop(train_dataset, dev_dataset, model):
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for index in range(EPOCH):
        train(train_dataset, model, optimizer)
        dev_acc = baseline_test(dev_dataset, model)
        print(f"Epoch{index: 2d}: {dev_acc}")

def train(dataset, model, optimizer):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    for batch_index, (batch_X, batch_Y) in enumerate(tqdm(dataloader)):
        _, __, loss = model(batch_X, batch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

train_loop(train_dataset, dev_dataset, model)
model.save_pretrained(f"./pretrained/model/fine-tune/{MODEL_NAME}")
