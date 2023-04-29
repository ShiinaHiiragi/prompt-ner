import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertForMaskedLM

from utils.saver import tokenizer_loader, model_loader
from utils.constants import MASK_TOKEN, LOG, DEVICE
from utils.tester import find_token
from utils.metrics import calc_acc, calc_f1
from operators.PTEDataset import PTEDataset
from PromptWeaver import EntailPromptOperator

DATASET_NAME = "min.entail"
LEARNING_RATE = 1e-5
EPOCH = 1
BATCH_SIZE = 4

tokenizer = tokenizer_loader(AutoTokenizer, "bert-base-chinese")
model = model_loader(BertForMaskedLM, "bert-base-chinese")
model.to(DEVICE)

train_dataset = PTEDataset(tokenizer=tokenizer, reader=f"./prompts/{DATASET_NAME}.train.tsv")
dev_dataset = PTEDataset(tokenizer=tokenizer, reader=f"./prompts/{DATASET_NAME}.dev.tsv")
dev_lite_dataset = PTEDataset(tokenizer=tokenizer, reader=f"./prompts/{DATASET_NAME}.lite.dev.tsv")

def train_loop(train_dataset, dev_dataset, dev_lite_dataset, model, tokenizer):
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for index in range(EPOCH):
        train(train_dataset, dev_lite_dataset, model, tokenizer, optimizer)
        model.save_pretrained(f"./pretrained/model/fine-tune/prompt-{DATASET_NAME.replace('.', '-')}-epoch-{index:02d}")
        dev_acc = validate(dev_dataset, model, tokenizer)
        LOG(f"\nEpoch {index:02d}: {dev_acc}")

def train(train_dataset, dev_dataset, model, tokenizer, optimizer):
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE)
    for batch_index, (batch_X, batch_Y) in enumerate(tqdm(dataloader)):
        for key in batch_X.keys():
            batch_X[key] = batch_X[key].to(DEVICE)
        batch_Y = batch_Y.to(DEVICE)

        loss = model(**batch_X, labels=batch_Y).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index > 0 and batch_index % 1 == 0:
            dev_acc = validate(dev_dataset, model, tokenizer)
            LOG(f"\nDEV ACC: {dev_acc}")

def validate(dataset, model, tokenizer):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    _, (X, Y) = next(enumerate(dataloader))

    positive_token = tokenizer.convert_tokens_to_ids(EntailPromptOperator.POSITIVE_FLAG)
    negative_token = tokenizer.convert_tokens_to_ids(EntailPromptOperator.NEGATIVE_FLAG)
    mask_index = (X["input_ids"] == MASK_TOKEN).nonzero()

    with torch.no_grad():
        for key in X.keys():
            X[key] = X[key].to(DEVICE)

        outputs = model(**X)[0]
        all_predict, all_ans = [[]], [[]]
        for index in tqdm(mask_index):
            prob_vector = outputs[index[0], index[1]]
            ans = Y[index[0], index[1]]
            positive, negative = prob_vector[positive_token], prob_vector[negative_token]
            all_predict[0].append(1 if positive > negative else 0)
            all_ans[0].append(1 if ans == positive_token else 0)
        return calc_acc(all_predict, all_ans), calc_f1(all_predict, all_ans)

LOG(train_dataset.flag)
train_loop(train_dataset, dev_dataset, dev_lite_dataset, model, tokenizer)
