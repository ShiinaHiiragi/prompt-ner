import torch
from transformers import AutoTokenizer, BertForMaskedLM

from utils.saver import tokenizer_loader, model_loader
from utils.constants import MASK_TOKEN, LOG
from utils.tester import find_token
from operators.PTEDataset import PTEDataset
from PromptWeaver import EntailPromptOperator

DATASET_NAME = "min.entail"
LEARNING_RATE = 1e-5
EPOCH = 1
BATCH_SIZE = 4

tokenizer = tokenizer_loader(AutoTokenizer, "bert-base-chinese")
model = model_loader(BertForMaskedLM, "bert-base-chinese")
train_dataset = PTEDataset(tokenizer=tokenizer, reader=f"./prompts/{DATASET_NAME}.train.tsv")
dev_dataset = PTEDataset(tokenizer=tokenizer, reader=f"./prompts/{DATASET_NAME}.dev.tsv")

def train_loop(train_dataset, dev_dataset, model, tokenizer):
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for _ in range(EPOCH):
        train(train_dataset, model, optimizer)
        dev_acc = validate(dev_dataset, model, tokenizer)
        print(dev_acc)

def train(dataset, model, optimizer):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    for batch_index, (batch_X, batch_Y) in enumerate(dataloader):
        loss = model(**batch_X, labels=batch_Y).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(dataset, model, tokenizer):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    _, (X, Y) = next(enumerate(dataloader))

    positive_token = tokenizer.convert_tokens_to_ids(EntailPromptOperator.POSITIVE_FLAG)
    negative_token = tokenizer.convert_tokens_to_ids(EntailPromptOperator.NEGATIVE_FLAG)
    mask_index = (X["input_ids"] == MASK_TOKEN).nonzero()

    correct, total = 0, 0
    with torch.no_grad():
        outputs = model(**X)[0]
        for index in mask_index:
            prob_vector = outputs[index[0], index[1]]
            ans = Y[index[0], index[1]]
            positive, negative = prob_vector[positive_token], prob_vector[negative_token]
            if (positive > negative and ans == positive_token) or \
                (positive < negative and ans == negative_token):
                correct += 1
            total += 1

    return correct / total

LOG(train_dataset.flag)
train_loop(train_dataset, dev_dataset, model, tokenizer)
model.save_pretrained(f"./pretrained/model/fine-tune/prompt-entail")
