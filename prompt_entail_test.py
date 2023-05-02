from transformers import AutoTokenizer, BertForMaskedLM
from utils.saver import tokenizer_loader, model_loader
from utils.constants import DEVICE, LOG
from utils.tester import entail_test
from utils.metrics import calc_acc, calc_f1_str
from operators.CONLLReader import CONLLReader

DATASET_NAME = "msra"
EPOCH_INDEX = 1

tokenizer = tokenizer_loader(AutoTokenizer, "bert-base-chinese")
model = model_loader(BertForMaskedLM, f"fine-tune/prompt-{DATASET_NAME}-entail-epoch-{EPOCH_INDEX:02d}")
test_reader = CONLLReader(filename=f"./data/{DATASET_NAME}.test")

model.to(DEVICE)
predict = entail_test(model, tokenizer, test_reader)
print(predict, test_reader.labels)
test_acc, test_f1 = calc_acc(predict, test_reader.labels), calc_f1_str(predict, test_reader.labels)
LOG(f"TEST ACC: {(test_acc, test_f1)}")
