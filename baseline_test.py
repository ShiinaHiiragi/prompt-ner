from transformers import AutoTokenizer
from utils.saver import tokenizer_loader
from utils.tester import baseline_test
from utils.constants import LOG
from operators.NERDataset import NERDataset
from operators.NERModel import NERModel

DATASET_NAME = "msra"
tokenizer = tokenizer_loader(AutoTokenizer, "bert-base-chinese")
test_dataset = NERDataset(tokenizer=tokenizer, reader=f"./data/{DATASET_NAME}.test")
model = NERModel(test_dataset.num_labels, bert_model="fine-tune/baseline-msra-epoch00")

test_acc = baseline_test(test_dataset, model)
LOG(f"TEST ACC: {test_acc}")
