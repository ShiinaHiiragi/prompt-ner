from transformers import BertTokenizer, BartForConditionalGeneration
from tqdm import tqdm

from utils.saver import tokenizer_loader, model_loader
from utils.constants import DEVICE, LOG
from utils.metrics import calc_acc, calc_f1_str
from utils.tester import predict_labels
from operators.CONLLReader import CONLLReader
from operators.NERDataset import NERDataset

DATASET_NAME = "msra"
dataset = NERDataset(
    reader=CONLLReader(f"./data/{DATASET_NAME}.test"),
    tokenizer=tokenizer_loader(BertTokenizer, "fnlp/bart-base-chinese")
)
model = BartForConditionalGeneration.from_pretrained("./outputs/best_model")
model.to(DEVICE)

infer_labels = []
for sentence in tqdm(dataset.reader.sentences):
    infer_label = predict_labels(model, dataset, "".join(sentence))
    infer_labels.append(infer_label)

test_acc = calc_acc(infer_labels, dataset.reader.labels)
test_f1 = calc_f1_str(infer_labels, dataset.reader.labels)
LOG(f"TEST ACC: {(test_acc, test_f1)}")
