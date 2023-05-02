import torch
import pandas as pd
from utils.seq2seq_model import Seq2SeqModel
from simpletransformers.seq2seq import Seq2SeqArgs

DATASET_NAME = "min"
train_data = pd.read_csv(f"./prompts/{DATASET_NAME}.bart.train.tsv", sep="\t").values.tolist()
dev_data = pd.read_csv(f"./prompts/{DATASET_NAME}.bart.dev.tsv", sep="\t").values.tolist()
train_df = pd.DataFrame(train_data, columns=["input_text", "target_text"])
dev_df = pd.DataFrame(dev_data, columns=["input_text", "target_text"])

model_args = Seq2SeqArgs()
model_args.num_train_epochs = 2
model_args.evaluate_generated_text = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True
model_args.overwrite_output_dir = True

model = Seq2SeqModel(
    encoder_decoder_type="bart",
    encoder_decoder_name="fnlp/bart-base-chinese",
    use_cuda=torch.cuda.is_available(),
    args=model_args
)

model.train_model(train_df, eval_data=dev_df)
model.eval_model(dev_df)
model.model.save_pretrained("./pretrained/model/fine-tune/prompt-bart")
