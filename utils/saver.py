import os
from transformers import PreTrainedTokenizerBase, PreTrainedModel

def __saver(model_loader, model_name, loader_type):
    local_path = f"./pretrained/{loader_type}/{model_name}"
    pretrained = model_loader.from_pretrained(model_name)
    pretrained.save_pretrained(local_path)

def __loader(model_loader, model_name, loader_type):
    local_path = f"./pretrained/{loader_type}/{model_name}"
    if not os.path.exists(local_path):
        __saver(model_loader, model_name, loader_type)
    return model_loader.from_pretrained(local_path)

def tokenizer_saver(model_loader, model_name):
    return __saver(model_loader, model_name, "tokenizer")

def model_saver(model_loader, model_name):
    return __saver(model_loader, model_name, "model")

def tokenizer_loader(model_loader, model_name):
    return __loader(model_loader, model_name, "tokenizer")

def model_loader(model_loader, model_name):
    return __loader(model_loader, model_name, "model")

if __name__ == "__main__":
    from transformers import AutoTokenizer
    from transformers import BertModel

    tokenizer_saver(AutoTokenizer, "bert-base-chinese")
    model_saver(BertModel, "bert-base-chinese")
