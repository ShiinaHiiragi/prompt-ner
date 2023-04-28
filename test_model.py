from pprint import pprint

def test_bert_base_chinese():

    from transformers import AutoTokenizer, BertForMaskedLM
    from transformers import pipeline
    from utils.saver import tokenizer_loader, model_loader

    tokenizer = tokenizer_loader(AutoTokenizer, "bert-base-chinese")
    model = model_loader(BertForMaskedLM, "bert-base-chinese")

    mask = pipeline("fill-mask", tokenizer=tokenizer, model=model)
    return mask("巴黎是[MASK]国的首都。")

ans = test_bert_base_chinese()
pprint(ans)
