from pprint import pprint

def truncate():
    from operators.CONLLReader import CONLLReader

    reader = CONLLReader("./data/msra.dev")
    reader.sentences = reader.sentences[:100]
    reader.labels = reader.labels[:100]
    reader.dump("./data/msra.lite.dev")

def test_bert_base_chinese():
    from transformers import AutoTokenizer, BertForMaskedLM
    from transformers import pipeline
    from utils.saver import tokenizer_loader, model_loader

    tokenizer = tokenizer_loader(AutoTokenizer, "bert-base-chinese")
    model = model_loader(BertForMaskedLM, "bert-base-chinese")

    mask = pipeline("fill-mask", tokenizer=tokenizer, model=model)
    return mask("巴黎是[MASK]国的首都。")

def test_bart_base_chinese():
    from transformers import BertTokenizer, BartForConditionalGeneration
    from transformers import pipeline
    from utils.saver import tokenizer_loader, model_loader

    tokenizer = tokenizer_loader(BertTokenizer, "fnlp/bart-base-chinese")
    model = model_loader(BartForConditionalGeneration, "fnlp/bart-base-chinese")

    generator = pipeline("text2text-generation", tokenizer=tokenizer, model=model)
    return generator("北京是[MASK]的首都")

ans = test_bart_base_chinese()
pprint(ans)
