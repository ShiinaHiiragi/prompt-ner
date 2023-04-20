import torch
from transformers import BertModel
from torchcrf import CRF

from operators.NERDataset import crf_mask
from utils.saver import model_loader

class NERModel(torch.nn.Module):
    def __init__(self, num_labels, bert_model=None):
        super(NERModel, self).__init__()

        assert bert_model != None
        if type(bert_model) == str:
            bert_model = model_loader(BertModel, bert_model)

        self.bert = bert_model
        self.dropout = torch.nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, inputs, labels=None):
        bert_outputs = self.bert(**inputs)
        dropout_output = self.dropout(bert_outputs[0])
        classifier_output = self.classifier(dropout_output)

        crf_emissions = classifier_output[:,1:,:]
        crf_masks = crf_mask(inputs["input_ids"])[:,1:]
        predict = self.crf.decode(crf_emissions, mask=crf_masks)

        if labels != None:
            crf_tags = labels[:,1:]
            loss = self.crf(crf_emissions, crf_tags, mask=crf_masks) * (-1)
            return predict, crf_tags[crf_masks == 1], loss

        return predict

    def save_pretrained(self, path):
        self.bert.save_pretrained(path)
