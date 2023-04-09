import torch
from transformers import BertModel
from torchcrf import CRF

from .NERDataset import crf_mask
from .saver import model_loader

class NERModel(torch.nn.Module):
    def __init__(self, num_labels, model_name=None, bert_model=None):
        super(NERModel, self).__init__()

        assert model_name != None or bert_model != None
        if bert_model == None:
            bert_model = model_loader(BertModel, model_name)

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
