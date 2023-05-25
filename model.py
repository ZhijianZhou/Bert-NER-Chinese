from transformers import BertForTokenClassification,BertModel,BertPreTrainedModel
import torch
import torch.nn as nn
from transformers import BertConfig
import torch.nn.functional as F
seed = 3407
torch.manual_seed(seed)
model_name = "path/to/save/model"
config = BertConfig.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

class CustomBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config,mod):
        super(CustomBertForTokenClassification, self).__init__(config)
        self.bert = bert_model
        self.classifier = nn.Linear(config.hidden_size, 5)
        nn.init.xavier_normal_(self.classifier.weight)
        self.mode = mod
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        probabilities = F.softmax(logits, dim=2)

        if self.mode == "train":
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, 5), labels.view(-1))
            return probabilities, loss
        elif self.mode == "test":
            return probabilities

class CustomBertForTokenClassificationTest(BertPreTrainedModel):
    def __init__(self, config):
        super(CustomBertForTokenClassificationTest, self).__init__(config)
        self.bert = bert_model
        self.classifier = nn.Linear(config.hidden_size, 17)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.classifier(sequence_output)
        probabilities = F.softmax(logits, dim=2)

        return probabilities


class BertCRFModel(torch.nn.Module):
    
    def __init__(self, bert_model, crf_layer):
        super(BertCRFModel, self).__init__()
        self.bert = bert_model
        self.crf = crf_layer
        
    def forward(self, input_ids, mask):
        outputs = self.bert(input_ids, attention_mask=mask)
        sequence_output = outputs.last_hidden_state  # 获取BERT模型的最后一层隐藏状态
        logits = self.crf.decode(sequence_output)  # 序列标注的输出
        return logits