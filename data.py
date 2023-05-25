import csv
import json
from transformers import BertTokenizerFast
from tqdm import tqdm
import torch
import random
def read_json(data_path):
    with open(data_path,"r") as fp:
            data = json.load(fp)
    return data
def write_json(data_path,data):
    with open(data_path,"w") as fp:
            json.dump(data,fp)

class DataSequence(torch.utils.data.Dataset):
    def __init__(self, texts,labels):
        self.tokenizer=BertTokenizerFast.from_pretrained('path/to/save/tokenizer')
        self.texts = [self.tokenizer(" ".join(i), padding='max_length',
                            max_length=512, truncation=True,
                            return_tensors="pt")  for i in texts]
        # 对齐标签
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)
        return batch_data, batch_labels

class DataSequenceTest(torch.utils.data.Dataset):
    def __init__(self, texts,lengths,tag):
        self.tokenizer=BertTokenizerFast.from_pretrained('path/to/save/tokenizer')
        self.texts = [self.tokenizer(" ".join(i), padding='max_length',
                            max_length=512, truncation=True,
                            return_tensors="pt")  for i in texts]
        self.lengths = lengths
        self.tag = tag
    def __len__(self):
        return len(self.lengths)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_lengths(self, idx):
        return self.lengths[idx]
    
    def get_batch_tag(self, idx):
        return self.tag[idx]

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        length = self.get_batch_lengths(idx)
        tag = self.get_batch_tag(idx)
        return batch_data,length,tag