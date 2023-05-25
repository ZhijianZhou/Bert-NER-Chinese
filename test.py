from model.data import *
from model.model import *
import json
from torch.utils.data import DataLoader
from torch.optim import SGD
import TorchCRF
from transformers import BertModel
from transformers import BertConfig
from transformers import BertConfig
from model.utils import get_label_map
import csv
config = {}
config["device"] = "cuda:6"
config["model_name"] = "path/to/save/model"
config["input_model"] = "Exp/Baseline+less_label/baseline+less_label+decay+more_model_3.pt"
config["data_dataset"] = "data/test_data/x.json"
config["corpus_dataset"] = "data/test_data/corpus.json"
config["corpus_tag"] = "data/test_data/corpus_tag.json"
config["config"] = BertConfig.from_pretrained(model_name)
config["outpath"] = "test.csv"
def test(Config):
    device = config["device"]
    model = CustomBertForTokenClassification(config["config"],mod="test")
    model.load_state_dict(torch.load(config["input_model"]))
    test_X = read_json(config["data_dataset"])
    corpus = read_json(config["corpus_dataset"])
    corpus_tag = read_json(config["corpus_tag"])
    corpus_length = [len(sublist) for sublist in corpus]
    test_dataset= DataSequenceTest(test_X,corpus_length,corpus_tag)
    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=1, shuffle=False)
    model = model.to(device)
    predict = []
    labels = []
    
    for test_data,lengh,tags in tqdm(test_dataloader):
        mask = test_data['attention_mask'][0].to(device)
        input_id = test_data['input_ids'][0].to(device)
        logits = model(input_id, mask,labels)
        logits_clean = logits[0][1:lengh+1]
        predictions = logits_clean.argmax(dim=1)
        last_tag = -1
        true_label = []
        for prediction,tag in zip(predictions,tags):
            if tag*last_tag == 1:
                continue
            else:
                true_label.append(prediction.item())
            last_tag = tag
        predict.append(true_label)
    return predict
if __name__ == "__main__":
    labels_to_ids,ids_to_labels = get_label_map("data/train_new/y.json")
    test_X = read_json(config["data_dataset"])
    predict = test(Config=config)
    with open(config["outpath"], 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["id","expected"])
        for sentence,predicts in zip(test_X,predict):
            for words,label in zip(sentence,predicts):
                writer.writerow(["".join(words),"".join(ids_to_labels[label])])
            