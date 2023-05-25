import csv
import json
from transformers import BertTokenizerFast
from tqdm import tqdm
import string
import unicodedata
import copy
from model.data import *
from transformers import BertModel,BertTokenizer
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import csv


seen_words = set()
def trans_data(df1_path,df2_path):
    df1 = pd.read_csv(df1_path)
    with open(df2_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["word","expected"])
        for i ,j in zip(df1.iloc[:,0],df1.iloc[:,1]):
            if j == "O":
                data = [i,"O"]
            else :
                data = [i,j[2:5]]
            writer.writerow(data)
def convert_to_halfwidth(text):
    """
    将全角数字字符转换为半角数字字符
    """
    halfwidth_text = ''
    for char in text:
        if unicodedata.east_asian_width(char) == 'F':  # 判断是否为全角字符
            halfwidth_char = unicodedata.normalize('NFKC', char)  # 将全角字符转换为半角字符
        else:
            halfwidth_char = char
        halfwidth_text += halfwidth_char
    return halfwidth_text
# def convert_to_halfwidth(text):
#     """
#     将全角数字字符转换为半角数字字符
#     """
#     halfwidth_text = ''
#     for char in text:
#         if unicodedata.east_asian_width(char) == 'F':  # 判断是否为全角字符
#             halfwidth_char = chr(ord(char) - 0xFEE0)  # 将全角字符转换为半角字符
#         else:
#             halfwidth_char = char
#         halfwidth_text += halfwidth_char
#     return halfwidth_text
def generate_corpus(dataset):
    with open(dataset,"r") as fp:
        count = 0
        reader = csv.reader(fp)
        x = []
        y = []
        text = []
        labels = []
        c = 0 
        last_length = 50
        for row in reader:
            if c == 0 :
                c+=1
                continue
            c += 1
            words = row[0]
            tag = row[1]
            real_words = copy.copy(words)
            real_words = convert_to_halfwidth(real_words)
            text.append(real_words)
            
            labels.append(tag)
            if (words == "。" or words == "．" or words == "！") and len(text) > 50 and len(text) < 500:
                count += 1
                if len(text) < 50:
                    print(c)
                    print("第",count,"句号有 ",len(text)," 个词")
                x.append(text)
                y.append(labels)
                text = []
                labels = []
    return x,y
def generate_corpus_test(dataset):
    with open(dataset,"r") as fp:
        count = 0
        reader = csv.reader(fp)
        x = []
        text = []
        c = 0 
        for row in reader:
            if c == 0 :
                c+=1
                continue
            c += 1
            words = row[1]
            real_words = copy.copy(words)
            real_words = convert_to_halfwidth(real_words)
            text.append(real_words)
            if words == "。" or words == "．" or words == "！" and len(text) > 50 and len(text) < 500:
                count += 1
                if len(text) > 100:
                    print(c)
                    print("第",count,"句号有 ",len(text)," 个词")
                x.append(text)
                text = []
    return x
def get_label_map(dataset):
    ## 获取标签的映射 
    with open(dataset,"r",encoding = "utf-8") as fp:
        data = json.load(fp)
    unique_labels = set()
    for lb in data:
        [unique_labels.add(i) for i in lb if i not in unique_labels]
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    return labels_to_ids,ids_to_labels
def generate_help_data_test(mod,model_path):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    data_x = read_json(os.path.join(mod,"x.json"))
    corpus = []
    corpus_tag = []
    for i in tqdm(range(0,len(data_x))):
        flag = 1
        sentences = []
        sentences_tag = []
        for j in data_x[i]:
            text = j
            text_tokenized = tokenizer(text,add_special_tokens=False, padding=False) 
            LsA = tokenizer.convert_ids_to_tokens(text_tokenized["input_ids"])
            sentences += LsA
            n = len(LsA)
            sentences_tag += n*[flag]
            flag = flag*(-1)
        corpus.append(sentences)
        corpus_tag.append(sentences_tag)
    write_json(os.path.join(mod,"corpus.json"),corpus)
    write_json(os.path.join(mod,"corpus_tag.json"),corpus_tag)
def generate_help_data(mod,model_path,labels_to_ids):
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    data_x = read_json(os.path.join("data",mod,"x.json"))
    data_y = read_json(os.path.join("data",mod,"y.json"))
    corpus = []
    corpus_label = []
    corpus_tag = []
    for i in tqdm(range(0,len(data_x))):
        flag = 1
        sentences = []
        sentences_label = []
        sentences_tag = []
        for j,label in zip(data_x[i],data_y[i]):
            text = j
            text_tokenized = tokenizer(text,add_special_tokens=False, padding=False) 
            LsA = tokenizer.convert_ids_to_tokens(text_tokenized["input_ids"])
            sentences += LsA
            n = len(LsA)
            
            sentences_label += n*[[labels_to_ids[label]]]
            sentences_tag += n*[flag]
            flag = flag*(-1)
        corpus.append(sentences)
        corpus_label.append(sentences_label)
        corpus_tag.append(sentences_tag)
    write_json(os.path.join("data",mod,"corpus.json"),corpus)
    write_json(os.path.join("data",mod,"corpus_label.json"),corpus_label)
    write_json(os.path.join("data",mod,"corpus_tag.json"),corpus_tag)
def align_label(mod,model_path):
    data_set = os.path.join("data",mod)
    x = read_json(os.path.join(data_set,"x.json"))
    y = read_json(os.path.join(data_set,"y.json"))
    corpus = read_json(os.path.join(data_set,"corpus.json"))
    corpus_label = read_json(os.path.join(data_set,"corpus_label.json"))
    corpus_tag = read_json(os.path.join(data_set,"corpus_tag.json"))
    labels = []
    tags = []
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    for i in tqdm(range(0,len(x))):
        sentence = " ".join(x[i])
        label = []
        tag = []
        text_tokenized = tokenizer(sentence, padding='max_length',
                                max_length=512, truncation=True,
                                return_tensors="pt")
        LsA = tokenizer.convert_ids_to_tokens(text_tokenized["input_ids"][0])
        count = 0
        cc = 0
        flag = 1
        for word in LsA:
            if word == "[CLS]":
                label.append(-100)
                tag.append(0)
                continue
            elif word == "[SEP]":
                label.append(-100)
                tag.append(0)
                continue
            elif word == "[PAD]":
                label.append(-100)
                tag.append(0)
                continue
            elif count < len(corpus[i]) and word != corpus[i][count]:
                print("error")
            else :
                label.append(corpus_label[i][count][0])
                tag.append(corpus_tag[i][count])
                count += 1
        labels.append(label)
        tags.append(tag)
    write_json(os.path.join(data_set,"labels.json"),labels)
    write_json(os.path.join(data_set,"tags.json"),tags)
def check_word(word):
    if word in seen_words:
        return False
    else:
        seen_words.add(word)
        return True
def contains_english_letter(string):
    pattern = r'[a-zA-Z]'
    return bool(re.search(pattern, string))
from collections import Counter

def generate_vocab(sentences):
    vocab_counter = Counter()
    for sentence in sentences:
        # words = sentence.split()  # 使用空格分割词语
        vocab_counter.update(sentence)
    
    vocab = list(vocab_counter.keys())
    return vocab
def plot_sublist_lengths(x,y, save_path=None):
    # 统计每个子列表的长度
    lengths = []
    for sentence,sublist in zip(x,y):
        length =  len(sublist)
        if length <512:
            lengths.append(length)
            if length > 100:
                print("".join(sentence))
    # plt.figure(dpi=300,figsize=(12,4))
    # # 绘制分布图
    # plt.hist(lengths, bins='auto')
    # plt.xlabel('Length')
    # plt.ylabel('Count')
    # plt.title('Sentence Length Distribution')

    # # 保存图像
    # if save_path is not None:
    #     plt.savefig(save_path)
    # else:
    #     plt.show()

def plot_label_distribution(data, save_path, labels_to_ids, ids_to_labels):
    # 统计每个标签的数量
    label_counts = {}
    for sublist in data:
        for label in sublist:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
    labels = list(label_counts.keys())
    counts = list(label_counts.values())
    print(label_counts)
    # 绘制分布图
    plt.figure(dpi=300,figsize=(12,6))
    x_coords = [0.8*x for x in range(len(counts))]
    bars = plt.bar(x_coords, counts,width=0.5)
    plt.xlabel('Label')
    plt.ylabel('Count')
    # plt.title('Label Distribution')
    plt.xticks(x_coords, labels)  # 使x轴的刻度标签与bar对齐
    plt.title("Label's count")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/6.0, yval, int(yval), va='bottom')  # va: vertical alignment y轴对齐方式
    
    # 保存图像
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
if __name__ == "__main__":
    # trans_data("data/dev.csv","data/new_data/dev.csv")
    # train_x,train_y = generate_corpus("data/new_data/train.csv")
    # with open("data/train_new/x.json","w") as fp :
    #     json.dump(train_x,fp)
    # with open("data/train_new/y.json","w") as fp :
    #     json.dump(train_y,fp)
    # train_x,train_y = generate_corpus("data/new_data/dev.csv")
    # with open("data/val_new/x.json","w") as fp :
    #     json.dump(train_x,fp)
    # with open("data/val_new/y.json","w") as fp :
    #     json.dump(train_y,fp)

    train_x,train_y = generate_corpus("data/test.csv")
    with open("data/test_new/x.json","w") as fp :
        json.dump(train_x,fp)
    with open("data/test_new/y.json","w") as fp :
        json.dump(train_y,fp)
    generate_help_data_test("test_new","path/to/save/tokenizer")
    # labels_to_ids,ids_to_labels = get_label_map("data/train_new/y.json")
    # data = read_json("data/train_data/y.json")
    # plot_label_distribution(data, "new2.jpg",labels_to_ids,ids_to_labels)
    # generate_help_data("val_new","path/to/save/tokenizer",labels_to_ids)
    # align_label("val_new","path/to/save/tokenizer")
    # generate_help_data("train_new","path/to/save/tokenizer",labels_to_ids)
    # align_label("train_new","path/to/save/tokenizer")
    # y = read_json("data/train_data/y.json")
    # x = read_json("data/train_data/x.json")
    # plot_sublist_lengths(x,y,"new.jpg")




    # x = read_json("data/train_data/x.json")
    # y = read_json("data/train_data/y.json")
    # new_x = []
    # new_y = []
    # for sublist_x,sublist_y in zip(x,y):
    #     for label in sublist_y:
    #         if label  != "O":
    #             flag = 1
    #     if flag == 1:
    #         new_x.append(sublist_x)
    #         new_y.append(sublist_y)
    #     else:
    #         continue
    #     flag = 0
    # write_json("data/train_data_plus/x.json",new_x)
    # write_json("data/train_data_plus/y.json",new_y)




    
            

    