from model.data import *
from model.model import *
import json
from torch.utils.data import DataLoader
from torch.optim import Adam
import TorchCRF
from transformers import BertModel
from transformers import BertConfig
from sklearn.metrics import f1_score
import os
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter  
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
# 设置随机种子
seed = 3407
torch.manual_seed(seed)  # 设置PyTorch随机种子
random.seed(seed)  # 设置Python内置random模块的随机种子
np.random.seed(seed)  # 设置NumPy的随机种子
configs = {}
configs["epoch"] = 5
configs["try"] = "baseline+less_label+decay+more+longtext"
configs["device"] = "cuda:6"
configs["exp"] = "Exp/Baseline+less_label"
configs["lr"] = 2e-5
configs["description"] = "Bert+2e-5+batch2+shuffleT"
configs["batchsize"] = 2
configs["logdir"] = os.path.join(configs["exp"],"log"+str(configs["try"]))
configs["log_frequency"] = 400
configs["decay"] = 100
model_name = "path/to/save/model"
config = BertConfig.from_pretrained(model_name)
loss_fn = nn.MSELoss()
def train_loop(model, train_X, train_Y,val_X,val_Y,device):
    writer = SummaryWriter(configs["logdir"])
    # 定义训练和验证集数据
    train_newset = DataSequence(train_X, train_Y,)
    val_newset = DataSequence(val_X,val_Y)
    # 批量获取训练和验证集数据
    train_newloader = DataLoader(train_newset, num_workers=4, batch_size=configs["batchsize"], shuffle=True)
    val_newloader = DataLoader(val_newset, num_workers=4, batch_size=1)
    gamma = 0.9  

# 创建学习率调度器
    
    lr_bert = 2e-5
    lr_classifier = 1e-4

    # prepare the grouped parameters
    param_groups = [
        {'params': model.bert.parameters(), 'lr': lr_bert},
        {'params': model.classifier.parameters(), 'lr': lr_classifier}
        # {'params': model.classifier2.parameters(), 'lr': lr_classifier},
        # {'params': model.classifier3.parameters(), 'lr': lr_classifier},
    ]

    # pass the parameter groups to the optimizer
    optimizer = Adam(param_groups)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    model = model.to(device)
    # 开始训练循环
    
    count = 0
    best_acc = 0
    best_loss = 1000
    batch_loss = 0
    batch_acc = 0
    print("start train")
    for epoch_num in range(EPOCHS):
        total_acc_train = 0
        total_loss_train = 0
        # 训练模型
        model.train()
        # 按批量循环训练模型
        for train_new, train_label in tqdm(train_newloader):
            # 从train_new中获取mask和input_id
            train_label = train_label[0].to(device)
            mask = train_new['attention_mask'][0].to(device)
            input_id = train_new['input_ids'][0].to(device)
            # 梯度清零！！
            optimizer.zero_grad()
            # 输入模型训练结果：损失及分类概率
            logits,loss = model(input_id, mask,train_label)
            # 过滤掉特殊token及padding的token
            logits_clean = logits[0][train_label != -100]
            label_clean = train_label[train_label != -100]
            predictions = logits_clean.argmax(dim=1)
            
            acc = (predictions == label_clean).float().mean()
            total_acc_train += acc
            total_loss_train += loss.item()
            ## record loss and accuracy
            batch_loss += loss.item()
            batch_acc += acc
            loss.backward()
            optimizer.step()
            count += 2
            if count % configs["decay"] == 0:
                scheduler.step()
            if count % configs["log_frequency"] == 0:
                result = evaluate_model(model,val_newloader,device)
                writer.add_scalar("train_loss", batch_loss/configs["log_frequency"], global_step=count, walltime=True)
                writer.add_scalar("train_acc", batch_acc*2/configs["log_frequency"], global_step=count, walltime=True)
                writer.add_scalar("val_loss", result["loss"], global_step=count, walltime=True)
                writer.add_scalar("val_acc", result["accuracy"], global_step=count, walltime=True)
                writer.add_scalar("f1_score", result["f1_scores"], global_step=count, walltime=True)
                batch_loss = 0
                batch_acc = 0
        torch.save(model.state_dict(), os.path.join(configs["exp"],str(configs["try"])+"_model_"+str(epoch_num)+".pt"))

def evaluate_model(model, val_newloader, device):
    model.eval()
    total_f1_score_val = 0
    total_acc_val = 0
    total_loss_val = 0
    real_label = []
    real_predict = []
    for val_new, val_label in val_newloader:
        # 批量获取验证数据
        val_label = val_label[0].to(device)
        mask = val_new['attention_mask'][0].to(device)
        input_id = val_new['input_ids'][0].to(device)
        # 输出模型预测结果
        logits,loss = model(input_id, mask,val_label)
        # 清楚无效token对应的结果
        logits_clean = logits[0][val_label != -100]
        label_clean = val_label[val_label != -100]
        # 获取概率值最大的预测
        predictions = logits_clean.argmax(dim=1)
        real_label += label_clean.cpu()
        real_predict +=  predictions.cpu()  
        f1_s = f1_score(predictions.cpu().numpy(),label_clean.cpu().numpy(), average='macro') 
        # 计算精度
        acc = (predictions == label_clean).float().mean()
        total_acc_val += acc
        total_loss_val += loss.item()
        total_f1_score_val += f1_s

    f1_s_all = f1_score(real_predict,real_label, average='macro')

    return {
        "loss": total_loss_val / len(val_newloader.dataset),
        "accuracy": total_acc_val / len(val_newloader.dataset),
        "f1_scores": f1_s_all
    }
              
if __name__ == "__main__":
      device = torch.device(configs["device"])
      if os.path.exists(configs["exp"]) :
          pass
      else:
          os.makedirs(configs["exp"])
      if os.path.exists(configs["logdir"]) :
          pass
      else:
          os.makedirs(configs["logdir"])
      num_labels = 5
      LEARNING_RATE = configs["lr"]
      EPOCHS = configs["epoch"]
      model = CustomBertForTokenClassification(config,"train")
      train_X = read_json("data/train_new/x.json")
      train_Y= read_json("data/train_new/labels.json")
      val_X = read_json("data/val_new/x.json")
      val_Y= read_json("data/val_new/labels.json")
      train_loop(model, train_X+val_X, train_Y+val_Y,val_X,val_Y,device)
      