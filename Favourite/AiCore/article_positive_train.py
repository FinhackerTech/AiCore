from plistlib import load
from tkinter.filedialog import test
import pandas as pd 
from utils import get_bert_input 
import torch
from tqdm import tqdm
import torch.nn as nn
from pytorch_pretrained import BertModel
from transformers import BertTokenizer, BertConfig
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim
class BertClassifier(nn.Module):
    def __init__(self,num_classes=3):
        super().__init__()
        self.bert =AutoModel.from_pretrained("bert-base-chinese", output_hidden_states=True, return_dict=True)
        self.classifier =  nn.Linear(768,num_classes)
        self.dropout = nn.Dropout(0.5)
    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return self.classifier(self.dropout(bert_output.pooler_output))


def load_data():
    df=pd.read_csv('news.csv')
    df.dropna(axis=0,how='any')
    data=df['context'].values.tolist()

    labels=df['NOP'].values.tolist()
    assert(len(data)==len(labels))
    # soft labels
    # for i in range(len(data)):
    #     if labels[i]=='积极':
    #         labels[i]=[0,1]
    #     elif labels[i]=='中立':
    #         labels[i]=[0.5,0.5]
    #     elif labels[i]=='消极':
    #         labels[i]=[1,0]
    #     else:
    #         assert(1)
    for i in range(len(data)):
        if labels[i]=='积极':
            labels[i]=0
        elif labels[i]=='中立':
            labels[i]=1
        elif labels[i]=='消极':
            labels[i]=2

    train_rate=0.7
    train_len=int(train_rate*len(labels)) 
    train_data=data[:train_len]
    test_data=data[train_len:]
    train_label=labels[:train_len]
    train_label=torch.tensor(train_label,dtype=torch.long)
    
    test_label=labels[train_len:]
    test_label=torch.tensor(test_label,dtype=torch.long)
    
    input_ids = []
    attention_mask = []
    token_type_ids = []

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    for text in train_data:
        
        input_id, attention_mask_id, token_type_id = get_bert_input(text, tokenizer)
        input_ids.append(input_id.unsqueeze(0))
        attention_mask.append(attention_mask_id.unsqueeze(0))
        token_type_ids.append(token_type_id.unsqueeze(0))
    input_ids = torch.cat(input_ids,0)
    attention_mask = torch.cat( attention_mask, 0)
    token_type_ids = torch.cat(token_type_ids, 0)
   
    train_dataset = TensorDataset(input_ids,attention_mask, token_type_ids,train_label)
    
    input_ids = []
    attention_mask = []
    token_type_ids = []

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    for text in test_data :
        input_id, attention_mask_id, token_type_id = get_bert_input(text, tokenizer)
        input_ids.append(input_id.unsqueeze(0))
        attention_mask.append(attention_mask_id.unsqueeze(0))
        token_type_ids.append(token_type_id.unsqueeze(0))
    input_ids = torch.cat(input_ids,0)
    attention_mask = torch.cat( attention_mask, 0)
    token_type_ids = torch.cat(token_type_ids, 0)
    
    test_dataset = TensorDataset(input_ids,attention_mask, token_type_ids,test_label)
    
    return train_dataset,test_dataset
if __name__=='__main__':
    train_dataset,test_dataset=load_data()
    
    model = BertClassifier().cuda()
    
    optimizer =optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    print(optimizer)
    dataloader=DataLoader(train_dataset, batch_size=2, shuffle=False)
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(200):
        tloss=0
        acc=0
        print('================================================================')
        for batch in tqdm(dataloader):
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            token_type_ids = batch[2].cuda()
            labels=batch[3].cuda()
            optimizer.zero_grad()
            output = model(input_ids,attention_mask,token_type_ids)
            loss= criterion(output, labels).cuda()
            loss.backward()
            optimizer.step()
            tloss+=loss.item()
            prd=torch.argmax(output,1)
            acc+=torch.sum(prd == labels)
        tloss=tloss/len(train_dataset)
        
        #acc=acc/len(train_dataset)
        model.train()
        print('================================================================')
        print('epoch {}: loss : {} acc: {}'.format(epoch,tloss,acc))
    torch.save(model, 'bert.bin')