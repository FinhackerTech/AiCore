import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from utils import get_k_neighbors
import torch.nn as nn
from transformers import  BertModel
from transformers import BertTokenizer, BertConfig
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def get_bert_iterator_batch(data_path, batch_size=32):
    '''
    return a interator of text
    each data is a list of string whose lenth is batch_size
    '''
    df = pd.read_csv(data_path)
    keras_bert_iter =iter(df['MAINBUSSINESS'].values.tolist())
    continue_iterator = True
    while continue_iterator:
        data_list = []
        for _ in range(batch_size):
            try:
                data = next(keras_bert_iter)
                data_list.append(data)
            except StopIteration:
                continue_iterator = False

        text_list = []
        for data in data_list:
            text_list.append(data)
        yield text_list
    return False






class BertClassifier(nn.Module):
    def __init__(self, bert_config):
        super().__init__()
        self.bert = BertModel(config=bert_config)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return bert_output[1]
    
def get_bert_input(text, tokenizer, max_len=512):
    '''
        生成单个句子的BERT模型的三个输入
        参数:   
            text: 文本(单个句子)
            tokenizer: 分词器
            max_len: 文本分词后的最大长度
        返回值:
            input_id, attention_mask, token_type_id
    '''
    cls_token = '[CLS]'
    sep_token = '[SEP]'

    word_piece_list = tokenizer.tokenize(text)  #分词
    input_id = tokenizer.convert_tokens_to_ids(word_piece_list) #把分词结果转成id
    if len(input_id) > max_len-2:   #如果input_id的长度大于max_len，则进行截断操作
        input_id = input_id[:510]
    input_id = tokenizer.build_inputs_with_special_tokens(input_id) #对input_id补上[CLS]、[SEP]

    attention_mask = [] # 注意力的mask，把padding部分给遮蔽掉
    for i in range(len(input_id)):
        attention_mask.append(1)    # 句子的原始部分补1
    while len(attention_mask) < max_len:
        attention_mask.append(0)    # padding部分补0

    while len(input_id) < max_len:  # 如果句子长度小于max_len, 做padding，在句子后面补0
        input_id.append(0)

    token_type_id = [0] * max_len # 第一个句子为0，第二个句子为1，第三个句子为0 ..., 也可记为segment_id

    assert len(input_id) == len(token_type_id) == len(attention_mask)

    return input_id, attention_mask, token_type_id
    

def load_data(data_path):
    '''
    读取数据文件，将数据封装起来返回，便于BERT输入
    输入：
        数据文件名
    输出：
        已经处理好的适合BERT输入的dataset
    '''
    df = pd.read_csv(data_path)
    data =df['MAINBUSSINESS'].values.tolist()
    
    input_ids = []
    attention_mask = []
    token_type_ids = []

    tokenizer = BertTokenizer(vocab_file='./chinese_wwm_pytorch/vocab.txt')

    for text in data:
        input_id, attention_mask_id, token_type_id = get_bert_input(text, tokenizer)
        input_ids.append(input_id)
        attention_mask.append(attention_mask_id)
        token_type_ids.append(token_type_id)
    
    input_ids = torch.tensor([i for i in input_ids], dtype=torch.long)
    attention_mask = torch.tensor([i for i in attention_mask], dtype=torch.long)
    token_type_ids = torch.tensor([i for i in token_type_ids], dtype=torch.long)
    dataset = TensorDataset(input_ids, attention_mask, token_type_ids)
    return dataset

if __name__ == '__main__':
    dataset=load_data('data.csv')
    bert_config = BertConfig.from_pretrained('chinese_wwm_pytorch')
    model = BertClassifier(bert_config)
    model.cuda()
    dataloader=DataLoader(dataset, batch_size=32, shuffle=False)
    model.eval()
    features=[]
    with torch.no_grad():
        
        for batch in tqdm(dataloader):
            input_ids = batch[0].cuda()
            attention_mask = batch[1].cuda()
            token_type_ids = batch[2].cuda()
            output = model(    # forward
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
            )
            features.append(output.cpu().numpy())
    features_np=np.concatenate(features, axis=0)
    print(features_np.shape)
    df = pd.read_csv('data.csv')['ShortName']
    center=0
    print('center name',df[center])
    indexs=get_k_neighbors(features_np,center,3)
    
    for i in indexs:
        print(df[i])