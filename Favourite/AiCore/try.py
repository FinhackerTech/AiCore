import pandas as pd
import torch
import jieba
import numpy as np
from tqdm import tqdm, trange
from utils import get_k_neighbors
import torch.nn as nn
from transformers import  BertModel
from transformers import BertTokenizer, BertConfig
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from load_data import get_bert_iterator_batch
epoch = 10
embedding_dim = 100
def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

# 初始化矩阵
torch.manual_seed(1)


# 加载停用词词表
def load_stop_words():
    """
        停用词是指在信息检索中，
        为节省存储空间和提高搜索效率，
        在处理自然语言数据（或文本）之前或之后
        会自动过滤掉某些字或词
    """
    with open('data/stopwords.txt', "r", encoding="utf-8") as f:
        return f.read().split("\n")


# 加载文本,切词
def cut_words():
    stop_words = load_stop_words()
    with open('data/zh.txt', encoding='utf8') as f:
        allData = f.readlines()
    
    dataloader = get_bert_iterator_batch('data.csv', 1)
    for x in dataloader:
        if x!=[]:
            allData.append(x[0])
    result = []
    for words in allData:
        c_words = jieba.lcut(words)
        result.append([word for word in c_words if word not in stop_words])
    return result


# 用一个集合存储所有的词
wordList = []
# 调用切词方法
data = cut_words()
count = 0
for words in data:
    for word in words:
        if word not in wordList:
            wordList.append(word)
print("wordList=", wordList)

raw_text = wordList
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

# cbow那个词表，即{[w1,w2,w4,w5],"label"}这样形式
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
                raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

print(data[:5])

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, 128)
        self.output = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1)
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        nll_prob = F.log_softmax(out, dim=-1)
        return nll_prob
    

if __name__ == '__main__':
    stop_words = load_stop_words()
    model = CBOW(vocab_size, embedding_dim).cuda()
    
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # 存储损失的集合
    losses = []
    """
        负对数似然损失函数，用于处理多分类问题，输入是对数化的概率值。
        对于包含N NN个样本的batch数据 D ( x , y ) D(x, y)D(x,y)，x xx 是神经网络的输出，
        进行了归一化和对数化处理。y yy是样本对应的类别标签，每个样本可能是C种类别中的一个。
    """
    loss_function = nn.NLLLoss()

    for epoch in trange(epoch):
        total_loss = 0
        for context, target in tqdm(data):
            # 把训练集的上下文和标签都放到GPU中
            context_vector = make_context_vector(context, word_to_idx).cuda()
            target = torch.tensor([word_to_idx[target]]).cuda()
            # print("context_vector=", context_vector)
            # 梯度清零
            model.zero_grad()
            # 开始前向传播
            train_predict = model(context_vector).cuda()  # 这里要从cuda里取出，不然报设备不一致错误
            loss = loss_function(train_predict, target)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)
    print("losses-=", losses)
    
    dataloader = get_bert_iterator_batch('data.csv', 1)
    features = []
    with torch.no_grad():
        for context in tqdm(dataloader):
            if context==[]:
                continue
            c_words = jieba.lcut(context[0])
            result = [word for word in c_words if (word not in stop_words)]
            print('result:',result,"context_vector",context_vector)
            context_vector = make_context_vector(result, word_to_idx).cuda()
            
            # 预测的值
            predict = model(context_vector).data.cpu().numpy()
            features.append(predict)
    # # 测试一下，用['present', 'food', 'can', 'specifically']这个上下预测一下模型，正确答案是‘surplus’
    # context = ['粮食', '出现', '过剩', '恰好']
    # # 这个变量要放到gpu中，不然又要报设备不一致错误，因为只有把这个数据 同cuda里训练好的数据比较，再能出结果。。很好理解吧
    # context_vector = make_context_vector(context, word_to_idx).cuda()
    # # 预测的值
    # predict = model(context_vector).data.cpu().numpy()
    # print('Raw text: {}\n'.format(' '.join(raw_text)))
    # print('Test Context: {}\n'.format(context))
    # max_idx = np.argmax(predict)
    # # 输出预测的值
    # print('Prediction: {}'.format(idx_to_word[max_idx]))

    # 获取词向量，这个Embedding就是我们需要的词向量，他只是一个模型的一个中间过程
    # print("CBOW embedding'weight=", model.embeddings.weight)
    # W = model.embeddings.weight.cpu().detach().numpy()

    # # 生成词嵌入字典，即{单词1:词向量1,单词2:词向量2...}的格式
    # word_2_vec = {}
    # for word in word_to_idx.keys():
    #     # 词向量矩阵中某个词的索引所对应的那一列即为所该词的词向量
    #     word_2_vec[word] = W[word_to_idx[word], :]
    # print("word2vec=", word_2_vec)
    
    # dataloader=get_bert_iterator_batch('data.csv', 1)
    # model.eval()
    # features=[]
    # with torch.no_grad():
        
    #     for context in tqdm(dataloader):
    #         context_vector = make_context_vector(context, word_to_idx).cuda()
    #         output = model(context_vector).cuda()
    #         features.append(output.cpu().numpy())
    features_np=np.concatenate(features, axis=0)
    print(features_np.shape)
    np.save('features.mpy',features_np)
    df = pd.read_csv('data.csv')['ShortName']
    center=1
    print('center name',df[center])
    indexs=get_k_neighbors(features_np,center,3)
    # np.save("filename.npy",a)
    # np.load("filename.npy")
    for i in indexs:
        print(df[i])