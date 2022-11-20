import math
from collections import defaultdict

import nltk
import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nltk.corpus import treebank

nltk.download('treebank')

class Vocab:
    def __init__(self,tokens = None):
        # 使用列表存储所有的标记，从而可根据索引值获取相应的标记
        self.idx_to_token = list()
        # 使用字典实现标记到索引值的映射
        self.token_to_idx = dict()

        if tokens is not None:
            if "<unk>" not in tokens:
                tokens = tokens + ["<unk>"]
            for token in tokens:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

            self.unk = self.token_to_idx["<unk>"]

    # 创建词表，text包含若干句子，每个句子由若干标记构成
    @classmethod
    def build(cls,text,min_freq = 1 , reserved_tokens = None):
        # 存储标记及其出现次数的映射字典
        token_freqs = defaultdict(int)
        # 无重复地进行标记
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1

        # 用户自定义的预留标记
        uniq_tokens = ["<unk>"] + (reserved_tokens if reserved_tokens else [])
        uniq_tokens += [token for token,freq in token_freqs.items() if freq >= min_freq and token != "<unk>"]

        return cls(uniq_tokens)

    # 返回词表的大小，即词表中有多少个互不相同的标记
    def __len__(self):
        return len(self.idx_to_token)

    # 查找输入标记对应的索引值
    def __getitem__(self, token):
        return self.token_to_idx.get(token,self.unk)

    # 查找一系列输入标记对应的索引值
    def convert_tokens_to_ids(self,tokens):
        return [self[token] for token in tokens]

    # 查找一系列索引值对应的标记
    def convert_ids_to_tokens(self,indices):
        return [self.idx_to_token[index] for index in indices]

# 根据批次中每个序列长度生成Mask矩阵，以便处理长度不一致的序列，忽略掉比较短的序列的无效部分
def length_to_mask(lengths):
    max_len = torch.max(lengths)
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    return mask

# 创建存储数据的TransformerDataset的子类 → Transformer
class TransformerDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

# 数据处理
def collate_fn(examples):
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = [torch.tensor(ex[1]) for ex in examples]
    # 对batch内的样本进行padding，使其具有相同长度
    inputs = pad_sequence(inputs,batch_first=True,padding_value=vocab["<pad>"])
    targets = pad_sequence(targets,batch_first=True,padding_value=vocab["<pad>"])
    return inputs , lengths , targets , inputs != vocab["<pad>"]

## 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self,d_model,dropout = 0.1,max_len = 512):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0 , max_len , dtype= torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0 , d_model ,2).float() * (-math.log(10000.0) / d_model))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer("pe",pe)

    def forward(self,x):
        x = x + self.pe[:x.size(0),:]
        return x

## 模型
class Transformer(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim,num_class,dim_feedforward = 512,num_head = 2,num_layers = 2,dropout=0.1,max_len = 512,activation:str = "relu"):
        super(Transformer, self).__init__()
        # 词嵌入层
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim,dropout,max_len)
        # 编码层
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim,num_head,dim_feedforward,dropout,activation)
        self.transformer = nn.TransformerEncoder(encoder_layer,num_layers)
        # 输出层
        self.output = nn.Linear(hidden_dim,num_class)

    def forward(self,inputs,lengths):
        inputs = torch.transpose(inputs,0,1)
        hidden_states = self.embeddings(inputs)
        hidden_states = self.position_embedding(hidden_states)
        attention_mask = length_to_mask(lengths) == False
        hidden_states = self.transformer(hidden_states,src_key_padding_mask = attention_mask).transpose(0,1)
        logits = self.output(hidden_states)
        log_probs = torch.log_softmax(logits,dim = -1)
        return log_probs

# 设置超参数
embedding_dim = 128
hidden_dim = 128
batch_size = 32
num_epoch = 5

# 加载数据
def load_treebank():
    sents, postags = zip(*(zip(*sent) for sent in treebank.tagged_sents()))

    vocab = Vocab.build(sents, reserved_tokens=["<pad>"])

    tag_vocab = Vocab.build(postags)

    train_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[:3000], postags[:3000])]
    test_data = [(vocab.convert_tokens_to_ids(sentence), tag_vocab.convert_tokens_to_ids(tags)) for sentence, tags in zip(sents[3000:], postags[3000:])]

    return train_data, test_data, vocab, tag_vocab

train_data , test_data , vocab ,pos_vocab = load_treebank()
train_dataset = TransformerDataset(train_data)
test_dataset = TransformerDataset(test_data)
train_data_loader = DataLoader(train_dataset,batch_size = batch_size,collate_fn=collate_fn,shuffle=True)
test_data_loader = DataLoader(test_dataset,batch_size = 1,collate_fn=collate_fn,shuffle=False)

num_class = len(pos_vocab)

# 加载模型
device = torch.device("cpu")
model = Transformer(len(vocab),embedding_dim,hidden_dim,num_class)
model.to(device)

# 训练过程
nll_loss = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

model.train()
for epoch in range(num_epoch):
    total_loss = 0
    for batch in tqdm(train_data_loader,desc=f"Training Epoch{epoch}"):
        inputs , lengths , targets , mask = [x.to(device) for x in batch]
        log_probs = model(inputs,lengths)
        loss = nll_loss(log_probs[mask],targets[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss:{total_loss:.2f}")

# 测试过程
acc = 0
total = 0
for batch in tqdm(test_data_loader,desc=f"Testing"):
    inputs , lengths , targets , mask = [x.to(device) for x in batch]
    with torch.no_grad():
        output = model(inputs,lengths)
        acc += (output.argmax(dim = -1) == targets)[mask].sum().item()
        total += mask.sum().item()

# 输出在测试集上的准确率
print(f"Acc:{acc / total:.2f}")

